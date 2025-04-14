"""Util file for Equinv scripts"""

import math
import resource
from pathlib import Path

import torch
import numpy as np
from tqdm import tqdm
from rdkit import RDLogger
from openbabel import pybel
from torchmetrics import MetricCollection

import util.rdkit as smolRD
import util.functional as smolF
import util.metrics as Metrics
from util.tokeniser import Vocabulary
from data.datasets import GeometricDataset
import time

# Declarations to be used in scripts
QM9_COORDS_STD_DEV = 1.723299503326416
GEOM_COORDS_STD_DEV = 2.407038688659668

QM9_BUCKET_LIMITS = [12, 16, 18, 20, 22, 24, 30]
GEOM_DRUGS_BUCKET_LIMITS = [24, 28, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 96, 192]

PROJECT_PREFIX = "equinv"
BOND_MASK_INDEX = 5
COMPILER_CACHE_SIZE = 128


def disable_lib_stdout():
    pybel.ob.obErrorLog.StopLogging()
    RDLogger.DisableLog('rdApp.*')


# Need to ensure the limits are large enough when using OT since lots of preprocessing needs to be done on the batches
# OT seems to cause a problem when there are not enough allowed open FDs
def configure_fs(limit=4096):
    """
    Try to increase the limit on open file descriptors
    If not possible use a different strategy for sharing files in torch
    """

    n_file_resource = resource.RLIMIT_NOFILE
    soft_limit, hard_limit = resource.getrlimit(n_file_resource)

    print(f"Current limits (soft, hard): {(soft_limit, hard_limit)}")

    if limit > soft_limit:
        try:
            print(f"Attempting to increase open file limit to {limit}...")
            resource.setrlimit(n_file_resource, (limit, hard_limit))
            print("Limit changed successfully!")

        except:
            print("Limit change unsuccessful. Using torch file_system file sharing strategy instead.")

            import torch.multiprocessing
            torch.multiprocessing.set_sharing_strategy("file_system")

    else:
        print("Open file limit already sufficiently large.")


# Applies the following transformations to a molecule:
# 1. Scales coordinate values by 1 / coord_std (so that they are standard normal)
# 2. Applies a random rotation to the coordinates
# 3. Removes the centre of mass of the molecule
# 4. Creates a one-hot vector for the atomic numbers of each atom
# 5. Creates a one-hot vector for the bond type for every possible bond
# 6. Encodes charges as non-negative numbers according to encoding map
def mol_transform(molecule, vocab, n_bonds, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    atomic_nums = [int(atomic) for atomic in molecule.atomics.tolist()]
    tokens = [smolRD.PT.symbol_from_atomic(atomic) for atomic in atomic_nums]
    one_hot_atomics = torch.tensor(vocab.indices_from_tokens(tokens, one_hot=True))

    bond_types = smolF.one_hot_encode_tensor(molecule.bond_types, n_bonds)

    charge_idxs = [smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(atomics=one_hot_atomics, bond_types=bond_types, charges=charge_idxs, property=molecule.property, property_name=molecule.property_name)
    return transformed


# When training a distilled model atom types and bonds are already distributions over categoricals
def distill_transform(molecule, coord_std):
    rotation = tuple(np.random.rand(3) * np.pi * 2)
    molecule = molecule.scale(1.0 / coord_std).rotate(rotation).zero_com()

    charge_idxs = [smolRD.CHARGE_IDX_MAP[charge] for charge in molecule.charges.tolist()]
    charge_idxs = torch.tensor(charge_idxs)

    transformed = molecule._copy_with(charges=charge_idxs)
    return transformed


def get_n_bond_types(cat_strategy):
    n_bond_types = len(smolRD.BOND_IDX_MAP.keys()) + 1
    n_bond_types = n_bond_types + 1 if cat_strategy == "mask" else n_bond_types
    return n_bond_types


def build_vocab():
    # Need to make sure PAD has index 0
    special_tokens = ["<PAD>", "<MASK>"]
    core_atoms = ["H", "C", "N", "O", "F", "P", "S", "Cl"]
    other_atoms = ["Br", "B", "Al", "Si", "As", "I", "Hg", "Bi"]
    tokens = special_tokens + core_atoms + other_atoms
    return Vocabulary(tokens)

    

# TODO support multi gpus
def calc_train_steps(dm, epochs, acc_batches):
    dm.setup("train")
    steps_per_epoch = math.ceil(len(dm.train_dataloader()) / acc_batches)
    return steps_per_epoch * epochs


def init_metrics(data_path, model, second_half=False):
    # Load the train data separately from the DM, just to access the list of train SMILES
    if second_half:
        train_path = Path(data_path) / "second_half_train.smol"
    else:
        train_path = Path(data_path) / "train.smol"
    train_dataset = GeometricDataset.load(train_path)
    train_smiles = [mol.str_id for mol in train_dataset]

    print("Creating RDKit mols from training SMILES...")
    train_mols = model.builder.mols_from_smiles(train_smiles, explicit_hs=True)
    train_mols = [mol for mol in train_mols if mol is not None]

    metrics = {
        "validity": Metrics.Validity(),
        "connected-validity": Metrics.Validity(connected=True),
        "uniqueness": Metrics.Uniqueness(),
        "novelty": Metrics.Novelty(train_mols),
        "energy-validity": Metrics.EnergyValidity(),
        "opt-energy-validity": Metrics.EnergyValidity(optimise=True),
        "energy": Metrics.AverageEnergy(),
        "energy-per-atom": Metrics.AverageEnergy(per_atom=True),
        "strain": Metrics.AverageStrainEnergy(),
        "strain-per-atom": Metrics.AverageStrainEnergy(per_atom=True),
        "opt-rmsd": Metrics.AverageOptRmsd()
    }
    stability_metrics = {
        "atom-stability": Metrics.AtomStability(),
        "molecule-stability": Metrics.MoleculeStability()
    }

    metrics = MetricCollection(metrics, compute_groups=False)
    stability_metrics = MetricCollection(stability_metrics, compute_groups=False)

    return metrics, stability_metrics


def init_metrics_conditional(data_path, model, classifier, mean, mad):
    conditional_metric = Metrics.PropertyMetric(classifier, mean, mad)
    metrics, stability_metrics = init_metrics(data_path, model, second_half=True)
    return metrics, stability_metrics, conditional_metric


def generate_molecules(model, dm, steps, strategy, stabilities=False):
    test_dl = dm.test_dataloader()
    model.eval()
    cuda_model = model.to("cuda")

    outputs = []
    s = time.time()
    for batch in tqdm(test_dl):
        batch = {k: (v.cuda() if v is not None else v) for k, v in batch[0].items()}
        output = cuda_model._generate(batch, steps, strategy)
        outputs.append(output)
    print(f"Time taken to generate {len(dm.test_dataset.lengths)} molecules: {time.time() - s}")

    molecules = [cuda_model._generate_mols(output) for output in outputs]
    molecules = [mol for mol_list in molecules for mol in mol_list]

    if not stabilities:
        return molecules, outputs

    stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
    stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
    print(cuda_model.pre_generated_reps_sizes.tolist())
    print((cuda_model.pre_generated_reps_sizes == -1).sum())
    assert torch.all(cuda_model.pre_generated_reps_sizes == -1), "Not all pre-generated reps are used. Please check your pre-generated reps procedure."
    
    
    return molecules, outputs, stabilities


def generate_molecules_conditional(model, dm, steps, strategy, stabilities=False, prop_sampler=None):
    test_dl = dm.test_dataloader()
    model.eval()
    cuda_model = model.to("cuda")

    outputs = []
    properties_list = []
    s = time.time()
    for batch in tqdm(test_dl):
        batch = {k: (v.cuda() if v is not None else v) for k, v in batch[0].items()}
        output, properties = cuda_model._generate(batch, steps, strategy, prop_sampler)
        outputs.append(output)
        properties_list.append(properties.tolist())
    print(f"Time taken to generate {len(dm.test_dataset.lengths)} molecules: {time.time() - s}")

    molecules = [cuda_model._generate_mols(output) for output in outputs]
    molecules = [mol for mol_list in molecules for mol in mol_list]
    properties = [prop for prop_list in properties_list for prop in prop_list]
    properties = torch.tensor(properties).cuda()

    if not stabilities:
        return molecules, outputs

    stabilities = [cuda_model._generate_stabilities(output) for output in outputs]
    stabilities = [mol_stab for mol_stabs in stabilities for mol_stab in mol_stabs]
    assert torch.all(cuda_model.pre_generated_reps_sizes == -1), "Not all pre-generated reps are used. Please check your pre-generated reps procedure."
    
    
    return molecules, outputs, stabilities, properties

def generate_reps(model, dm, steps, strategy, stabilities=False):
    lengths = dm.test_dataset.lengths
    lengths = torch.tensor(lengths).cuda()
    model.eval()
    cuda_model = model.to("cuda")
    # batch_size: 1000
    for i in range(0, len(lengths), 5000):
        cuda_model._generate_reps(lengths[i:i + 5000], steps, strategy)
    cuda_model.pre_generated_reps = torch.cat(cuda_model.pre_generated_reps, dim=0)
    cuda_model.pre_generated_reps_sizes = torch.cat(cuda_model.pre_generated_reps_sizes, dim=0)
    
def generate_reps_conditional(model, dm, steps, strategy, stabilities=False, prop_sampler=None):
    lengths = dm.test_dataset.lengths
    lengths = torch.tensor(lengths).cuda()
    model.eval()
    s = time.time()
    cuda_model = model.to("cuda")
    print(f"Time to move to GPU: {time.time() - s}")
    # batch_size: 1000
    for i in range(0, len(lengths), 5000):
        _lengths = lengths[i:i + 5000]
        _properties = prop_sampler.sample_batch(_lengths)
        cuda_model._generate_reps_conditional(_lengths, steps, strategy, _properties)
    cuda_model.pre_generated_reps = torch.cat(cuda_model.pre_generated_reps, dim=0)
    cuda_model.pre_generated_reps_sizes = torch.cat(cuda_model.pre_generated_reps_sizes, dim=0)
    cuda_model.pre_generated_reps_properties = torch.cat(cuda_model.pre_generated_reps_properties, dim=0).to("cuda")
    


def calc_metrics_(rdkit_mols, metrics, stab_metrics=None, mol_stabs=None):
    metrics.reset()
    metrics.update(rdkit_mols)
    results = metrics.compute()

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_results = stab_metrics.compute()

    results = {
        **results,
        **stab_results
    }
    return results


def calc_metrics_conditional(rdkit_mols, metrics, conditional_metric, stab_metrics=None, mol_stabs=None, classifier=None, properties=None):
    metrics.reset()
    metrics.update(rdkit_mols)
    results = metrics.compute()

    if stab_metrics is None:
        return results

    stab_metrics.reset()
    stab_metrics.update(mol_stabs)
    stab_results = stab_metrics.compute()
    
    conditional_metric.update(rdkit_mols, labels=properties)
    
    

    results = {
        **results,
        **stab_results,
        "property": conditional_metric.compute()
    }
    return results

def print_results(results, std_results=None):
    print()
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, value in results.items():
        result_str = f"{metric:<22}{value:.5f}"
        if std_results is not None:
            std = std_results[metric]
            result_str = f"{result_str} +- {std:.7f}"

        print(result_str)
    print()
