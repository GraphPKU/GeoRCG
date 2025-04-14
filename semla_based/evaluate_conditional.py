import argparse
import sys
sys.path.append(".")
from pathlib import Path
from functools import partial

import torch
import numpy as np
import lightning as L

import semlaflow.scriptutil as util
from semlaflow.flowmodels.fm_property import Integrator, MolecularCFM_property
from semlaflow.flowmodels.semla import EquiInvDynamics, SemlaGenerator

from semlaflow.data.datasets import GeometricDataset
from semlaflow.data.datamodules import GeometricInterpolantDM
from semlaflow.data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from semlaflow.flowmodels.encoders import initialize_encoder
from semlaflow.flowmodels.rep_samplers import *
import hydra

import time

from os.path import join
from qm9.property_prediction import main_qm9_prop
import pickle


def get_classifier(dir_path='', device='cpu'):
    with open(join(dir_path, 'args.pickle'), 'rb') as f:
        args_classifier = pickle.load(f)
    args_classifier.device = device
    args_classifier.model_name = 'egnn'
    classifier = main_qm9_prop.get_model(args_classifier)
    classifier_state_dict = torch.load(join(dir_path, 'best_checkpoint.npy'), map_location=torch.device('cpu'))
    classifier.load_state_dict(classifier_state_dict)

    return classifier
qm9_to_eV = {'U0': 1., 'U': 1., 'G': 1., 'H': 1., 'zpve': 1., 'gap': 1., 'homo': 1., 'lumo': 1.}

def load_model(args, vocab, property_norms):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]

    hparams["compile_model"] = False
    hparams["integration-steps"] = args.integration_steps
    hparams["sampling_strategy"] = args.ode_sampling_strategy
    
    # NOTE: These parameters are freely set during evaluation.
    hparams["cfg_coef"] = args.cfg_coef
    # NOTE: Some parameters are not saved in the checkpoint, so we need to set them manually. We may delete this part if we save all the hyperparameters in the future.
    try:
        hparams["use_gate"]
    except:
        hparams["use_gate"] = args.use_gate
    # NOTE: During evaluation, this hyperparameter is not used, so we set it to 0.1 for simplicity
    hparams["rep_dropout_prob"] = 0.1 # During evaluation, this hyperparameter is not used, so we set it to 0.1 for simplicity
    hparams["noise_sigma"] = 0.1 # During evaluation, this hyperparameter is not used, so we set it to 0.1 for simplicity
    hparams["dropout"] = 0.1 
    hparams["sparse_rep_condition"] = hparams.get("sparse_rep_condition", args.sparse_rep_condition)
    
    hparams["cond_type"] = hparams.get("cond_type", args.cond_type)
    hparams["time_condition"] = hparams.get("time_condition", args.time_condition)
    
    # # For reproduction
    # if hparams.get("d_rep") is None:
    #     hparams["d_rep"] = 512
    # if hparams.get("attn_block_num") is None:
    #     hparams["attn_block_num"] = 1
    # if hparams.get("original") is None:
    #     hparams["original"] = True
    # if hparams.get("use_gate") is None:
    #     hparams["use_gate"] = True
        
    n_bond_types = util.get_n_bond_types(hparams["integration-type-strategy"])

    # Set default arch to semla if nothing has been saved
    if hparams.get("architecture") is None:
        hparams["architecture"] = "semla"

    if hparams["architecture"] == "semla":
        dynamics = EquiInvDynamics(
            hparams["d_model"],
            hparams["d_message"],
            hparams["n_coord_sets"],
            hparams["n_layers"],
            n_attn_heads=hparams["n_attn_heads"],
            d_message_hidden=hparams["d_message_hidden"],
            d_edge=hparams["d_edge"],
            self_cond=hparams["self_cond"],
            coord_norm=hparams["coord_norm"],
            
            d_rep=hparams["d_rep"],
            attn_block_num=hparams["attn_block_num"],
            dropout=hparams["dropout"],
            original=hparams["original"], 
            use_gate=hparams["use_gate"],
            sparse_rep_condition=hparams["sparse_rep_condition"],
            cond_type=hparams["cond_type"],
            
            
            
        )
        egnn_gen = SemlaGenerator(
            hparams["d_model"],
            dynamics,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            self_cond=hparams["self_cond"],
            size_emb=hparams["size_emb"],
            max_atoms=hparams["max_atoms"],
            property_condition=hparams["property_condition"],
        )

    elif hparams["architecture"] == "eqgat":
        from semlaflow.models.eqgat import EqgatGenerator

        egnn_gen = EqgatGenerator(
            hparams["d_model"],
            hparams["n_layers"],
            hparams["n_equi_feats"],
            vocab.size,
            hparams["n_atom_feats"],
            hparams["d_edge"],
            hparams["n_edge_types"],
            d_rep=hparams["d_rep"]
        )

    elif hparams["architecture"] == "egnn":
        from semlaflow.models.egnn import VanillaEgnnGenerator

        n_layers = args.n_layers if hparams.get("n_layers") is None else hparams["n_layers"]
        if n_layers is None:
            raise ValueError("No hparam for n_layers was saved, use script arg to provide n_layers")

        egnn_gen = VanillaEgnnGenerator(
            hparams["d_model"],
            n_layers,
            vocab.size,
            hparams["n_atom_feats"],
            d_edge=hparams["d_edge"],
            n_edge_types=n_bond_types,
            d_rep=hparams["d_rep"]
        )

    else:
        raise ValueError(f"Unknown architecture hyperparameter.")

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["train-type-interpolation"] == "mask" else None
    bond_mask_index = None

    integrator = Integrator(
        args.integration_steps,
        type_strategy=hparams["integration-type-strategy"],
        bond_strategy=hparams["integration-bond-strategy"],
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        cat_noise_level=args.cat_sampling_noise_level
    )
    # Set up for encoder
    encoder = initialize_encoder(encoder_type=args.encoder_type,
                                 encoder_ckpt_path=args.encoder_path)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()  

    rep_sampler = initilize_rep_sampler(args, args, dataset=hparams["dataset"])

    classifier = get_classifier(args.classifiers_path, args.device)
    property_conversion = 1. if args.property not in qm9_to_eV else qm9_to_eV[args.property]

    fm_model = MolecularCFM_property.load_from_checkpoint(
        args.ckpt_path,
        strict=False,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        encoder=encoder,
        rdm=rep_sampler,
        classifier=classifier,
        property_conversion=property_conversion,
        property_norms=property_norms,
        **hparams,
    )
    return fm_model


def build_dm(args, hparams, vocab):
    if args.dataset == "qm9" or args.dataset == "qm9_second_half":
        coord_std = util.QM9_COORDS_STD_DEV
        bucket_limits = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        bucket_limits = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")
 
    n_bond_types = 5 # By default, we use uniform-sampling strategy which does not contain mask token, so we directly set n_bond_types to 5 for simplicity
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    if args.dataset_split == "train":
        dataset_path = Path(args.data_path) / "train.smol"
    elif args.dataset_split == "val":
        dataset_path = Path(args.data_path) / "val.smol"
    elif args.dataset_split == "test":
        dataset_path = Path(args.data_path) / "test.smol"

    dataset = GeometricDataset.load(dataset_path, transform=transform)
    dataset = dataset.sample(args.n_molecules, replacement=True)
    
    
    assert args.dataset == "qm9_second_half"
    train_dataset = GeometricDataset.load(Path(args.data_path) / "second_half_train.smol", transform=transform)
    
    val_dataset = GeometricDataset.load(Path(args.data_path) / "val.smol", transform=transform)

    type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0] if hparams["val-type-interpolation"] == "mask" else None
    bond_mask_index = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=hparams["val-prior-type-noise"],
        bond_noise=hparams["val-prior-bond-noise"],
        scale_ot=hparams["val-prior-noise-scale-ot"],
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=hparams["val-type-interpolation"],
        bond_interpolation=hparams["val-bond-interpolation"],
        equivariant_ot=False,
        batch_ot=False
    )
    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        dataset,
        args.batch_cost,
        test_interpolant=eval_interpolant,
        bucket_limits=bucket_limits,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False
    )
    return dm


def dm_from_ckpt(args, vocab):
    checkpoint = torch.load(args.ckpt_path)
    hparams = checkpoint["hyper_parameters"]
    dm = build_dm(args, hparams, vocab)
    return dm

from qm9.models import get_model, DistributionProperty_Semla

def evaluate(args, model, dm, metrics, stab_metrics, prop_sampler, conditional_metric):
    # torch.set_float32_matmul_precision("medium")
    results_list = []
    for replicate_index in range(args.n_replicates):
        print(f"Running replicate {replicate_index + 1} out of {args.n_replicates}")
        
        # Pregenerate all reps for the replicate
        s = time.time()
        util.generate_reps_conditional(model, dm, args.integration_steps, args.ode_sampling_strategy, stabilities=True, prop_sampler=prop_sampler)
        print(f"Time to generate {args.n_molecules} reps: {time.time() - s}")
        
        molecules, _, stabilities, properties = util.generate_molecules_conditional(
            model,
            dm,
            args.integration_steps,
            args.ode_sampling_strategy,
            stabilities=True,
            prop_sampler=prop_sampler
        )
        
        
        model.pre_generated_reps_sizes = []
        model.pre_generated_reps = []
        model.pre_generated_reps_properties = []

        print("Calculating metrics...")
        
        
        results = util.calc_metrics_conditional(molecules, metrics, conditional_metric, stab_metrics=stab_metrics, mol_stabs=stabilities, classifier=model.classifier, properties=properties)
        results_list.append(results)
        print(results)

    results_dict = {key: [] for key in results_list[0].keys()}
    for results in results_list:
        for metric, value in results.items():
            results_dict[metric].append(value.item())

    mean_results = {metric: np.mean(values) for metric, values in results_dict.items()}
    std_results = {metric: np.std(values) for metric, values in results_dict.items()}

    return mean_results, std_results, results_dict

def compute_mean_mad_from_dataloader(dataloader):
    
    property_norm = {}
    values = dataloader.dataset._data.properties
    mean = torch.mean(values)
    ma = torch.abs(values - mean)
    mad = torch.mean(ma)
    property_norm['mean'] = mean
    property_norm['mad'] = mad
    return property_norm


@hydra.main(config_path="./configs", config_name="eval_conditional.yaml", version_base="1.3")
def main(args):
    print(f"Running evaluation script for {args.n_replicates} replicates with {args.n_molecules} molecules each...")
    print(f"Using model stored at {args.ckpt_path}")

    if args.n_replicates < 1:
        raise ValueError("n_replicates must be at least 1.")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = dm_from_ckpt(args, vocab)
    print("Datamodule complete.")
    
    
    # Prepare the property sampler
    prop_sampler = DistributionProperty_Semla(dm.train_dataloader(), [args.property])
    property_norms = compute_mean_mad_from_dataloader(dm.val_dataloader())
    prop_sampler.set_normalizer(property_norms)

    print(f"Loading model...")
    model = load_model(args, vocab, property_norms)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, stab_metrics, conditional_metric = util.init_metrics_conditional(args.data_path, model, model.classifier, property_norms["mean"], property_norms["mad"])
    print("Metrics complete.")
    
    



    print("Running evaluation...")
    avg_results, std_results, list_results = evaluate(args, model, dm, metrics, stab_metrics, prop_sampler, conditional_metric)
    print("Evaluation complete.")

    util.print_results(avg_results, std_results=std_results)

    print("All replicate results...")
    print(f"{'Metric':<22}Result")
    print("-" * 30)

    for metric, results_list in list_results.items():
        print(f"{metric:<22}{results_list}")
    print()


if __name__ == "__main__":
    main()
