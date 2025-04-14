# Rdkit import should be first, do not move it
import utils
from qm9 import dataset
from qm9.models import get_model
import torch
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.utils import  compute_mean_mad
from GeoRCG_models.rep_samplers import *
from qm9.models import get_model, DistributionProperty
from qm9.rdkit_functions import BasicMolecularMetrics, preprocess_generated_molecules


import logging
logger = logging.getLogger(name=__name__)


def analyze_all_metrics(molecules, dataset_info):
    
    processed_list = preprocess_generated_molecules(molecules)
    metrics = BasicMolecularMetrics(dataset_info)
    stability_dict = metrics.compute_stability(processed_list)
    rdkit_metrics = metrics.evaluate(processed_list)[0]
    pb_ratio = metrics.compute_posebusters(processed_list)
    _, relaxed_validity = metrics.compute_relaxed_validity(processed_list)
    
    # Calculate energy and strain metrics introduced in Semla
    energy = metrics.compute_energy(processed_list)
    energy_per_atom = metrics.compute_energy_per_atom(processed_list)
    strain = metrics.compute_strain_energy(processed_list)
    strain_per_atom = metrics.compute_strain_energy_per_atom(processed_list)
    
    calculated_metrics = {
        "molecule_stable": stability_dict['mol_stable'],
        "atom_stable": stability_dict['atm_stable'],
        "validity": rdkit_metrics[0],
        "uniqueness": rdkit_metrics[1],
        "novelty": rdkit_metrics[2],
        "pb_valid": pb_ratio,
        "relaxed_validity": relaxed_validity,
        "energy": energy,
        "energy_per_atom": energy_per_atom,
        "strain": strain,
        "strain_per_atom": strain_per_atom,
    }
    
    # Process the metric
    calculated_metrics["valid&unique"] = calculated_metrics["validity"] * calculated_metrics["uniqueness"]
    calculated_metrics["valid&unique&novelty"] = calculated_metrics["validity"] * calculated_metrics["uniqueness"] * calculated_metrics["novelty"]
    calculated_metrics["relaxed_valid&unique"] = calculated_metrics["relaxed_validity"] * calculated_metrics["uniqueness"] * calculated_metrics["novelty"]
    calculated_metrics["relaxed_valid&unique&novelty"] = calculated_metrics["relaxed_validity"] * calculated_metrics["uniqueness"] * calculated_metrics["novelty"]

    
    return calculated_metrics

def prepare_model_and_dataset_info(
    eval_args, device="cuda", only_dataset_info=False
    ):
    if only_dataset_info:
        return None, None, None, None, get_dataset_info(eval_args.dataset, eval_args.remove_h)
    
    
    # For evaluation of conditional RDM
    if eval_args.property is not None:
        conditioning = [eval_args.property]
        dataset_args = {
            "dataset": "qm9_second_half",
            "conditioning": conditioning,
            "include_charges": True,
            "world_size": 1,
            "rank": 0,
            "filter_n_atoms": None,
            "remove_h": False,
            "batch_size": 128,
            "num_workers": 4,
            "datadir": "./data"
        }
        dataset_args = OmegaConf.create(dataset_args)
        
        dataloaders_condition, _ = dataset.retrieve_dataloaders(dataset_args)
        dataset_condition_info = get_dataset_info(dataset_args.dataset, dataset_args.remove_h)
        prop_dist_condition = DistributionProperty(dataloaders_condition['train'], conditioning)
        property_norms = compute_mean_mad(dataloaders_condition, conditioning, dataset_args.dataset)
        prop_dist_condition.set_normalizer(property_norms)


    # Load the gen model args
    assert eval_args.gen_model_path is not None
    with open(join(eval_args.gen_model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
    gen_args = args.gen_args
    if gen_args.get("attn_dropout", None) is None:
        gen_args.attn_dropout = 0.
    # Modify the gen_args for sampling
    if eval_args.cfg is not None:
        print(f"Warning: Changing gen_args.cfg to {eval_args.cfg}")
        gen_args.cfg = eval_args.cfg
    
    # Replace the  args with gen_args
    args = gen_args
    # For retrivial datasets
    gen_args.device = str(device)
    gen_args.world_size = 1
    gen_args.rank = 0
    
    
    utils.create_folders(args)
    # print(f"gen_args: {args}")
    # print(f"eval_args: {eval_args}")
    

    # Retrieve QM9 dataloaders
    dataloaders, charge_scale = dataset.retrieve_dataloaders(args)
    # Handle midi evaluation
    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    
    # Load gen model
    generative_model, nodes_dist, prop_dist = get_model(args, device, dataset_info, dataloaders['train'])
    assert prop_dist is None, "We only use unconditional gen."
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
        
    if eval_args.property is not None:
        prop_dist = prop_dist_condition
        
    generative_model.to(device)
    generative_model.eval()

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(eval_args.gen_model_path, fn), map_location=device)
    load_profiles = generative_model.load_state_dict(flow_state_dict, strict=False)
    print(load_profiles)
    assert len(load_profiles.missing_keys) == 0
    if len(load_profiles.unexpected_keys) != 0:
        print(f"Warning: Loading generative model, but meet unexpected keys: {load_profiles.unexpected_keys}")
    
    rep_sampler = initilize_rep_sampler(eval_args, device, args)
        
    from GeoRCG_models.wrapper import SelfConditionWrappedSampler
    self_conditioned_sampler = SelfConditionWrappedSampler(
        gen_sampler=generative_model,
        rdm_sampler=rep_sampler
    )
    
    return self_conditioned_sampler, gen_args, nodes_dist, prop_dist, dataset_info



