import argparse
import sys
sys.path.append(".")
from pathlib import Path
from functools import partial

import torch
import numpy as np
import lightning as L

import scriptutil as util
from flowmodels.fm import Integrator, MolecularCFM
from flowmodels.semla import EquiInvDynamics, SemlaGenerator

from data.datasets import GeometricDataset
from data.datamodules import GeometricInterpolantDM
from data.interpolate import GeometricInterpolant, GeometricNoiseSampler
from flowmodels.encoders import initialize_encoder
from flowmodels.rep_samplers import initilize_rep_sampler
import hydra

import time

def load_model(args, vocab):
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
            max_atoms=hparams["max_atoms"]
        )

    elif hparams["architecture"] == "eqgat":
        from models.eqgat import EqgatGenerator

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
        from models.egnn import VanillaEgnnGenerator

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
    encoder = initialize_encoder(
        encoder_type=args.encoder_type,
        encoder_ckpt_path=args.encoder_path,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()  

    rep_sampler = initilize_rep_sampler(args, args, dataset=hparams["dataset"])



    fm_model = MolecularCFM.load_from_checkpoint(
        args.ckpt_path,
        strict=False,
        gen=egnn_gen,
        vocab=vocab,
        integrator=integrator,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index,
        encoder=encoder,
        rdm=rep_sampler,
        **hparams,
    )
    return fm_model


def build_dm(args, hparams, vocab):
    if args.dataset == "qm9":
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
        None,
        None,
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


def evaluate(args, model, dm, metrics, stab_metrics):
    # torch.set_float32_matmul_precision("medium")
    results_list = []
    for replicate_index in range(args.n_replicates):
        print(f"Running replicate {replicate_index + 1} out of {args.n_replicates}")
        
        # Pregenerate all reps for the replicate
        util.generate_reps(model, dm, args.integration_steps, args.ode_sampling_strategy, stabilities=True)
        
        molecules, _, stabilities = util.generate_molecules(
            model,
            dm,
            args.integration_steps,
            args.ode_sampling_strategy,
            stabilities=True
        )
        
        
        model.pre_generated_reps_sizes = []
        model.pre_generated_reps = []

        print("Calculating metrics...")
        results = util.calc_metrics_(molecules, metrics, stab_metrics=stab_metrics, mol_stabs=stabilities)
        results_list.append(results)
        print(results)

    results_dict = {key: [] for key in results_list[0].keys()}
    for results in results_list:
        for metric, value in results.items():
            results_dict[metric].append(value.item())

    mean_results = {metric: np.mean(values) for metric, values in results_dict.items()}
    std_results = {metric: np.std(values) for metric, values in results_dict.items()}

    return mean_results, std_results, results_dict

@hydra.main(config_path="./configs", config_name="eval.yaml", version_base="1.3")
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

    print(f"Loading model...")
    model = load_model(args, vocab)
    print("Model complete.")

    print("Initialising metrics...")
    metrics, stab_metrics = util.init_metrics(args.data_path, model)
    print("Metrics complete.")

    print("Running evaluation...")
    avg_results, std_results, list_results = evaluate(args, model, dm, metrics, stab_metrics)
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
