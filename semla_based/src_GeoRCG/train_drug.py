import sys
sys.path.append(".")
from pathlib import Path
from functools import partial
import hydra
import torch
import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

import scriptutil as util
from flowmodels.fm import MolecularCFM, Integrator
from flowmodels.semla import SemlaGenerator, EquiInvDynamics
from flowmodels.encoders import initialize_encoder
from flowmodels.rep_samplers import *
from data.datasets import GeometricDataset
from data.datamodules import GeometricInterpolantDM
from data.interpolate import GeometricInterpolant, GeometricNoiseSampler




# bfloat16 training produced significantly worse flowmodels than full so use default 16-bit instead
def get_precision(args):
    return "32"
    # return "16-mixed" if args.mixed_precision else "32"


def build_model(args, dm, vocab):
    # Get hyperparameeters from the datamodule, pass these into the model to be saved
    hparams = {
        "epochs": args.epochs,
        "gradient_clip_val": args.gradient_clip_val,
        "dataset": args.dataset,
        "precision": get_precision(args),
        "architecture": args.arch,
        **dm.hparams
    }

    # Add 1 for the time (0 <= t <= 1 for flow matching)
    n_atom_feats = vocab.size + 1
    n_bond_types = util.get_n_bond_types(args.categorical_strategy)

    if args.arch == "semla":
        dynamics = EquiInvDynamics(
            args.d_model,
            args.d_message,
            args.n_coord_sets,
            args.n_layers,
            n_attn_heads=args.n_attn_heads,
            d_message_hidden=args.d_message_hidden,
            d_edge=args.d_edge,
            bond_refine=True,
            self_cond=args.self_condition,
            coord_norm=args.coord_norm,
            
            d_rep=args.d_rep,
            attn_block_num=args.attn_block_num,
            dropout=args.dropout,
            original=args.original,
            use_gate=args.use_gate,
        )
        egnn_gen = SemlaGenerator(
            args.d_model,
            dynamics,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            self_cond=args.self_condition,
            size_emb=args.size_emb,
            max_atoms=args.max_atoms
        )

    elif args.arch == "eqgat":
        from models.eqgat import EqgatGenerator

        # Hardcode for now since we only need one model size
        d_model_eqgat = 256
        n_equi_feats_eqgat = 256
        n_layers_eqgat = 12
        d_edge_eqgat = 128

        egnn_gen = EqgatGenerator(
            d_model_eqgat,
            n_layers_eqgat,
            n_equi_feats_eqgat,
            vocab.size,
            n_atom_feats,
            d_edge_eqgat,
            n_bond_types,
            d_rep=args.d_rep
        )

    elif args.arch == "egnn":
        from models.egnn import VanillaEgnnGenerator

        egnn_gen = VanillaEgnnGenerator(
            args.d_model,
            args.n_layers,
            vocab.size,
            n_atom_feats,
            d_edge=args.d_edge,
            n_edge_types=n_bond_types,
            d_rep=args.d_rep
        )

    else:
        raise ValueError(f"Unknown architecture '{args.arch}'")

    if args.dataset == "qm9":
        coord_scale = util.QM9_COORDS_STD_DEV
    elif args.dataset == "geom-drugs":
        coord_scale = util.GEOM_COORDS_STD_DEV
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        train_strategy = "mask"
        sampling_strategy = "mask"

    elif args.categorical_strategy == "uniform-sample":
        train_strategy = "ce"
        sampling_strategy = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        train_strategy = "ce"
        sampling_strategy = "dirichlet"

    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported.")

    train_steps = util.calc_train_steps(dm, args.epochs, args.acc_batches)
    train_smiles = None if args.trial_run else [mols.str_id for mols in dm.train_dataset]

    print(f"Total training steps {train_steps}")

    integrator = Integrator(
        args.num_inference_steps,
        type_strategy=sampling_strategy,
        bond_strategy=sampling_strategy,
        cat_noise_level=args.cat_sampling_noise_level,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )

    # Set up for encoder
    encoder = initialize_encoder(encoder_type=args.encoder_type,
                                 encoder_ckpt_path=args.encoder_path)
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()


    # Set up the sampler
    rep_sampler = initilize_rep_sampler(args, args, dataset=args.dataset)

    if args.resume_checkpoint is not None:
        print(f"Loading model from checkpoint {args.resume_checkpoint}")
        fm_model = MolecularCFM.load_from_checkpoint(
            checkpoint_path=args.resume_checkpoint,
            strict=False,
            gen=egnn_gen,
            vocab=vocab,
            lr=args.lr,
            integrator=integrator,
            coord_scale=coord_scale,
            type_strategy=train_strategy,
            bond_strategy=train_strategy,
            type_loss_weight=args.type_loss_weight,
            bond_loss_weight=args.bond_loss_weight,
            charge_loss_weight=args.charge_loss_weight,
            pairwise_metrics=False,
            use_ema=args.use_ema,
            compile_model=False,
            self_condition=args.self_condition,
            distill=False,
            lr_schedule=args.lr_schedule,
            warm_up_steps=args.warm_up_steps,
            total_steps=train_steps,
            train_smiles=train_smiles,
            type_mask_index=type_mask_index,
            bond_mask_index=bond_mask_index,
            
            rep_condition=True,
            encoder=encoder,
            rdm=rep_sampler,
            rep_dropout_prob=args.rep_dropout_prob,
            noise_sigma=args.noise_sigma,
            d_rep=args.d_rep,
            cfg_coef=args.cfg_coef,
            scheduled_noise=args.scheduled_noise,
            rep_loss_weight=args.rep_loss_weight,
            **hparams,
            )
    else:
        fm_model = MolecularCFM(
            egnn_gen,
            vocab,
            args.lr,
            integrator,
            coord_scale=coord_scale,
            type_strategy=train_strategy,
            bond_strategy=train_strategy,
            type_loss_weight=args.type_loss_weight,
            bond_loss_weight=args.bond_loss_weight,
            charge_loss_weight=args.charge_loss_weight,
            pairwise_metrics=False,
            use_ema=args.use_ema,
            compile_model=False,
            self_condition=args.self_condition,
            distill=False,
            lr_schedule=args.lr_schedule,
            warm_up_steps=args.warm_up_steps,
            total_steps=train_steps,
            train_smiles=train_smiles,
            type_mask_index=type_mask_index,
            bond_mask_index=bond_mask_index,
            
            rep_condition=True,
            encoder=encoder,
            rdm=rep_sampler,
            rep_dropout_prob=args.rep_dropout_prob,
            noise_sigma=args.noise_sigma,
            d_rep=args.d_rep,
            cfg_coef=args.cfg_coef,
            scheduled_noise=args.scheduled_noise,
            rep_loss_weight=args.rep_loss_weight,
            **hparams
        )
    
    
    return fm_model


def build_dm(args, vocab):
    if args.dataset == "qm9":
        coord_std = util.QM9_COORDS_STD_DEV
        padded_sizes = util.QM9_BUCKET_LIMITS

    elif args.dataset == "geom-drugs":
        coord_std = util.GEOM_COORDS_STD_DEV
        padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS

    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    data_path = Path(args.data_path)

    n_bond_types = util.get_n_bond_types(args.categorical_strategy)
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    # Load generated dataset with different transform fn if we are distilling a model
    # if args.distill:
    #     distill_transform = partial(util.distill_transform, coord_std=coord_std)
    #     train_dataset = GeometricDataset.load(data_path / "distill.smol", transform=distill_transform)
    # else:
    #     train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)

    if not args.trial_run:
        train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)
        val_dataset = GeometricDataset.load(data_path / "val.smol", transform=transform)
        val_dataset = val_dataset.sample(args.n_validation_mols)
    else:
        val_dataset = GeometricDataset.load(data_path / "val.smol", transform=transform)
        train_dataset = val_dataset
        

    type_mask_index = None
    bond_mask_index = None

    if args.categorical_strategy == "mask":
        type_mask_index = vocab.indices_from_tokens(["<MASK>"])[0]
        bond_mask_index = util.BOND_MASK_INDEX
        categorical_interpolation = "unmask"
        categorical_noise = "mask"

    elif args.categorical_strategy == "uniform-sample":
        categorical_interpolation = "unmask"
        categorical_noise = "uniform-sample"

    elif args.categorical_strategy == "dirichlet":
        categorical_interpolation = "dirichlet"
        categorical_noise = "uniform-dist"

    else:
        raise ValueError(f"Interpolation '{args.categorical_strategy}' is not supported.")

    scale_ot = False
    batch_ot = False
    equivariant_ot = False

    if args.optimal_transport == "batch":
        batch_ot = True
    elif args.optimal_transport == "equivariant":
        equivariant_ot = True
    elif args.optimal_transport == "scale":
        scale_ot = True
        equivariant_ot = True
    elif args.optimal_transport not in ["None", "none", None]:
        raise ValueError(f"Unknown value for optimal_transport '{args.optimal_transport}'")

    # train_fixed_time = 0.5 if args.distill else None
    train_fixed_time = None

    prior_sampler = GeometricNoiseSampler(
        vocab.size,
        n_bond_types,
        coord_noise="gaussian",
        type_noise=categorical_noise,
        bond_noise=categorical_noise,
        scale_ot=scale_ot,
        zero_com=True,
        type_mask_index=type_mask_index,
        bond_mask_index=bond_mask_index
    )
    train_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        coord_noise_std=args.coord_noise_std_dev,
        type_dist_temp=args.type_dist_temp,
        equivariant_ot=equivariant_ot,
        batch_ot=batch_ot,
        time_alpha=args.time_alpha,
        time_beta=args.time_beta,
        fixed_time=train_fixed_time
    )
    eval_interpolant = GeometricInterpolant(
        prior_sampler,
        coord_interpolation="linear",
        type_interpolation=categorical_interpolation,
        bond_interpolation=categorical_interpolation,
        equivariant_ot=False,
        batch_ot=False,
        fixed_time=0.9
    )

    dm = GeometricInterpolantDM(
        train_dataset,
        val_dataset,
        None,
        args.batch_cost,
        train_interpolant=train_interpolant,
        val_interpolant=eval_interpolant,
        test_interpolant=None,
        bucket_limits=padded_sizes,
        bucket_cost_scale=args.bucket_cost_scale,
        pad_to_bucket=False
    )
    return dm


def build_trainer(args):
    epochs = 1 if args.trial_run else args.epochs
    log_steps = 1 if args.trial_run else 50

    if args.dataset == "qm9":
        val_check_epochs = args.val_check_epochs
    elif args.dataset == "geom-drugs":
        val_check_epochs = args.val_check_epochs
    else:
        raise ValueError(f"Unknown dataset {args.dataset}")

    project_name = f"{util.PROJECT_PREFIX}-{args.dataset}"
    precision = get_precision(args)

    print(f"Using precision '{precision}'")

    logger = WandbLogger(project=project_name, save_dir="wandb", log_model=True, version=args.version)
    lr_monitor = LearningRateMonitor(logging_interval="step")
    checkpointing = ModelCheckpoint(
        every_n_epochs=val_check_epochs,
        monitor="val-validity",
        mode="max",
        save_last=True
    )

    # Overwrite if doing a trial run
    val_check_epochs = 1 if args.trial_run else val_check_epochs
    logger = None if args.trial_run else logger
    num_sanity_val_steps = 0 if args.trial_run else 2
    trainer = L.Trainer(
        min_epochs=epochs,
        max_epochs=epochs,
        logger=logger,
        log_every_n_steps=log_steps,
        accumulate_grad_batches=args.acc_batches,
        gradient_clip_val=args.gradient_clip_val,
        check_val_every_n_epoch=val_check_epochs,
        callbacks=[lr_monitor, checkpointing],
        precision=precision,
        num_sanity_val_steps=num_sanity_val_steps,
    )
    return trainer


@hydra.main(config_path="../configs", config_name="train_gen.yaml", version_base="1.3")
def main(args):
    args = args.experiments_gen
    # Set some useful torch properties
    # Float32 precision should only affect computation on A100 and should in theory be a lot faster than the default setting
    # Increasing the cache size is required since the model will be compiled seperately for each bucket
    torch.set_float32_matmul_precision("high")
    # torch._dynamo.config.cache_size_limit = util.COMPILER_CACHE_SIZE

    # print(f"Set torch compiler cache size to {torch._dynamo.config.cache_size_limit}")

    L.seed_everything(12345)
    util.disable_lib_stdout()
    util.configure_fs()

    print("Building model vocab...")
    vocab = util.build_vocab()
    print("Vocab complete.")

    print("Loading datamodule...")
    dm = build_dm(args, vocab)
    print("Datamodule complete.")

    print(f"Building equinv model...")
    model = build_model(args, dm, vocab)
    print("Model complete.")

    trainer = build_trainer(args)

    print("Fitting datamodule to model...")
    # if args.resume_checkpoint is None:
    #     trainer.fit(model, datamodule=dm)
    # else:
    #     print(f"Resuming training from checkpoint {args.resume_checkpoint}")
    #     trainer.fit(model, datamodule=dm, ckpt_path=args.resume_checkpoint)
    
    trainer.fit(model, datamodule=dm)
    
    print("Training complete.")

if __name__ == "__main__":
    main()
