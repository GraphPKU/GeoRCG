from torch.utils.data import DataLoader
from qm9.data.args import init_argparse
from qm9.data.collate import PreprocessQM9
from qm9.data.utils import initialize_datasets
import torch


def retrieve_dataloaders(cfg, raw_datasets_and_collate_fn=False):
    if 'qm9' in cfg.dataset:
        batch_size = cfg.batch_size
        num_workers = cfg.num_workers
        filter_n_atoms = cfg.filter_n_atoms
        # Initialize dataloader
        args = init_argparse('qm9')
        # data_dir = cfg.data_root_dir
        args, datasets, num_species, charge_scale = initialize_datasets(args, cfg.datadir, cfg.dataset,
                                                                        subtract_thermo=args.subtract_thermo,
                                                                        force_download=args.force_download,
                                                                        remove_h=cfg.remove_h)
        qm9_to_eV = {'U0': 27.2114, 'U': 27.2114, 'G': 27.2114, 'H': 27.2114, 'zpve': 27211.4, 'gap': 27.2114, 'homo': 27.2114,
                     'lumo': 27.2114}

        for dataset in datasets.values():
            dataset.convert_units(qm9_to_eV)

        if filter_n_atoms is not None:
            print("Retrieving molecules with only %d atoms" % filter_n_atoms)
            datasets = filter_atoms(datasets, filter_n_atoms)

        # Construct PyTorch dataloaders from datasets
        preprocess = PreprocessQM9(load_charges=cfg.include_charges)
        
        
        sampler_train = torch.utils.data.DistributedSampler(
            datasets["train"], num_replicas=cfg.world_size, rank=cfg.rank, shuffle=True, 
        )
        sampler_valid = torch.utils.data.DistributedSampler(
            datasets["valid"], num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False
        )
        sampler_test = torch.utils.data.DistributedSampler(
            datasets["test"], num_replicas=cfg.world_size, rank=cfg.rank, shuffle=False
        )
        
        samplers = [sampler_train, sampler_valid, sampler_test]
        dataloaders = {split: DataLoader(dataset,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         collate_fn=preprocess.collate_fn,
                                         sampler=samplers[i],
                                         ) 
                             for i, (split, dataset) in enumerate(datasets.items())}
        if raw_datasets_and_collate_fn:
            return datasets, preprocess.collate_fn

    elif 'geom' in cfg.dataset:
        import build_geom_dataset
        from configs.datasets_config import get_dataset_info
        dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)

        # Retrieve QM9 dataloaders
        split_data = build_geom_dataset.load_split_data(cfg.data_file,
                                                        val_proportion=0.1,
                                                        test_proportion=0.1,
                                                        filter_size=cfg.filter_molecule_size)
        transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                          cfg.include_charges,
                                                          cfg.device,
                                                          cfg.sequential)
        dataloaders = {}
        for key, data_list in zip(['train', 'val', 'test'], split_data):
            dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                          transform=transform, debug=cfg.debug)
            shuffle = (key == 'train') and not cfg.sequential
            
            if raw_datasets_and_collate_fn:
                assert key == "train"
                from build_geom_dataset import collate_fn as geom_collate_fn
                datasets = {}
                datasets["train"] = dataset
                return datasets, geom_collate_fn

            # Sequential dataloading disabled for now.
            dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
                sequential=cfg.sequential, dataset=dataset,
                batch_size=cfg.batch_size,
                shuffle=shuffle,
                world_size=cfg.world_size,
                rank=cfg.rank
                )
        del split_data
        charge_scale = None
    else:
        raise ValueError(f'Unknown dataset {cfg.dataset}')

    return dataloaders, charge_scale


def filter_atoms(datasets, n_nodes):
    for key in datasets:
        dataset = datasets[key]
        idxs = dataset.data['num_atoms'] == n_nodes
        for key2 in dataset.data:
            dataset.data[key2] = dataset.data[key2][idxs]

        datasets[key].num_pts = dataset.data['one_hot'].size(0)
        datasets[key].perm = None
    return datasets


def semlaflow_drug_train_dataloader(cfg, return_data_list=False):
    print("Using semlaflow data.")
    
    import sys
    sys.path.append("../semla_based")
    
    import scriptutil as util
    from data.datasets import GeometricDataset
    
    sys.path.remove("../semla_based")

    
    from pathlib import Path
    from functools import partial

    coord_std = util.GEOM_COORDS_STD_DEV
    padded_sizes = util.GEOM_DRUGS_BUCKET_LIMITS
    data_path = Path("data/geom/smol")

    n_bond_types = util.get_n_bond_types("uniform-sample")
    vocab = util.build_vocab()
    transform = partial(util.mol_transform, vocab=vocab, n_bonds=n_bond_types, coord_std=coord_std)

    print("Loading geom data ...")
    train_dataset = GeometricDataset.load(data_path / "train.smol", transform=transform)
    train_data = train_dataset._data
    train_data_mask = train_data.mask.to(torch.bool)
    train_data_atomics = train_data.atomics
    train_data_coords = train_data.coords
    train_data_list = [(atomics[mask], coords[mask]) for mask, atomics, coords in zip(train_data_mask, train_data_atomics, train_data_coords)]
    
    # To create a list of arrays, shaped (n_atoms, 4), where the columns are (Z, x, y, z)
    train_data_list = [torch.cat([atomics.unsqueeze(-1).to(torch.float), coords], dim=-1) for atomics, coords in train_data_list]
    # make it a list of arrays, shaped (n_atoms, 4)
    train_data_list = [arr.numpy() for arr in train_data_list]
    print("Geom data loaded.")
    if return_data_list:
        return train_data_list
    
    import build_geom_dataset
    from configs.datasets_config import get_dataset_info
    dataset_info = get_dataset_info(cfg.dataset, cfg.remove_h)
    split_data = [
        train_data_list,
        None,
        None
]
    transform = build_geom_dataset.GeomDrugsTransform(dataset_info,
                                                        cfg.include_charges,
                                                        cfg.device,
                                                        cfg.sequential)
    dataloaders = {}
    for key, data_list in zip(['train', 'val', 'test'], split_data):
        if data_list is None:
            dataloaders[key] = None
            continue
        
        dataset = build_geom_dataset.GeomDrugsDataset(data_list,
                                                        transform=transform, debug=cfg.debug)
        shuffle = (key == 'train') and not cfg.sequential

        # Sequential dataloading disabled for now.
        dataloaders[key] = build_geom_dataset.GeomDrugsDataLoader(
            sequential=cfg.sequential, dataset=dataset,
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            world_size=cfg.world_size,
            rank=cfg.rank
            )
    del split_data
    charge_scale = None
    
    return dataloaders, charge_scale