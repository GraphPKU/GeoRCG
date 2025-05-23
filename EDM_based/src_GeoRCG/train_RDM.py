import sys
sys.path.append(".")
import numpy as np
import torch
import time
import datetime
import wandb
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import torch.distributed as dist
from pathlib import Path

import models_GeoRCG.util.misc as misc
from models_GeoRCG.engine_rdm import train_one_epoch
from omegaconf import OmegaConf
from initialize_models import initialize_RDM
from qm9 import dataset
from qm9.models import DistributionNodes
from configs.datasets_config import get_dataset_info
from models_GeoRCG.rep_samplers import initilize_rep_sampler
from qm9.utils import compute_mean_mad


def vis_tsne(running_rdm_args, save_dir, epoch, n_datapoints=10000, device="cuda", inv_temp=None, gtsampler=None, nodes_dist=None, prop_dist=None):
    # PCSampler
    sampler = "PCSampler"
    rdm_ckpt = running_rdm_args.output_dir + "/checkpoint-last.pth"
    inv_temp = inv_temp
    n_steps = 5

    rep_sampler_args = {
        "sampler": sampler,
        "rdm_ckpt": rdm_ckpt,
        "inv_temp": inv_temp,
        "n_steps": n_steps,
        "snr": 0.01
    }
    rep_sampler_args = OmegaConf.create(rep_sampler_args)

    pcsampler = initilize_rep_sampler(rep_sampler_args, device, dataset_args=running_rdm_args)
    
    # Sampling
    print("Sampling GT Reps...")
    gt_nodesxsample = nodes_dist.sample(n_datapoints)
    gt_addtional_cond = None
    if prop_dist is not None:
        gt_addtional_cond = prop_dist.sample_batch(gt_nodesxsample)
    gt_reps = gtsampler.sample(
        device=device,
        nodesxsample=gt_nodesxsample,
        additional_cond=gt_addtional_cond,
        running_batch_size=100,
    )
    gt_y = torch.zeros((gt_reps.shape[0]), device=device)
    print("Finished Sampling GT Reps.")
    
    print("Sampling PC Reps...")
    pc_nodesxsample = nodes_dist.sample(n_datapoints)
    pc_addtional_cond = None
    if prop_dist is not None:
        pc_addtional_cond = prop_dist.sample_batch(pc_nodesxsample)
    pc_reps = pcsampler.sample(
        device=device,
        nodesxsample=pc_nodesxsample,
        additional_cond=pc_addtional_cond,
        running_batch_size=2000,
    )
    pc_y = torch.ones((pc_reps.shape[0]), device=device)
    print("Finished Sampling PC Reps.")
    
    # Step 1: Combine representations and labels
    combined_reps = torch.cat((gt_reps, pc_reps), dim=0).cpu().numpy()
    combined_y = torch.cat((gt_y, pc_y), dim=0).cpu().numpy()
    


    # Step 2: Perform t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    tsne_results = tsne.fit_transform(combined_reps)
    
    
    # now, calculate the SS score of the clustering result.
    silhouette_avg = silhouette_score(tsne_results, combined_y)
    silhouette_avg_4_abs = abs(silhouette_avg * 1e4)
    wandb.log({"SS_4_abs": silhouette_avg_4_abs}, commit=False)
    
    # Step 3: Visualize the results
    plt.figure(figsize=(20, 16))
    plt.scatter(tsne_results[combined_y == 0, 0], tsne_results[combined_y == 0, 1], label='gt_reps', alpha=0.6)
    plt.scatter(tsne_results[combined_y == 1, 0], tsne_results[combined_y == 1, 1], label='pc_reps', alpha=0.6)
    plt.legend()
    plt.title('t-SNE Visualization of gt_reps and pc_reps')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')

    
    
    save_path = f"{save_dir}/epoch{epoch}_inv_temp{inv_temp}.pdf"
    
    plt.savefig(save_path)
    plt.close()
        
def dist_setup():
    assert torch.cuda.device_count() > 1, "Only one cuda but using distributed training."
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=100))
    assert dist.is_initialized() and dist.is_available()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    return rank, world_size
    
import hydra           
@hydra.main(config_path="../configs", config_name="train_RDM.yaml", version_base="1.3")
def main(args):
    OmegaConf.set_struct(args, False)
    args = args.experiments_RDM
    rdm_args = args.rdm_args
    model_args = args.model_args
    
    # Set up for debugging
    if rdm_args.debug:
        print("Warning: You are using the debug mode!!!")
        rdm_args.dp = False
        rdm_args.exp_name = "debug"
        rdm_args.no_wandb = True
    
    
    # Set up for DP
    if rdm_args.dp:
        rank, world_size = dist_setup()
    else:
        rank = 0
        world_size = 1
    rdm_args.rank = rank
    rdm_args.world_size = world_size
        
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda")
    dtype = torch.float32
    assert torch.cuda.is_available(), "Only support cuda training!"

    # Fix the seed for reproducibility
    seed = rdm_args.seed + rank
    torch.manual_seed(seed)
    np.random.seed(seed)

    
    # Set up for the datasets, data_loaders, node_dist, prop_dist, dataset_info
    if not rdm_args.semlaflow_data:
        data_loaders, charge_scale = dataset.retrieve_dataloaders(rdm_args)
        data_loader_train = data_loaders["train"]
    else:
        data_loaders, charge_scale = dataset.semlaflow_drug_train_dataloader(rdm_args)
        data_loader_train = data_loaders["train"]
    data_dummy = next(iter(data_loader_train))
    if not rdm_args.semlaflow_data:
        dataset_info = get_dataset_info(rdm_args.dataset, rdm_args.remove_h)
        histogram = dataset_info['n_nodes']
    else:
        data_list = dataset.semlaflow_drug_train_dataloader(rdm_args, return_data_list=True)
        lengths = [data.shape[0] for data in data_list]
        # To create a histogram
        histogram = {}
        for length in lengths:
            if length not in histogram:
                histogram[length] = 1
            else:
                histogram[length] += 1


    # Set up for gt_sampler, which is used for visualization
    dataset_info = get_dataset_info(rdm_args.dataset, rdm_args.remove_h)
    nodes_dist = DistributionNodes(histogram)
    
    # Set up for class_cond and lr and dirs
    rdm_args.class_cond = model_args.params.get("class_cond", False)    
    
    eff_batch_size = rdm_args.batch_size * rdm_args.accum_iter * world_size
    rdm_args.lr = rdm_args.blr * eff_batch_size
    rdm_args.output_dir = f'./outputs/rdm/{rdm_args.exp_name}/model'
    rdm_args.vis_output_dir = f'./outputs/rdm/{rdm_args.exp_name}/vis'
    rdm_args.log_dir = f'./outputs/rdm/{rdm_args.exp_name}/log'
    exp_dir = f'./outputs/rdm/{rdm_args.exp_name}'
    Path(exp_dir).mkdir(parents=True, exist_ok=True)
    Path(rdm_args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(rdm_args.vis_output_dir).mkdir(parents=True, exist_ok=True)
    Path(rdm_args.log_dir).mkdir(parents=True, exist_ok=True)

    # Set up for basic models
    model, model_without_ddp, loss_scaler, optimizer = initialize_RDM(rdm_args, model_args, device)

    
    # Set up for wandb logging
    if rank == 0:
        if rdm_args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if rdm_args.online else 'offline'
        kwargs = {'entity': rdm_args.wandb_usr, 'name': rdm_args.exp_name, 'project': 'e3_diffusion', 'config': {k: v for k, v in rdm_args.items()},
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
        
        
    if rdm_args.rdm_ckpt is not None and rdm_args.rdm_ckpt != "":
        # When resuming, we reset the initial global step of wandb.
        global_step = rdm_args.start_epoch
        if rank == 0: wandb.log({}, step=global_step) 
    else: global_step = -1
    
    # Prepare conditioning variables
    if len(rdm_args.conditioning) > 0:
        property_norms = compute_mean_mad(data_loaders, rdm_args.conditioning, rdm_args.dataset)
    else:
        property_norms = None
        
        

    # prepare GtSampler
    gt_sampler_args = {
        "sampler": "GtSampler",
        "Gt_dataset": "train",
        "encoder_path": rdm_args.encoder_path,
        "encoder_type": rdm_args.encoder_type,
    }
    gtsampler = initilize_rep_sampler(OmegaConf.create(gt_sampler_args), device, rdm_args)
    
    
    # Now, we can start training.
    for epoch in range(rdm_args.start_epoch, rdm_args.epochs):
        data_loader_train.sampler.set_epoch(epoch)
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args=rdm_args, property_norms=property_norms, dtype=dtype
        )
        
        if rank == 0:
            misc.save_model_last(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            if rdm_args.output_dir and (epoch % rdm_args.vis_interval == 0 or epoch + 1 == rdm_args.epochs):
                misc.save_model(
                    args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                    loss_scaler=loss_scaler, epoch=epoch)
                s = time.time()
                if not rdm_args.debug:
                    vis_tsne(running_rdm_args=rdm_args, save_dir=rdm_args.vis_output_dir, epoch=epoch, inv_temp=1.0, device=device, nodes_dist=nodes_dist, gtsampler=gtsampler)
                print(f"Visualization took {time.time() - s}s.")
            
            wandb.log(train_stats, commit=True)
        if rdm_args.dp:
            dist.barrier()




if __name__ == "__main__":
    main()