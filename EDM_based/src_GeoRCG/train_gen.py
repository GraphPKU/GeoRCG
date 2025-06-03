import sys
sys.path.append(".")
import copy
import utils
import wandb
from os.path import join
import torch
import time
import pickle
from omegaconf import OmegaConf
import torch.distributed as dist
import copy
import datetime
import hydra           


from qm9 import dataset
from qm9.models import get_optim, get_model
from equivariant_diffusion import en_diffusion
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
from models_GeoRCG.rep_samplers import *
from models_GeoRCG.encoders import initialize_encoder
from utils import reduced_mean
from qm9.utils import prepare_context, compute_mean_mad
from train_test import train_epoch, test, analyze_and_save
from configs.datasets_config import get_dataset_info

def dist_setup():
    assert torch.cuda.device_count() > 1, "Only one cuda but using distributed training."
    dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=600))
    assert dist.is_initialized() and dist.is_available()
    rank, world_size = dist.get_rank(), dist.get_world_size()
    return rank, world_size

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)
@hydra.main(config_path="../configs", config_name="train_gen.yaml", version_base="1.3")
def main(args):
    OmegaConf.set_struct(args, False)
    args = args.experiments_gen
    
    gen_args = args.gen_args

    
    if gen_args.debug:
        print("Warning: You are using the debug mode!!!")
        gen_args.dp = False
        gen_args.exp_name = "debug"
        gen_args.no_wandb = True
        gen_args.n_stability_samples = 4

    
    # Set up for DP
    if gen_args.dp:
        rank, world_size = dist_setup()
        gen_args.rank = rank
        gen_args.world_size = world_size
        print("World_size", gen_args.world_size)
        print("Rank", gen_args.rank)
    else:
        rank = 0
        world_size = 1
        gen_args.rank = rank
        gen_args.world_size = world_size
        print("World_size", 1)
        print("Rank", 0)
        
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda")
    assert torch.cuda.is_available() # only support cuda training 
    gen_args.device = "cuda"
    
    # Set up for resume (NOTE: Careful with this!)
    if gen_args.resume is not None:
        with open(join(gen_args.resume, 'args.pickle'), 'rb') as f:
            resumed_args = pickle.load(f)
        new_args = copy.deepcopy(args)

        current_epoch = resumed_args.gen_args.current_epoch
        resumed_exp_name = resumed_args.gen_args.exp_name
        
        
        # Now, compare them. You should yourself ensure the same batch size
        if new_args.gen_args != resumed_args.gen_args:
            print(
                f"WARNING: detected difference in gen args between current args and the resumed args:\n"
            )
            union_keys = list(set(new_args.gen_args.keys()).union(set(resumed_args.gen_args.keys()))) 
            for key in union_keys:
                if new_args.gen_args.get(key, "UNKNOWN") != resumed_args.gen_args.get(key, "UNKNOWN"):
                    print(f"     Different in {key}: new_args {new_args.gen_args.get(key, 'UNKNOWN')}, resumed_args {resumed_args.gen_args.get(key, 'UNKNOWN')}")

            
        # gen_args.exp_name = resumed_exp_name + "_resume"
        gen_args.start_epoch = current_epoch
        

    if rank == 0:
        if gen_args.no_wandb:
            mode = 'disabled'
        else:
            mode = 'online' if gen_args.online else 'offline'
        kwargs = {'entity': gen_args.wandb_usr, 'name': gen_args.exp_name, 'project': 'e3_diffusion', 'config': {k: v for k, v in args.items()},
                'settings': wandb.Settings(_disable_stats=True), 'reinit': True, 'mode': mode}
        wandb.init(**kwargs)
        wandb.save('*.txt')
    import setproctitle
    setproctitle.setproctitle(f"{gen_args.exp_name}")
    
    # Set up for encoder
    encoder = initialize_encoder(encoder_type=gen_args.encoder_type, device=device, encoder_ckpt_path=gen_args.encoder_path)
    
    
    for param in encoder.parameters():
        param.requires_grad = False
    encoder.eval()
    encoder_dp = encoder
    
    
    

    # Set up for the datasets
    dataset_info = get_dataset_info(gen_args.dataset, gen_args.remove_h)
    dataloaders, charge_scale = dataset.retrieve_dataloaders(gen_args)
    data_dummy = next(iter(dataloaders['train']))
    
    
    # Set up for conditioning
    if len(gen_args.conditioning) > 0:
        if rank == 0:
            print(f'Conditioning on {gen_args.conditioning}')
        property_norms = compute_mean_mad(dataloaders, gen_args.conditioning, gen_args.dataset)
        context_dummy = prepare_context(gen_args.conditioning, data_dummy, property_norms)
        context_node_nf = context_dummy.size(2)
        
    else:
        property_norms = None
        context_node_nf = 0
    gen_args.context_node_nf = context_node_nf
        
    # Set up for point cloud diffusion model
    model, nodes_dist, prop_dist = get_model(gen_args, device, dataset_info, dataloaders['train'])
        
    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
    model = model.to(device)
    optim = get_optim(gen_args, model)
    
    
    if gen_args.dp:
        model_dp = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
        model_dp = model_dp.cuda()
    else:
        model_dp = model

    if gen_args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(gen_args.ema_decay)

        if gen_args.dp and torch.cuda.device_count() > 1:
            model_ema_dp = torch.nn.parallel.DistributedDataParallel(model_ema, find_unused_parameters=True)
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp
    
    model_ema.eval()
    model_ema_dp.eval()

    if gen_args.resume is not None:
        flow_state_dict = torch.load(join(gen_args.resume, 'generative_model.npy'))
        ema_flow_state_dict = torch.load(join(gen_args.resume, 'generative_model_ema.npy'))
        optim_state_dict = torch.load(join(gen_args.resume, 'optim.npy'))
        model.load_state_dict(flow_state_dict)
        optim.load_state_dict(optim_state_dict)
        model_ema.load_state_dict(ema_flow_state_dict)


    # Set up the sampler
    rep_sampler = initilize_rep_sampler(gen_args, device, gen_args, debug=gen_args.debug)
    from models_GeoRCG.wrapper import SelfConditionWrappedSampler
    sampler = SelfConditionWrappedSampler(gen_sampler=model_ema, rdm_sampler=rep_sampler)


    # Other preparations
    utils.create_folders(gen_args)
    dtype = torch.float32
    
    best_nll_val = 1e8
    best_nll_test = 1e8
    
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000)  # Add large value that will be flushed.
    
    # Print meta informations
    if rank == 0:
        print(f"Args: {args}")
        print(f"Training Using {world_size} GPUs")
        print(f"Point Cloud Diffusion Model: {model}")
        print(f"Encoder Model: {encoder}")
    santi_check_about_sampling = False
    if gen_args.resume is not None:
        # When resuming, we reset the initial global step of wandb. But notice that the global step calculated in the following way is not very precise.
        global_step = (gen_args.start_epoch) * len(dataloaders["train"]) 
        if rank == 0: wandb.log({}, step=global_step) 
        santi_check_about_sampling = True
    else: global_step = -1
    
    
    assert gen_args.dataset != 'geom' or (not gen_args.include_charges), "We do not use charge features for geom dataset."
    
    for epoch in range(gen_args.start_epoch, gen_args.n_epochs):
        start_epoch = time.time()
        train_epoch(args=gen_args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist,
                    encoder=encoder_dp, rank=rank, encoder_dp=encoder_dp, sampler=sampler
                    )
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")
        
        if gen_args.save_model and rank == 0 and gen_args.dataset == "geom": # We save the model every epoch for geom dataset, since it takes a long time to train.
            utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (gen_args.exp_name, epoch))
            utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (gen_args.exp_name, epoch))
            if gen_args.ema_decay > 0:
                utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (gen_args.exp_name, epoch))
            with open('outputs/%s/args_%d.pickle' % (gen_args.exp_name, epoch), 'wb') as f:
                pickle.dump(args, f)
        
        if epoch % gen_args.test_epochs == 0 or santi_check_about_sampling:
            santi_check_about_sampling = False
            if isinstance(model, en_diffusion.EnVariationalDiffusion) and rank == 0:
                wandb.log(model.log_info(), commit=True)

            if not gen_args.break_train_epoch and rank == 0 and len(gen_args.conditioning) == 0:
                analyze_and_save(args=gen_args, epoch=epoch, model_sample=sampler, nodes_dist=nodes_dist,
                                 dataset_info=dataset_info, device=device,
                                 prop_dist=prop_dist, n_samples=gen_args.n_stability_samples,
                                 batch_size=gen_args.eval_batch_size)
            if gen_args.dp: dist.barrier()
            nll_epoch_val, n_samples_val = test(args=gen_args, loader=dataloaders['valid'] if gen_args.dataset != 'geom' else dataloaders['val'], epoch=epoch, eval_model=model_ema_dp,
                           partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                           property_norms=property_norms, encoder=encoder, rank=rank)
            nll_epoch_test, n_samples_test = test(args=gen_args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Test', device=device, dtype=dtype,
                            nodes_dist=nodes_dist, property_norms=property_norms, encoder=encoder, rank=rank)
            
            if gen_args.dp:
                nll_val = reduced_mean(nll_epoch_val, n_samples_val)
                nll_test = reduced_mean(nll_epoch_test, n_samples_test)
            else:
                nll_val = nll_epoch_val / n_samples_val
                nll_test = nll_epoch_test / n_samples_test
            
            if rank == 0:
                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test
                    if gen_args.save_model and rank == 0:
                        gen_args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % gen_args.exp_name)
                        utils.save_model(model, 'outputs/%s/generative_model.npy' % gen_args.exp_name)
                        if gen_args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % gen_args.exp_name)
                        with open('outputs/%s/args.pickle' % gen_args.exp_name, 'wb') as f:
                            pickle.dump(args, f)
                        

                    if gen_args.save_model and rank == 0:
                        utils.save_model(optim, 'outputs/%s/optim_%d.npy' % (gen_args.exp_name, epoch))
                        utils.save_model(model, 'outputs/%s/generative_model_%d.npy' % (gen_args.exp_name, epoch))
                        if gen_args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema_%d.npy' % (gen_args.exp_name, epoch))
                        with open('outputs/%s/args_%d.pickle' % (gen_args.exp_name, epoch), 'wb') as f:
                            pickle.dump(args, f)
                print('Reduced mean Val loss: %.4f \t Reduced mean Test loss:  %.4f' % (nll_val, nll_test))
                print('Reduced mean Best val loss: %.4f \t Reduced mean Best test loss:  %.4f' % (best_nll_val, best_nll_test))
                
                wandb.log({"Val loss ": nll_val})
                wandb.log({"Test loss ": nll_test})
                wandb.log({"Best cross-validated test loss ": best_nll_test})
            if gen_args.dp: dist.barrier()


if __name__ == "__main__":
    main()
