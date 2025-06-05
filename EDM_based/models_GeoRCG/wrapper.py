import torch
import logging
from time import time


class SelfConditionWrappedSampler(torch.nn.Module):
    def __init__(
            self,
            gen_sampler,
            rdm_sampler
    ):
        super().__init__()
        self.rdm_sampler = rdm_sampler
        self.gen_sampler = gen_sampler
        
        
    @torch.no_grad()
    def sample(
        self, 
        n_samples, 
        n_nodes, 
        node_mask, 
        edge_mask, 
        context, 
        fix_noise=False,
        fixed_rep=None,
        rep_context=None,
        ddim_S=None
        ):
        assert len(node_mask.shape) == 2 or node_mask.shape == (node_mask.shape[0], node_mask.shape[1], 1), f"node mask should have shape (batch size, max n nodes) or (batch size, max n nodes, 1), while it now has shape {node_mask.shape}"
        assert context == None, "We always use unconditional sampling in gen."
        
        
        device = node_mask.device
        batch_size, max_n_nodes = node_mask.shape[0], node_mask.shape[1]
        nodesxsample = node_mask.sum(-1) if len(node_mask.shape) == 2 else node_mask.sum(-1).sum(-1)
        nodesxsample = nodesxsample.to(torch.int64)
        
        
        # To sample rep first. We only use context in rep samplers, and gen sampler is always unconidtional.
        self.rdm_sampler.eval()
        if fixed_rep is None:
            print("Sampling Reps ...")
            
            s = time()
            sampled_rep = self.rdm_sampler.sample(
                nodesxsample=nodesxsample,
                device=device,
                additional_cond=rep_context,
                running_batch_size=batch_size
            )
            logging.info(f"RDM sampling of {batch_size} samples took {time() - s}s. {(time() - s)/batch_size}s for each sample in average.")
            
            print("Reps Sampling Done!")
        else:
            print("Using provided fixed rep.")
            sampled_rep = fixed_rep
        
        noise_scale = 0.
        sampled_rep = sampled_rep * torch.randn(
            sampled_rep.shape, 
            device=device
        ) * noise_scale
        print(f"Adding {noise_scale} noise to the sampled rep.")
            
        # Now sample from gen conditioning on the rep
        print("Sampling Molecules conditioning on Reps ...")
        
        self.gen_sampler.eval()
        s = time()
        if ddim_S is None:
            x, h = self.gen_sampler.sample(
                n_samples=batch_size, 
                n_nodes=max_n_nodes,
                node_mask=node_mask, 
                edge_mask=edge_mask, 
                context=None, # NOTE: Always unconditional!
                fix_noise=fix_noise, 
                rep=sampled_rep
                )
        else:
            x, h = self.gen_sampler.ddim_sample(
                n_samples=batch_size, 
                n_nodes=max_n_nodes,
                node_mask=node_mask, 
                edge_mask=edge_mask, 
                context=None, # NOTE: Always unconditional!
                fix_noise=fix_noise, 
                rep=sampled_rep,
                S=ddim_S
                )
        logging.info(f"gen sampling of {batch_size} samples took {time() - s}s. {(time() - s)/batch_size}s for each sample in average.")
        
        print("Molecules Sampling Done!")

        return x, h
    
    @torch.no_grad()
    def sample_chain(
        self, 
        n_samples, 
        n_nodes, 
        node_mask, 
        edge_mask, 
        context, 
        keep_frames=None, 
        rep=None,
        fixed_rep=None,
        rep_context=None
        ):
        assert n_samples == 1, f"For chain sampling, we should ensure that n_samples equals 1, but it now equals {n_samples}"
        assert len(node_mask.shape) == 2 or node_mask.shape == (node_mask.shape[0], node_mask.shape[1], 1), f"node mask should have shape (batch size, max n nodes) or (batch size, max n nodes, 1), while it now has shape {node_mask.shape}"
        assert context == None, "We always use unconditional sampling in gen."
        
        device = node_mask.device
        batch_size, max_n_nodes = node_mask.shape[0], node_mask.shape[1]
        nodesxsample = node_mask.sum(-1) if len(node_mask.shape) == 2 else node_mask.sum(-1).sum(-1)
        nodesxsample = nodesxsample.to(torch.int64)
        
        
        # To sample rep first. We only use context in rep samplers, and gen sampler is always unconidtional.
        self.rdm_sampler.eval()
        if fixed_rep is None:
            print("Sampling Reps ...")
            sampled_rep = self.rdm_sampler.sample(
                nodesxsample=nodesxsample,
                device=device,
                additional_cond=rep_context,
                running_batch_size=batch_size
            )
            print("Reps Sampling Done!")
        else:
            print("Using provided fixed rep.")
            sampled_rep = fixed_rep
            
        # Now sample from gen conditioning on the rep
        print("Sampling Molecules conditioning on Reps ...")
        self.gen_sampler.eval()
        chain = self.gen_sampler.sample_chain(n_samples, n_nodes, node_mask, edge_mask, context, keep_frames=keep_frames, rep=sampled_rep)
        print("Molecules Sampling Done!")

        return chain
