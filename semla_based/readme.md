Our results:

python semlaflow/evaluate.py ckpt_path=/home/lizian/Self-Conditioned-3D-Diffusion/wandb/equinv-geom-drugs/unimolglobal_first4_reploss0.5_noise0.3_NEW/checkpoints/epoch299.ckpt n_molecules=5000 n_replicates=3 encoder_path=/home/lizian/Self-Conditioned-3D-Diffusion/checkpoints/encoder_ckpts/unimol_drug_and_originalDataset_global_1860000.pt rdm_ckpt=/home/lizian/Self-Conditioned-3D-Diffusion/outputs/rdm/drug_lowest_unimol_global/model/checkpoint-last.pth cfg_coef=-0.9 inv_temp=1.0  


Original:


semlaflow/train_drug.py version=original_rerun num_inference_steps=100 dropout=0.1 noise_sigma=0.3 rep_loss_weight=0.0 scheduled_noise=false sampler=GtSampler encoder_path=/home/lizian/Self-Conditioned-3D-Diffusion/unimol_pretrain_more_global_save/unimol_global_1860000.pt encoder_type=unimol_global_first4 original=true
