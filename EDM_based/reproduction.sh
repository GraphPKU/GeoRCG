

# For conditional generation of QM9
CUDA=3

# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_alpha/model/checkpoint-last.pth cfg=2.0 version=alpha_cfg2.0 property=alpha classifiers_path=../checkpoints/classifiers_ckpts/exp_class_alpha inv_temp=1.0


# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_Cv/model/checkpoint-last.pth cfg=2.0 version=cv_cfg2.0  property=cv classifiers_path=../checkpoints/classifiers_ckpts/exp_class_Cv inv_temp=1.0

# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_gap/model/checkpoint-last.pth cfg=2.0 version=gap_cfg2.0 property=gap classifiers_path=../checkpoints/classifiers_ckpts/exp_class_gap inv_temp=1.0


# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_homo/model/checkpoint-last.pth cfg=2.0 version=homo_cfg2.0 property=homo classifiers_path=../checkpoints/classifiers_ckpts/exp_class_homo inv_temp=1.0


# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_lumo/model/checkpoint-last.pth cfg=2.0 version=lumo_cfg2.0 property=lumo classifiers_path=../checkpoints/classifiers_ckpts/exp_class_lumo inv_temp=1.0


# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_mu/model/checkpoint-last.pth cfg=2.0 version=mu_cfg2.0 property=mu classifiers_path=../checkpoints/classifiers_ckpts/exp_class_mu inv_temp=1.0

 

# For DRUG

# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_analyze.py  batch_size_gen=100 gen_model_path=../checkpoints/gen_ckpts/edm_drug_unimol6layers_fullepochs_noise0.5/ rdm_ckpt=../checkpoints/rdm_ckpts/drug_unimol6layers_fullepochs/checkpoint-98.pth save_molecules=true inv_temp=0.5 cfg=0.0 version=drug_cfg0.0_invtemp0.5_S1000 ddim_S=1000

CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_analyze.py  batch_size_gen=100 gen_model_path=../checkpoints/gen_ckpts/edm_drug_unimol6layers_fullepochs_noise0.5/ rdm_ckpt=../checkpoints/rdm_ckpts/drug_unimol6layers_fullepochs/checkpoint-98.pth save_molecules=true inv_temp=0.5 cfg=0.0 version=drug_cfg0.0_invtemp0.5_S50 ddim_S=50


CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_analyze.py  batch_size_gen=100 gen_model_path=../checkpoints/gen_ckpts/edm_drug_unimol6layers_fullepochs_noise0.5/ rdm_ckpt=../checkpoints/rdm_ckpts/drug_unimol6layers_fullepochs/checkpoint-98.pth save_molecules=true inv_temp=0.5 cfg=0.0 version=drug_cfg0.0_invtemp0.5_S100 ddim_S=100


CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_analyze.py  batch_size_gen=100 gen_model_path=../checkpoints/gen_ckpts/edm_drug_unimol6layers_fullepochs_noise0.5/ rdm_ckpt=../checkpoints/rdm_ckpts/drug_unimol6layers_fullepochs/checkpoint-98.pth save_molecules=true inv_temp=0.5 cfg=0.0 version=drug_cfg0.0_invtemp0.5_S500 ddim_S=500


# For QM9 unconditional

# CUDA_VISIBLE_DEVICES=$CUDA python eval_src/eval_analyze.py n_samples=10000 cfg=1.0 gen_model_path=../checkpoints/gen_ckpts/edm_qm9_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_uncond/model/checkpoint-last.pth batch_size_gen=1000 version=qm9_uncond_cfg1.0_invtemp1.0