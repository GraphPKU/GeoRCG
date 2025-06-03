**Reproducing GeoRCG (Semla) on DRUG**

CUDA_VISIBLE_DEVICES=2 python semlaflow/evaluate.py \
ckpt_path=checkpoints/gen_ckpts/semlaflow_unimolglobal_first4_reploss0.5_noise0.3/checkpoints/last.ckpt n_molecules=2000 n_replicates=3 \
encoder_path=checkpoints/encoder_ckpts/unimol_drug_and_originalDataset_global_1860000.pt \
rdm_ckpt=checkpoints/rdm_ckpts/drug_lowest_unimol_global_first4_resume/model/checkpoint-last.pth \
cfg_coef=0.0 \
inv_temp=1.0 \
integration_steps=100

**Reproducing GeoRCG (EDM) on QM9**

CUDA_VISIBLE_DEVICES=2 python eval_src/eval_analyze.py \
n_samples=5000 \
cfg=1.0 \
gen_model_path=checkpoints/gen_ckpts/gen_batch128_noisy0.3_drop0.2_pretrainedEnc \
sampler=PCSampler \
rdm_ckpt=rdm_ckpts/rdm_batch128_pretrainedEnc/model/checkpoint-last.pth \
batch_size_gen=1000


**Reproducing GeoRCG (EDM) on DRUG**
CUDA_VISIBLE_DEVICES=0,2,3,7 python -m torch.distributed.run --nproc_per_node=4 --master-port=20001 eval_src/eval_analyze.py \
batch_size_gen=25 \
cfg=1.0 \
inv_temp=1.0 \
n_samples=5000 \
n_steps=5 \
gen_model_path=checkpoints/gen_ckpts/Drug_unimol_twoattn_repproj_0.5/ \
rdm_ckpt=checkpoints/rdm_ckpts/Drug_unimol_huge/checkpoint-last.pth \
sampler=PCSampler \
save_molecules=true \
use_dist=true \
version=unimol_huge_cfg1invtemp1_S1000

**If you want to evaluate saved molecules**
python eval_src/eval_analyze.py \
saved_molecules_path=checkpoints/gen_ckpts/Drug_unimol_twoattn_repproj_0.5/molecules_5000_20250327105019.pt \
use_dist=false \
dataset=geom \
remove_h=false


**Reproducing GeoRCG (Semlaflow) on QM9 with conditional generation**

python semlaflow/evaluate_conditional.py \
ckpt_path=checkpoints/gen_ckpts/semlaflow_qm9_second_half_reploss0.1_noise0.3/epoch300.ckpt \
property=lumo \
rdm_ckpt=checkpoints/rdm_ckpts/rdm_batch128_pretrainedEncCrossAttn_LUMO \
n_molecules=3000 \
n_replicates=3 \
cfg_coef=1.0 \
batch_cost=2048 \
integration_steps=100


**Reproducing Semlaflow on QM9 with conditional generation**

python semlaflow/evaluate_conditional.py \
ckpt_path=checkpoints/gen_ckpts/semlaflow_original_alpha/epoch360.ckpt \
property=alpha \
rdm_ckpt=checkpoints/rdm_ckpts/rdm_batch128_pretrainedEncCrossAttn_ALPHA \
n_molecules=3000 \
n_replicates=3 \
cfg_coef=1.0 \
batch_cost=2048 \
integration_steps=100


**Training Semlaflow original on QM9 with conditional generation**
python semlaflow/train_qm9_conditional.py \
original=true \
property_condition=true \
property=alpha \
version=alpha







python eval_src/eval_conditional_qm9.py gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_batch128_pretrainedEncCrossAttn_ALPHA/model/checkpoint-last.pth cfg=2.