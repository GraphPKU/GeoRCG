# Geometric Representation Condition Improves Equivariant Molecule Generation

This repository is the official PyTorch implementation of the paper ["Geometric Representation Condition Improves Equivariant Molecule Generation"](https://arxiv.org/pdf/2410.03655).

## Table of Contents


- [Environment Setup](#environment-setup)
- [Checkpoints](#checkpoints)
- [Datasets](#datasets)
- [Training](#training)  
  - [RDM Training](#rdm-training)  
  - [Molecule Generator Training](#molecule-generator-training)  
  - [GeoRCG (Semla) Training](#georcg-semla-training)
- [Evaluation](#evaluation)  
  - [GeoRCG (EDM) Evaluation](#georcg-edm-evaluation)  
  - [GeoRCG (Semla) Evaluation](#georcg-semla-evaluation)




---

## Environment Setup

To install the required packages, run:

```bash
bash ./install.sh
```

---

## Checkpoints

You can download all the checkpoints used in the paper from [this link](https://drive.google.com/drive/folders/1_c8gTFLlpiiFmKA_h12EvD25cGK_d-CQ?usp=sharing), and place them in the `checkpoints` directory. 

---

## Datasets

- **QM9 dataset**: This dataset will be automatically downloaded when you run the EDM training script.

- **GEOM-DRUG dataset**: Follow the instructions on the [EDM GitHub](https://github.com/ehoogeboom/e3_diffusion_for_molecules) to download this dataset.

- **GEOM-DRUG dataset (Semla)**: Semlaflow uses a smaller version of GEOM-DRUG with lower-energy conformations. You can download it from the [SemlaFlow GitHub](https://github.com/rssrwn/semla-flow).

---

## Training

💡 You can train both the RDM and the Molecule Generator simultaneously, as the RDM is not required for sampling during Molecule Generator training (we use the training dataset for sampling).

### RDM Training

- **QM9 Dataset (Unconditional)**:

  To train an RDM that produces Frad representation on the QM9 dataset:

  ```bash
  python src_GeoRCG/train_RDM.py experiments_RDM=qm9_uncond
  ```

  We use Frad encoder's public checkpoint, available [here](https://drive.google.com/file/d/1O6f6FzYogBS2Mp4XsdAAEN4arLtLH38G/view?usp=share_link). You can also download it from our provided checkpoints at `checkpoints/encoder_ckpts/qm9_frad.ckpt`.

- **QM9 Dataset (Conditional)**:

  To train a conditional RDM for the second half of the QM9 dataset:

  ```bash
  python src_GeoRCG/train_RDM.py experiments_RDM=qm9_cond experiments_RDM.rdm_args.conditioning=["lumo"] experiments_RDM.rdm_args.exp_name="qm9_cond_lumo"
  ```

  Modify the `"lumo"` property for other properties like `"alpha"` or `"homo"`.

- **GEOM-DRUG Dataset**:

  To train an RDM on the GEOM-DRUG dataset:

  ```bash
  python src_GeoRCG/train_RDM.py experiments_RDM=drug
  ```

    > The Unimol encoder was trained by ourselves for two primary reasons:
    > 1. The provided checkpoint by [Unimol](https://github.com/deepmodeling/Uni-Mol) was trained on a setting with the hydrogen (H) atom removed, whereas our setup requires hydrogen to be included.
    > 2. The provided checkpoint was trained on datasets that do not contain some of the rarer elements found in the GEOM-DRUG dataset. As a result, we pretrained the encoder on a more comprehensive dataset that includes GEOM-DRUG.

    > In initial experiments, we used a 15-layer Unimol encoder, which should replicate the results in the GeoRCG paper. In later experiments, we found that a 6-layer Unimol encoder performed more effectively for the generation task, which may be due to a relatively lower Lipschitz constant for the conformations of the 6-layer encoder. As of now, the default configuration uses the 6-layer Unimol encoder. You can find the checkpoint for this model at `checkpoints/encoder_ckpts/drug_unimol_6layers_noise1.5.pt`. Alternatively, you can pretrain the encoder yourself using the provided script located in `EDM_based/models_GeoRCG/unimol/unimol`.

- **SemlaFlow Setting**:

  To train an RDM on GEOM-DRUG dataset using the SemlaFlow data:

  ```bash
  python src_GeoRCG/train_RDM.py experiments_RDM=drug experiments_RDM.rdm_args.semlaflow_data=true experiments_RDM.rdm_args.encoder_type=unimol_global experiments_RDM.rdm_args.encoder_path=../checkpoints/encoder_ckpts/drug_unimol_global_iter1.8M.pt
  ```

  > In the SemlaFlow setting, we experiment with two configurations:
  > 1. **Unimol Global Configuration** (default setting in this code):  
    We train a modified version of the 15-layer Unimol model, referred to as `unimol_global`, and use the first 4 layers for representations. The modified model includes a global output head and a global pretext pretraining task, which helps achieve a lower Lipschitz constant for the conformations, benefiting GeoRCG. The checkpoint for this configuration can be found at `checkpoints/encoder_ckpts/drug_unimol_global_iter1.8M.pt`. You can also pretrain this version using the provided task file at `models_GeoRCG/unimol/unimol/tasks/unimol_global.py`.
  > 2. **Standard Unimol Configuration**:  
    In this configuration, we train a standard 15-layer Unimol model and use the first 4 layers for representations. The checkpoint for this model is located at `../checkpoints/encoder_ckpts/drug_unimol_15layers_noise1_iter2M.pt`. You can train RDM under this configuration by modifying the `encoder_type` option to `unimol_truncated` and specifying the corresponding checkpoint path: `encoder_path=../checkpoints/encoder_ckpts/drug_unimol_15layers_noise1_iter2M.pt`. Similarly, modify the `encoder_type` and `encoder_path` options in the Semlaflow molecule generator training to match the desired configuration. This configuration provides improved Strain and Energy metrics, but it slightly sacrifices stability and validity metrics. 

---

### Molecule Generator Training

- **QM9 Dataset**:

  To train a molecule generator on the QM9 dataset:

  ```bash
  python src_GeoRCG/train_gen.py experiments_gen=qm9
  ```

- **GEOM-DRUG Dataset**:

  To train on the GEOM-DRUG dataset using 4 GPUs:

  ```bash
  python -m torch.distributed.run --nproc_per_node=4 --master-port=20001 src_GeoRCG/train_gen.py experiments_gen=drug
  ```

  For single GPU, disable distributed processing:

  ```bash
  python src_GeoRCG/train_gen.py experiments_gen=drug experiments_gen.gen_args.dp=false
  ```

---

### GeoRCG (Semla) Training

To train on the GEOM-DRUG dataset using SemlaFlow:

```bash
python src_GeoRCG/train_drug.py experiments_gen=drug
```

---

## Evaluation

### GeoRCG (EDM) Evaluation

- **Unconditionally Generated QM9 Molecules**:

  ```bash
  python eval_src/eval_analyze.py  cfg=1.0 inv_temp=1.0 gen_model_path=../checkpoints/gen_ckpts/edm_qm9_frad_noise0.3 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_qm9_frad_uncond/model/checkpoint-last.pth
  ```


  Feel free to adjust the `cfg` and `inv_temp` parameters as needed. The `cfg` parameter controls the sampling temperature, while `inv_temp` sets the inverse temperature for sampling.

- **Unconditionally Generated GEOM-DRUG Molecules**:

  ```bash
  python eval_src/eval_analyze.py cfg=1.0 inv_temp=1.0 gen_model_path=../checkpoints/gen_ckpts/edm_drug_unimol6layers_noise0.5 rdm_ckpt=../checkpoints/rdm_ckpts/rdm_drug_unimol_6layers/checkpoint-98.pth
  ```

You can adjust the sampling step number of above  by adding `ddim_S=10` or other values.


- **Conditionally Generated QM9 Molecules**:

  ```bash
  python eval_src/eval_conditional_qm9.py  classifiers_path=../checkpoints/classifiers_ckpts/exp_class_alpha property=alpha gen_model_path=../checkpoints/gen_ckpts/edm_qm9_second_half_frad_noise0.3  rdm_ckpt=../checkpoints/rdm_ckpts/rdm_qm9_frad_alpha/model/checkpoint-last.pth cfg=2.0  inv_temp=1.0
  ```

  Change the `alpha` property to other properties like `lumo` or `homo`, and update the corresponding paths accordingly.



---

### GeoRCG (Semla) Evaluation

To evaluate unconditionally generated GEOM-DRUG molecules using SemlaFlow:

```bash
python evaluate.py ckpt_path=../checkpoints/gen_ckpts/semlaflow_unimol_global_truncated4_noise0.3_reploss0.5/checkpoints/last.ckpt rdm_ckpt=../checkpoints/rdm_ckpts/rdm_drug_semla_unimol_global_truncated4/model/checkpoint-last.pth n_molecules=10000 n_replicates=3 cfg_coef=-0.9 batch_cost=2048 integration_steps=100
```

Change `integration_steps` to adjust the sampling step number.

> Another version of GeoRCG (Semla) produces better Strain and Energy metrics but slightly worse stability and validity, as mentioned in [this section](#rdm-training). You can use the following command to evaluate this version:
> ```bash
> `ckpt_path=../checkpoints/gen_ckpts/semlaflow_unimol_truncated4_noise0.3_reploss0.1/checkpoints/last.ckpt rdm_ckpt=../checkpoints/rdm_ckpts/rdm_drug_semla_unimol_truncated4`
> ```


---


## Citation

If you find this repository useful and use it in your research, please cite our paper:

```bibtex
@article{li2024geometric,
  title={Geometric Representation Condition Improves Equivariant Molecule Generation},
  author={Li, Zian and Zhou, Cai and Wang, Xiyuan and Peng, Xingang and Zhang, Muhan},
  journal={arXiv preprint arXiv:2410.03655},
  year={2024}
}
```


## Acknowledgements

This code repository is built upon the following works:

- [RCG](https://github.com/LTH14/rcg)
- [EDM](https://github.com/ehoogeboom/e3_diffusion_for_molecules)
- [Unimol](https://github.com/deepmodeling/Uni-Mol)
- [Frad](https://github.com/fengshikun/Frad)

Thanks for all the authors for their contributions!