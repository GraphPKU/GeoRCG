# Geometric Representation Condition Improves Equivariant Molecule Generation

This repository contains the PyTorch implementation of the paper **Geometric Representation Condition Improves Equivariant Molecule Generation**.

## Dependencies

To get started, you'll need the following dependencies:

- `torch==2.4.1`
- `torch-geometric==2.6.0`
- `pyg-lib==0.4.0+pt24cu121`
- `torch-scatter==2.1.2+pt24cu121`
- `torch-sparse==0.6.18+pt24cu121`
- `torch-spline-conv==1.2.2+pt24cu121`
- `torch-cluster==1.6.3+pt24cu121`
- `hydra-core==1.3.2`
- `networkx==3.1`
- `posebusters==0.3.1`
- `unicore==0.0.1` (only when you use unimol as the encoder.)

For detailed setup instructions, refer to the `environment.yml` file provided in the repository.

## Usage

### Pre-trained Encoder

For the QM9 dataset, we leverage [Frad](https://github.com/fengshikun/Frad) as the geometric encoder. You can download the pre-trained weights [here](https://drive.google.com/file/d/1O6f6FzYogBS2Mp4XsdAAEN4arLtLH38G/view?usp=share_link).  

For the GEOM-DRUG dataset, we pre-trained [Unimol](https://openreview.net/forum?id=6K2RM6wVqKu) using their official codebase on GEOM-DRUG dataset itself (only use the training dataset to avoid data leak). If you opt to use Unimol as the encoder, ensure you install [uni-core](https://github.com/dptech-corp/Uni-Core).

### Dataset

The QM9 dataset is automatically downloaded upon execution. For the GEOM-DRUG dataset, please follow the instructions in the `README.md` provided in the [EDM GitHub repository](https://github.com/ehoogeboom/e3_diffusion_for_molecules). Notice that the EDM code is integrated into this codebase, which should simplify the setup process.

### Config Management

This project uses [Hydra](https://hydra.cc/) for configuration management. The relevant configuration files are located in `configs`. You can conveniently adjust specific configs in command-line. See the [documentation](https://hydra.cc/) for detailed use.

### Training the Representation Generator

You can train the representation generator using the following commands for QM9 and DRUG dataset, respectively.

```bash
python src/self_condition_train_qm9_RDM.py
python src/self_condition_train_drug_RDM.py
```

The config files are:

- `./configs/qm9_rdm_config.yaml` for QM9
- `./configs/drug_rdm_config.yaml` for GEOM-DRUG

These configuration files can be modified to fit your specific requirements. Ensure you update `qm9_rdm.rdm_args.encoder_path` for QM9 (or `rdm_args.encoder_path` for GEOM-DRUG) to point to the path of your downloaded encoder checkpoint.

If you wish to conditionally train the representation generator (only for QM9), change the argument `qm9_rdm` in `./configs/qm9_rdm_config.yaml` as `qm9_rdm=qm9_rdm_conditional`. You can then modify the hyperparameters in `./configs/qm9_rdm/qm9_rdm_conditional.yaml` as needed. 

### Training the Molecule Generator

To train the molecule generator, use the following commands:

```bash
python src/self_condition_train_qm9.py
python src/self_condition_train_drug.py
```

The configuration files are:

- `./configs/qm9_pcdm_config.yaml` for QM9
- `./configs/drug_pcdm_config.yaml` for GEOM-DRUG

We note that for the GEOM-DRUG dataset, training was conducted on two Nvidia A100 GPUs with a batch size of 64 using `torch.distributed.run`. If you lack access to resources, you can reduce the batch size to match your hardware capabilities (but may lead to different model performance). Additionally, be sure to update the `pcdm_args.encoder_path` in the configuration file with your downloaded encoder checkpoint path.

In our experiments, the molecule generator is always trained unconditionally, but using datasets of different sizes. For conditional molecule generation, simply modify the dataset from `qm9` to `qm9_second_half`.

### Evaluation

#### Unconditional Evaluation

For unconditional evaluation, you can use the following command:

```bash
python ./eval_src/eval_analyze.py
```

The configuration file for this evaluation is `./configs/eval_analyze.yaml`. 

Be sure to specify the paths to checkpoints of the molecule and representation generators you want to evaluate. You may also adjust various hyper-parameters such as the number of molecules to sample (`n_samples`), the batch size for generation (`batch_size_gen`), the sampler used (`sampler`. You can specify it as `GtSampler` for generating representations from the training dataset or `PCSampler` for using your representation generator. If you're using `GtSampler`, be sure to specify the encoder checkpoint path in `encoder_path`.), and the balacing control parameters `inv_temp`, `cfg`.

For evaluating MiDi metrics (e.g., BondLengthW1 and stability with relaxed valency), enable `eval_midi=true` and, if desired, use OpenBabel for bond calculation by setting `midi_args.use_openbabel=true`. If you use openbabel, ensure that OpenBabel is installed on your system; installation instructions can be found [here](https://openbabel.org/docs/Installation/install.html).

#### Conditional Evaluation

For conditional evaluation, you can run:

```bash
python ./eval_src/eval_conditional_qm9.py
```

The configuration file for this is `./configs/eval_conditional.yaml`. 

Most configs are similar with those in unconditional evaluation. Additionally, you will need to specify the `classifiers_path` pointing to the EGNN classifier checkpoint you trained for evaluating the quality of sampled molecules. You can follow the README.md in the [EDM codebase](https://github.com/ehoogeboom/e3_diffusion_for_molecules) for the classifier training, and note that EDM's code is already integrated into our repository.

#### Visualization

For conditional evaluation, you can run:

```bash
python ./eval_src/eval_visualize_samples.py
```

The configuration file for this is `./configs/eval_visualize_samples.yaml`. 


Most configs are similar with those in unconditional evaluation. Additionally, you can perform conditional sampling by specifying `property` to some property from `[alpha, Cv, mu, gap, homo, lumo]`. If you want to perform sweep sampling, set `sweep` to `true` and set the `start_value` and `end_value`.