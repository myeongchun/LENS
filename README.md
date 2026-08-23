# LENS

Implementation of **L**earning an **E**nsemble of **N**ormality **S**ubspaces for Unsupervised Anomaly Detection

## Abstract

This paper proposes a boosting-based ensemble framework that represents diverse notions of normality across multiple latent spaces to mitigate the *information loss* inherent in dimensionality reduction. During latent-space construction, LENS quantifies the contribution of each sample using a *latent leverage score*. Samples with disproportionately high contributions are excluded through *leverage-aware filtering*, while *reliability-aware weighting* regulates the boosting process by assigning greater reliability to samples whose contributions remain stable across successive latent spaces. By integrating these mechanisms, LENS captures heterogeneous normal patterns through complementary latent representations for robust unsupervised anomaly detection.

## Dataset Description
We use 57 benchmark datasets from [ADBench](https://github.com/Minqi824/ADBench.git).

Download the datasets and place the NPZ files in the `datasets/` directory.

## Requirements

```txt
python==3.10.19
torch==2.6.0
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.1
psutil==7.2.1
openpyxl==3.1.5
```

## Running LENS

Run LENS on GPU 0:

```bash
python main.py --dataset "dataset name" --device 0
```

Run LENS on CPU:

```bash
python main.py --dataset "dataset name" --device cpu
```

Use a GPU when available and otherwise fall back to the CPU:

```bash
python main.py --dataset "dataset name" --device auto
```

Run all available datasets:

```bash
python main.py --dataset all --device 0
```

Specify a different random seed with `--seed`:

```bash
python main.py --dataset "dataset name" --device 0 --seed 1
```
