# LENS
Boosting Autoencoders via Latent-density aware Sampling for Unsupervised Anomaly Detection

## Abstract
This study aims to construct a boosting ensemble of autoencoders that handles diverse data patterns and are robust to data contamination.

## Dataset description
<img width="373" height="932" alt="image" src="https://github.com/user-attachments/assets/9dfb86ce-5e50-432e-a43d-8a2affa55a31" />


## Running LENS
In this paper, we utilized 47 tabular benchmark datasets from ADBench: [ADBench](https://github.com/Minqi824/ADBench.git)

To run **LENS**, follow commands:
```bash
python main.py --dataset "dataset name" --device 0
```
If you want to run the model on all available datasets, enter all as the dataset name:
```bash
python main.py --dataset all --device 0
```
