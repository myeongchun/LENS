# LENS
Boosting Autoencoders via **L**atent-d**EN**sity aware **S**ampling for Unsupervised Anomaly Detection

## Abstract
This paper aims to construct a boosting ensemble of autoencoders that handles diversity in data and are robust to data contamination.

## Dataset description
<img width="373" height="932" alt="image" src="https://github.com/user-attachments/assets/9dfb86ce-5e50-432e-a43d-8a2affa55a31" />


## Running LENS
In this paper, we utilized 47 tabular benchmark datasets from ADBench: [ADBench](https://github.com/Minqi824/ADBench.git)

### Requirement
```txt
python==3.10.19
torch==2.6.0
numpy==1.26.4
scipy==1.15.3
scikit-learn==1.7.2
tqdm==4.67.1
```

To run **LENS**, follow commands:
```bash
python main.py --dataset "dataset name" --device 0
```
If you want to run the model on all available datasets, enter all as the dataset name:
```bash
python main.py --dataset all --device 0
```
