# Instructions
## Overview
Task: develop image features that enable in-modality OOD detection, specifically on the task of sifting out axial CTs from a dataset that also includes sagittal and coronal views.
Data: MedicalMNISTv2, Organ{A/C/S}MNIST
## Folder structure
Construct folder structure as follows.
```
├── bash_scripts/
├── cfg.json
├── data/
├── features.py
├── figs/
├── medmnist_datasets.py
├── feature_methods
│   ├── conv_autoencoder.py
│   └── resnet_big.py
├── numpy_files/
├── ood_detection.py
├── README.md
├── requirements.txt
├── saves/
├── train_autoencoder.py
├── train_cnn.py
├── train_contrastive.py
└── utils.py
```

## 1.) Training models

Training scripts for each feature-extraction method (convolutional autoencoder, supervised CNN, supervised contrastive-learning) available in project directory: ```train_autoencoder.py```, ```train_cnn.py```, ```train_contrastive.py```. Respective bash scripts containing training hyperparameters are ```bash_scripts/autoencoder.sh```, ```bash_scripts/cnn.sh```, ```bash_scripts/ctr.sh```. Running these scripts will save .pt files to the ```saves/``` directory.

## 2.) Extract features

```features.py``` extracts features using each of the three approaches. Command-line arguments are each method's model path; see ```bash_scripts/features.py``` for an example. Numpy compressed files will be saved to ```numpy_files```.

## 3.) OOD Detection

The extracted features are then statistically evaluated for OOD detection using the ```ood_detection.py``` script. Command-line arguments are metric (cosine or mahalonobis) and method (see above). Bootstrapped confidence intervals for accuracy, precision, and recall are printed to screen; a sample OOD detection plot (in the style of SPC charts) is saved to ```figs/```.

## CUSUM & Bernoulli CUSUM

See ```notebooks/CT_SPC_Charts.ipynb``` for CUSUM and Bernoulli CUSUM plots for a simulated scenario over 100 days.