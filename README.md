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
├── datasets/
├── feature_methods/
├── features.py
├── figs/
├── get_features.py
├── model_saves/
├── notebooks/
├── numpy_files/
├── ood_detection.py
├── README.md
├── requirements.txt
├── train.py
└── utils.py
```
## Training models
Training each feature-extraction method (convolutional autoencoder, supervised CNN, supervised contrastive-learning) is made available using a single script in project directory: ```train.py```. Bash scripts containing training hyperparameters are ```bash_scripts/autoencoder_runner.py```, ```bash_scripts/cnn_runner.py```, ```bash_scripts/ctr_runner.py```. Running these scripts will save .pt files to the ```model_saves/``` directory.
## Extract features
```features.py``` extracts features using each of the three approaches. Command-line arguments are each method's model path; see ```bash_scripts/features.py``` for an example. Numpy compressed files will be saved to ```numpy_files```.
## OOD Detection
The extracted features are then statistically evaluated for OOD detection using the ```ood_detection.py``` script. Command-line arguments are metric (cosine or mahalonobis) and method (see above). Bootstrapped confidence intervals for accuracy, precision, and recall are printed to screen; a sample OOD detection plot (in the style of SPC charts) is saved to ```figs/```.
## SPC-based monitoring experiments
See ```SPC_Charts/CT_SPC_Simulation.ipynb``` for simulation experiments.