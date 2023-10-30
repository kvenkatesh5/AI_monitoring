"""
MedMNIST OOD Detection: Phase 1
Identify out-of-modality images in a dataset of abnominal CTs (from three views: axial, saggital, coronal).
"""

import os
import json
import random
import time
import sys
sys.path.append("..")
from abc import ABC, abstractmethod

from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score
from scipy.spatial.distance import cdist
from networks.resnet_big import SupConResNet

from medmnist_datasets import load_default_data
from medmnist_datasets import matrixify

"""
Datasets

Source: https://medmnist.com/

Task: Distinguish axial views from sagittal/coronal views on abdominal CT scans from the Organ{A/C/S} Medical MNIST dataset. (The axial view was chosen as "OOD" since it was the most difficult for a logistic regression classifier to distinguish.)
    "Positive Dataset": Axial CT scans
    "OOD Dataset": Coronal/Sagittal CT scans

Dataset Transforms:  ToTensor, Normalization (mu=0.5, std=0.5)

Training Transforms: RandomResized Crop, Rotations (+- 10 degrees), ToTensor, Normalization (mu=0.5, std=0.5)
"""

"""Compute distances per test image against train distribution"""
def compute_distance(feature_train, feature_test):
    # ntest x ntrain
    time1 = time.time()
    matrix = cdist(feature_test, feature_train, metric="cosine")
    print("...Computed distances! {:.4f} secs".format(time.time()-time1))
    return matrix

"""Evaluate a feature space, binary labels"""
class EvaluateFeatureSpace(ABC):
    def __init__(self, model: nn.Module, device, y_train: np.ndarray, y_test: np.ndarray,
                 training_set, testing_set,
                 subsample=True, n_sub=1000, random_seed=2023):
        random.seed(random_seed)
        
        self.device = device
        self.model = model
        self.model = self.model.to(self.device)
        self.feature_train = self.get_features(training_set)
        self.y_train = y_train
        self.feature_test = self.get_features(testing_set)
        self.y_test = y_test
        
        # Save the original features
        self.original_attrs = {
            "y_train": self.y_train,
            "training_features": self.feature_train,
            "y_test": self.y_test,
            "testing_features": self.feature_test
        }
        
        self.subsample = subsample
        
        if self.subsample:
            idxs1 = random.sample(list(range(self.feature_train.shape[0])), k=n_sub)
            idxs2 = random.sample(list(range(self.feature_test.shape[0])), k=n_sub)
            self.feature_train = self.feature_train[idxs1,:]
            self.y_train = y_train[idxs1]
            self.feature_test = self.feature_test[idxs2,:]
            self.y_test = y_test[idxs2]
        
        # Make a mask to easily query where the positive images are
        self.y_train_mask = self.y_train==1
        self.y_test_mask = self.y_test==1
        
    @abstractmethod
    def get_features(self, dset: Dataset):
        pass
    
    def eval_internal(self, suptitle):
        dist_matrix = compute_distance(self.feature_train, self.feature_test)
        dists = [[[], []], [[], []]]
        # compute distributions
        m00 = dist_matrix[~(self.y_test_mask),:]
        m00 = m00[:,~(self.y_train_mask)]
        dists[0][0] = np.mean(m00, axis=1)

        m01 = dist_matrix[~(self.y_test_mask),:]
        m01 = m01[:,self.y_train_mask]
        dists[0][1] = np.mean(m01, axis=1)

        ood_dist = np.mean(dist_matrix[~(self.y_test_mask),:], axis=1)

        m10 = dist_matrix[self.y_test_mask,:]
        m10 = m10[:,~(self.y_train_mask)]
        dists[1][0] = np.mean(m10, axis=1)

        m11 = dist_matrix[self.y_test_mask,:]
        m11 = m11[:,self.y_train_mask]
        dists[1][1] = np.mean(m11, axis=1)

        id_dist = np.mean(dist_matrix[self.y_test_mask,:], axis=1)
        
        # plots
        fig, ax = plt.subplots(1,2, sharey=True)
        fig.tight_layout()

        # cosine distance ranges from 0-2
        # see https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html
        ax[0].set_xlim([0,2])
        ax[1].set_xlim([0,2])
        ax[0].set_ylim([0,0.5])

        sns.histplot(dists[1][0], color="green", label="tr=0", kde=True, stat="probability", ax=ax[0])
        sns.histplot(dists[1][1], color="black", label="tr=1", kde=True, stat="probability", ax=ax[0])
        sns.histplot(id_dist, color="orange", label="total", kde=True, stat="probability", ax=ax[0])

        sns.histplot(dists[0][0], color="red", label="tr=0", kde=True, stat="probability", ax=ax[1])
        sns.histplot(dists[0][1], color="blue", label="tr=1", kde=True, stat="probability", ax=ax[1])
        sns.histplot(ood_dist, color="orange", label="total", kde=True, stat="probability", ax=ax[1])

        ax[0].set_title("ID Cosine Distance")
        ax[1].set_title("OOD Cosine Distance")
        ax[0].legend()
        ax[1].legend()

        plt.subplots_adjust(top=0.80)
        fig.suptitle(suptitle)

        # fig.savefig("figs/internal_eval2.png")

"""Forward hook for ResNet"""
cnn_layers = {}
def get_inputs(name):
    def hook(model, inpt, output):
        cnn_layers[name] = inpt[0].detach()
    return hook

"""Load CNN model"""
def load_cnn(pth):
    print("Loading! <= {}".format(os.path.basename(pth)))
    state = torch.load(pth)
    hparams = state["hparams"]
    model = resnet18(pretrained=hparams["pretrained"])
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=2),
        torch.nn.Sigmoid()
    )
    model.load_state_dict(state["model"])
    h = model.fc.register_forward_hook(get_inputs('fts'))
    model = model.to(hparams["device"]) 
    return model

"""Evaluate CNN feature space"""
class EvaluateCNN(EvaluateFeatureSpace):
    def get_features(self, dset):
        self.model.eval()
        features = []
        ys = []
        yhats = []
        ldr = DataLoader(dset, batch_size=32, shuffle=False)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
                targets = targets.to(self.device)
                # Pass images through model to update cnn_layers dictionary
                outputs = self.model(images)
                ys.extend(targets.detach().cpu().argmax(dim=1).tolist())
                yhats.extend(outputs[:,1].detach().squeeze().cpu().tolist())
                # Get the (just populated!) features
                features.append(cnn_layers['fts'].detach().cpu().numpy())
        auroc = roc_auc_score(ys, yhats)
        print(f"AUROC: {auroc:.4f}")
        return np.vstack(features)
    
"""Evaluate CTR features"""
class EvaluateCTR(EvaluateFeatureSpace):
    def get_features(self, dset):
        self.model.eval()
        features = []
        ldr = DataLoader(dset, batch_size=32)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                outputs = self.model.module.encoder(images)
                norm_outputs = F.normalize(outputs)
                features.append(norm_outputs.detach().cpu().numpy())
        return np.vstack(features)
    
"""Load contrastive model"""
def load_ctr(pth):
    d = torch.load(pth)
    model = SupConResNet(name=d["opt"]["model"], head=d["opt"]["projection"])
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
    try:
        model.load_state_dict(d["model"])
    except Exception:
        raise ValueError("Incorrect model preconditions provided.")
    return model

"""Main fxn: extract each approach's features here
since load_default_data changes ordering between calls"""
def main():
    # Cfg
    with open("cfg.json", "r") as f:
        cfg = json.load(f)

    # Set GPU vis
    use_gpus = "5,6"
    if use_gpus != 'all':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = use_gpus
    device = 'cpu'
    ncpus = os.cpu_count()
    dev_n = ncpus
    if torch.cuda.is_available():
        device = 'cuda'
        dev_n = torch.cuda.device_count()
    print(f"Device: {device} | # {dev_n}")

    # Load data
    # NOTE: this is not consistent between calls, so all features will be computed using this data load
    train_set, val_set, test_set = load_default_data({
        "data_dir": cfg["data_dir"],
        "positive_dataset": "organamnist",
    })

    # Matrixify datasets
    Xtr, ytr = matrixify(train_set)
    Xvl, yvl = matrixify(val_set)
    Xtt, ytt = matrixify(test_set)
    np.savez(os.path.join(cfg["data_dir"], "../numpy_files/data_splits"),
        Xtr=Xtr, ytr=ytr,
        Xvl=Xvl, yvl=yvl,
        Xtt=Xtt, ytt=ytt,
    )

    # CNN
    cnn_pth = "./saves/resnet18_lr0.01_decay0.001_bsz64_time1698682893.8338432.pt"
    cnn_model = load_cnn(cnn_pth)
    cnn_evl = EvaluateCNN(cnn_model, device, ytr, ytt, train_set, test_set, subsample=True)
    # cnn_evl.eval_internal("testing")
    cnn_Ftr = cnn_evl.original_attrs["training_features"]
    cnn_Ftt = cnn_evl.original_attrs["testing_features"]
    assert cnn_Ftr.shape[0] == ytr.shape[0]
    assert cnn_Ftt.shape[0] == ytt.shape[0]
    np.savez(os.path.join(cfg["data_dir"], "../numpy_files/cnn_features"),
        cnn_Ftr=cnn_Ftr,
        cnn_Ftt=cnn_Ftt,
        cnn_pth=cnn_pth,
    )
    print(f"Saved CNN features for model: {os.path.basename(cnn_pth)}")

    # CTR
    ctr_pth = "./saves/SupCon_resnet18_lr0.05_decay0.0001_bsz256_temp0.07_time1698645310.917615.pt"
    ctr_model = load_ctr(ctr_pth)
    ctr_evl = EvaluateCTR(ctr_model, device, ytr, ytt, train_set, test_set, subsample=True)
    ctr_evl.eval_internal("ctr testing")
    ctr_Ftr = ctr_evl.original_attrs["training_features"]
    ctr_Ftt = ctr_evl.original_attrs["testing_features"]
    assert ctr_Ftr.shape[0] == ytr.shape[0]
    assert ctr_Ftt.shape[0] == ytt.shape[0]
    np.savez(os.path.join(cfg["data_dir"], "../numpy_files/ctr_features"),
        ctr_Ftr=ctr_Ftr,
        ctr_Ftt=ctr_Ftt,
        model_pth=ctr_pth,
    )
    print(f"Saved CTR features for model: {os.path.basename(ctr_pth)}")

if __name__ == "__main__":
    main()