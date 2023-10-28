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
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms, datasets
from torchvision.models import resnet18
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from scipy.spatial.distance import cdist
from medmnist_datasets import AbnominalCTDataset
from utils import SupConLoss
from networks.resnet_big import SupConResNet

"""
Datasets

Source: https://medmnist.com/

Task: Distinguish axial views from sagittal/coronal views on abdominal CT scans from the Organ{A/C/S} Medical MNIST dataset. (The axial view was chosen as "OOD" since it was the most difficult for a logistic regression classifier to distinguish.)

Dataset Transforms:  ToTensor, Normalization (mu=0.5, std=0.5)

Training Transforms: RandomResized Crop, Rotations (+- 10 degrees), ToTensor, Normalization (mu=0.5, std=0.5)
"""

"""Matrixify fxn"""
def matrixify(dset, label_mode="cheap-supervised"):
    if label_mode!="cheap-supervised":
        raise NotImplementedError("label mode not implemented: {}".format(label_mode))
    X = np.empty((len(dset), 28*28))
    y = np.empty((len(dset)))
    for i in tqdm(range(len(dset))):
        image, label = dset[i]
        X[i,:] = image.flatten()
        y[i] = int(label)
    y = y.astype(np.int)
    return X,y

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

        plt.show()

