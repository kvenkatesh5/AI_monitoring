"""
Package `__init__`: dataset loading function
"""
import numpy as np
from tqdm import tqdm

from .medmnist_abdominalCT import AbnominalCTDataset
from .pediatricCXR import PediatricCXRDataset
from feature_methods.supcon_loss import TwoCropTransform

# Master function for loading datasets
def load_data(options: dict):
    if options["dataset"] == "MedMNIST-AbdominalCT":
        train_tfms = AbnominalCTDataset.get_default_transform()
        val_tfms = AbnominalCTDataset.get_default_transform()
        test_tfms = AbnominalCTDataset.get_default_transform()
        if options["method"] == "supervised-ctr":
            train_tfms = TwoCropTransform(train_tfms)
            val_tfms = TwoCropTransform(val_tfms)
        # train/val/test datasets
        train_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=train_tfms,
            split="train"
        )
        val_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=val_tfms,
            split="val"
        )
        test_set = AbnominalCTDataset(
            data_dir=options["data_dir"],
            label_mode="cheap-supervised",
            positive_dataset=options["positive_dataset"],
            tfms=test_tfms,
            split="test"
        )
        return train_set, val_set, test_set
    
    if options["dataset"] == "kaggle-pediatric-cxr":
        train_tfms = PediatricCXRDataset.get_default_transform()
        val_tfms = PediatricCXRDataset.get_default_transform()
        test_tfms = PediatricCXRDataset.get_default_transform()
        if options["method"] == "supervised-ctr":
            train_tfms = TwoCropTransform(train_tfms)
            val_tfms = TwoCropTransform(val_tfms)
        # train/val/test datasets
        train_set =  PediatricCXRDataset(
            data_dir='data',
            peds_data_dir=options["peds_dataset"],
            adult_data_dir=options["adult_dataset"],
            tfms=train_tfms,
            split="train"
        )
        val_set = PediatricCXRDataset(
            data_dir='data',
            peds_data_dir=options["peds_dataset"],
            adult_data_dir=options["adult_dataset"],
            tfms=val_tfms,
            split="val"
        )
        test_set = PediatricCXRDataset(
            data_dir='data',
            peds_data_dir=options["peds_dataset"],
            adult_data_dir=options["adult_dataset"],
            tfms=test_tfms,
            split="test"
        )
        return train_set, val_set, test_set

    else:
        raise NotImplementedError(f"requested dataset is not available: {options['dataset']}")

# reformat dataset into X,y matrix/vector pair
def matrixify(dset, label_mode="cheap-supervised", sz=(28, 28)):
    if label_mode!="cheap-supervised":
        raise NotImplementedError("label mode not implemented: {}".format(label_mode))
    X = np.empty((len(dset), sz[0]*sz[1]))
    y = np.empty((len(dset)))
    for i in tqdm(range(len(dset))):
        image, label = dset[i]
        X[i,:] = image.flatten()
        y[i] = int(label)
    y = y.astype(np.int32)
    return X,y
