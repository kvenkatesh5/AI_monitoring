"""
Package `__init__`: dataset loading function
"""
import numpy as np
from tqdm import tqdm

from .medmnist_abdominalCT import AbnominalCTDataset
from feature_methods.supcon_loss import TwoCropTransform

def load_data(options: dict):
    if options["dataset"] == "MedMNIST-AbdominalCT":
        if options["method"] == "supervised-ctr":
            # training dataset (apply TwoCropTransform)
            train_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=TwoCropTransform(options["dataset_transforms"]),
                split="train"
            )
            # validation dataset (apply TwoCropTransform)
            val_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=TwoCropTransform(options["dataset_transforms"]),
                split="val"
            )
            # testing dataset
            test_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=options["dataset_transforms"],
                split="test"
            )
        else:
            # training dataset
            train_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=options["dataset_transforms"],
                split="train"
            )
            # validation dataset
            val_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=options["dataset_transforms"],
                split="val"
            )
            # testing dataset
            test_set = AbnominalCTDataset(
                data_dir=options["data_dir"],
                label_mode="cheap-supervised",
                positive_dataset=options["positive_dataset"],
                tfms=options["dataset_transforms"],
                split="test"
            )
    else:
        raise NotImplementedError(f'requested dataset not available: {options["dataset"]}')
    
    return train_set, val_set, test_set


# reformat dataset into X,y matrix/vector pair
def matrixify(dset, label_mode="cheap-supervised"):
    if label_mode!="cheap-supervised":
        raise NotImplementedError("label mode not implemented: {}".format(label_mode))
    X = np.empty((len(dset), 28*28))
    y = np.empty((len(dset)))
    for i in tqdm(range(len(dset))):
        image, label = dset[i]
        X[i,:] = image.flatten()
        y[i] = int(label)
    y = y.astype(np.int32)
    return X,y
