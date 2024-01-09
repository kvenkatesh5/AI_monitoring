"""
Calculate features for all of the methods.
"""
import argparse
import os
import json
from glob import glob

import numpy as np
import torch
from torchvision import transforms

from datasets import matrixify
from datasets import load_data
from feature_methods import load_model
from feature_methods import load_eval

"""Main fxn: extract each approach's features here"""
def main():
    # Get model paths
    parser = argparse.ArgumentParser('command-line arguments')
    parser.add_argument('--use_gpus', type=str, default="5,6")
    parser.add_argument('--dataset', choices=["MedMNIST-AbdominalCT"])
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                            help='which dataset is in-distribution')
    parser.add_argument('--autoencoder_path', type=str)
    parser.add_argument('--cnn_path', type=str)
    parser.add_argument('--ctr_path', type=str)
    parser.add_argument('--save_path', type=str, default='numpy_files/data_splits.npz')
    opt = parser.parse_args()

    # Cfg
    with open("cfg.json", "r") as f:
        cfg = json.load(f)

    # Set GPU vis
    use_gpus = opt.use_gpus
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

    # All features will be computed using this data load
    # see documentation of ```load_data```
    train_set, val_set, test_set = load_data({
        "dataset": opt.dataset,
        "data_dir": cfg["data_dir"],
        "positive_dataset": opt.positive_dataset,
        # this argument ensures that TwoCropTransform is not loaded
        "method": "none",
    })

    # Matrixify datasets
    Xtr, ytr = matrixify(train_set)
    Xvl, yvl = matrixify(val_set)
    Xtt, ytt = matrixify(test_set)
    data_splits_path = os.path.join(cfg["data_dir"], "../numpy_files/data_splits")
    np.savez(data_splits_path,
        Xtr=Xtr, ytr=ytr,
        Xvl=Xvl, yvl=yvl,
        Xtt=Xtt, ytt=ytt,
    )
    print(f"Saved dataset as matrices at: {data_splits_path} !")

    # Autoencoder
    autoencoder_load = torch.load(glob(opt.autoencoder_path)[0])
    autoencoder_model = load_model(autoencoder_load["options"], mode="testing")
    autoencoder_eval = load_eval(autoencoder_model, train_set, val_set, test_set)
    autoencoder_fts_path = os.path.join(cfg["data_dir"], "../numpy_files/autoencoder_features")
    np.savez(autoencoder_fts_path,
        autoencoder_Ftr=autoencoder_eval.train_features,
        autoencoder_Ftt=autoencoder_eval.test_features,
        autoencoder_pth=opt.autoencoder_path,
    )
    print(f"Saved autoencoder features at: {autoencoder_fts_path} !")

    # Supervised CNN
    cnn_load = torch.load(glob(opt.cnn_path)[0])
    cnn_model = load_model(cnn_load["options"], mode="testing")
    cnn_eval = load_eval(cnn_model, train_set, val_set, test_set)
    cnn_fts_path = os.path.join(cfg["data_dir"], "../numpy_files/cnn_features")
    np.savez(cnn_fts_path,
        cnn_Ftr=cnn_eval.train_features,
        cnn_Ftt=cnn_eval.test_features,
        cnn_pth=opt.cnn_path,
    )
    print(f"Saved ood-supervised CNN features at: {cnn_fts_path} !")

    # Supervised Contrastive CNN
    ctr_load = torch.load(glob(opt.ctr_path)[0])
    ctr_model = load_model(ctr_load["options"], mode="testing")
    ctr_eval = load_eval(ctr_model, train_set, val_set, test_set)
    ctr_fts_path = os.path.join(cfg["data_dir"], "../numpy_files/ctr_features")
    np.savez(ctr_fts_path,
        ctr_Ftr=ctr_eval.train_features,
        ctr_Ftt=ctr_eval.test_features,
        ctr_pth=opt.ctr_path,
    )
    print(f"Saved ood-supervised CTR features at: {ctr_fts_path} !")


if __name__ == "__main__":
    main()
