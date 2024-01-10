"""
Training script
"""
import argparse
import os
import json
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from datasets import load_data
from feature_methods import load_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)

def parse_options():
    parser = argparse.ArgumentParser('command-line arguments for model training')

    # general arguments
    parser.add_argument("--use-gpus", default='all', type=str, help='gpu device numbers')
    parser.add_argument('--model_path', type=str, default='saves')

    # method arguments
    parser.add_argument('--method', type=str, \
                        choices=["conv-autoencoder", "supervised-cnn", "supervised-ctr"])
    parser.add_argument('--c_hid', type=int, default=16, help='number of hidden channels')
    parser.add_argument('--latent_dim', type=int, default=100, help='number of latent dims')
    parser.add_argument('--pretrained', type=bool, default=False, help='pretrained weights (for supervised CNN)')
    parser.add_argument('--base_model', type=str,\
                        choices=["resnet18"], default = "resnet18", help="base CNN architecture (for CNN-based methods)")
    parser.add_argument('--projection', type=str, default='mlp',
                        choices=['linear', 'mlp'], help='projection head for CLR')
    parser.add_argument('--temp', type=float, default=0.07, help="temperature for CLR loss fxn")


    # data arguments
    parser.add_argument('--dataset', choices=["MedMNIST-AbdominalCT", "kaggle-pediatric-cxr"], default="MedMNIST-AbdominalCT")
    parser.add_argument('--dataset_transforms', type=str, default='default')
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                            help='which dataset is in-distribution')

    # training loop arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--print_mode', type=bool, default=True)
    
    # extract
    opt = parser.parse_args()

    # set GPU visibility
    if opt.use_gpus != 'all':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.use_gpus

    # get in-distribution CT view
    views = {
        "A": "Axial",
        "C": "Coronal",
        "S": "Sagittal"
    }
    id_view = views[opt.positive_dataset.replace("organ", "")[0].capitalize()]

    # set dataset transforms
    if opt.dataset_transforms == "default":
        dataset_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])
        ])
    else:
        raise NotImplementedError(f"requested transform is not implemented: {opt.dataset_transforms}")

    # storage files
    if not os.path.exists(opt.model_path):
        os.mkdir(opt.model_path)

    if not os.path.exists(cfg['data_dir']):
        os.mkdir(cfg['data_dir'])

    opt.model_name = '{}_lr{}_bsz{}_nep{}_indist{}_time{}'.\
        format(opt.method, opt.learning_rate, opt.batch_size, opt.max_epochs, id_view, time.time())
    opt.save_path = os.path.join(opt.model_path, opt.model_name + ".pt")

    # make options dictionary
    options = {
        "data_dir": cfg["data_dir"],
        "method": opt.method,
        "c_hid": opt.c_hid,
        "latent_dim": opt.latent_dim,
        "pretrained": opt.pretrained,
        "base_model": opt.base_model,
        "projection": opt.projection,
        "temp": opt.temp,
        "dataset": opt.dataset,
        "dataset_transforms": dataset_transforms,
        "positive_dataset": opt.positive_dataset,
        "batch_size": opt.batch_size,
        "max_epochs": opt.max_epochs,
        "learning_rate": opt.learning_rate,
        "print_mode": opt.print_mode,
        "save_path": opt.save_path,
        "peds_dataset": cfg['peds_dataset'],
        "adult_dataset": cfg['adult_dataset']
    }

    return options

def main():
    options = parse_options()

    train_set, val_set, test_set = load_data(options)

    train_loader = DataLoader(train_set, batch_size=options["batch_size"], shuffle=True)
    val_loader = DataLoader(val_set, batch_size=options["batch_size"], shuffle=True)

    feature_model = load_model(options)
    feature_model.train_model(train_loader, val_loader)

    # done
    print(f'==> Model is saved at: {options["save_path"]} !')

if __name__ == "__main__":
    main()
