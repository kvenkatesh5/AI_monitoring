"""
Train a convolutional autoencoder
"""
import argparse
import os
import sys
import json
import time

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist_datasets import AbnominalCTDataset
from networks.conv_autoencoder import AutoEncoder
from features import EvaluateFeatureSpace
from medmnist_datasets import matrixify
from utils import save_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)

"""Load model"""
def load_autoencoder(pth):
    print("Loading! <= {}".format(os.path.basename(pth)))
    state = torch.load(pth)
    hparams = state["hparams"]
    model = AutoEncoder(c_hid=hparams["c_hid"], latent_dim=hparams["latent_dim"])
    model.load_state_dict(state["model"])
    model = model.to(hparams["device"])
    return model
    
"""Train model"""
def train_autoencoder(tr_set, vl_set, hparams):
    # loaders
    train_loader = DataLoader(tr_set, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    val_loader = DataLoader(vl_set, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    # model
    model = AutoEncoder(c_hid=hparams["c_hid"], latent_dim=hparams["latent_dim"])
    model = model.to(hparams["device"])
    # loss and optim
    loss_fxn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["initial_lr"], weight_decay=hparams["weight_decay"])
    # lr_scheduler
    if hparams['scheduler_family'] == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer=optimizer,
            step_size=hparams['plateau_patience'],
            gamma=hparams['drop_factor'],
        )
    elif hparams['scheduler_family'] == 'drop':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=optimizer,
            factor=hparams['drop_factor'],
            patience=hparams['plateau_patience'],
            verbose=False
        )
    elif hparams['scheduler_family'] == "no-scheduler":
        pass
    else:
        raise NotImplementedError('scheduler family not supported: {}'.format(
            hparams["scheduler_family"]
        ))
    # loop
    train_loss_history = []
    val_loss_history = []
    stopping_step = 0
    best_loss = np.inf
    for e in range(hparams["max_epochs"]):
        # training
        model.train()
        train_loss = 0.0
        tot_images = 0
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # number of images
            n_images = images.shape[0]
            tot_images += n_images
            # make grayscale into 3 channel
            images = torch.cat([images, images, images], dim=1)
            images = images.to(hparams["device"])
            # model output
            outputs = model(images)
            # compute loss
            batch_loss = loss_fxn(images, outputs)
            # optimizer step
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            # training loss
            train_loss += batch_loss.item() * n_images
        train_loss = train_loss / tot_images
        train_loss_history.append(train_loss)
        # validation
        model.eval()
        val_loss = 0.0
        tot_images = 0
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # number of images
                n_images = images.shape[0]
                tot_images += n_images
                # make grayscale into 3 channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(hparams["device"])
                # model output
                outputs = model(images)
                # compute loss
                batch_loss = loss_fxn(images, outputs)
                val_loss += batch_loss.item() * n_images
        val_loss = val_loss / tot_images
        val_loss_history.append(val_loss)
        print("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f}".format(e, train_loss, val_loss))
        # lr scheduler
        if hparams["scheduler_family"] == "drop":
            scheduler.step(val_loss)
        elif hparams["scheduler_family"] != "no-scheduler":
            scheduler.step()
        # early breaking
        if best_loss - val_loss > hparams["loss_threshold"]:
            stopping_step = 0
            best_loss = val_loss
            save_model(model, optimizer, e, hparams, hparams["save_path"])
        else:
            stopping_step += 1
        if stopping_step >= hparams["break_patience"]:
            print("...Early breaking!")
            break
    print("Model saved at: {}".format(hparams["save_path"]))
    return model, train_loss_history, val_loss_history

"""Evaluate autoencoder"""
class EvaluateAutoEncoder(EvaluateFeatureSpace):
    def get_features(self, dset: Dataset, return_loss=False):
        self.model.eval()
        loss_fxn = torch.nn.MSELoss()
        features = []
        loss = 0.0
        ldr = DataLoader(dset, batch_size=32)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                fts = self.model.get_features(images).detach().cpu().numpy()
                features.append(fts)
                outputs = self.model(images)
                batch_loss = loss_fxn(images, outputs)
                loss += batch_loss.item() * (images.shape[0])
        loss = loss / len(ldr)
        if return_loss:
            return np.vstack(features), loss
        return np.vstack(features)

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--scheduler_family', type=str, default="drop",
                        choices=["drop", "step", "no-scheduler"], help='kind of lr scheduler')
    parser.add_argument('--drop_factor', type=float, default=0.1,
                        help='drop factor in lr scheduler')
    parser.add_argument('--plateau_patience', type=int, default=3,
                        help='patience in lr scheduler')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--loss_threshold', type=float, default=1e-4,
                        help='min change in loss to update best model')
    parser.add_argument('--break_patience', type=int, default=5,
                        help='early breaking patience')

    # dataset
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--shuffle', type=str, default='y')
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                        help='which dataset is positive')

    # method
    parser.add_argument('--c_hid', type=int, default=16, help='number of hidden channels')
    parser.add_argument('--latent_dim', type=int, default=100, help='number of latent dims')

    # other setting
    parser.add_argument("--use-gpus", default='all', type=str, help='')

    # extract
    opt = parser.parse_args()

    # storage files
    opt.model_path = './saves'
    opt.model_name = 'autoencoder_lr{}_decay{}_bsz{}_time{}'.\
        format(opt.learning_rate, opt.weight_decay, opt.batch_size, time.time())

    opt.save_path = os.path.join(opt.model_path, opt.model_name + ".pt")

    # Set GPU vis
    if opt.use_gpus != 'all':
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = opt.use_gpus
    device = 'cpu'
    ncpus = os.cpu_count()
    dev_n = ncpus
    if torch.cuda.is_available():
        device = 'cuda'
        dev_n = torch.cuda.device_count()

    # Make options dictionary
    options = {
        # Storage
        "data_dir": opt.data_dir,
        "save_path": opt.save_path,
        # Training
        "device": device,
        "c_hid": opt.c_hid,
        "latent_dim": opt.latent_dim,
        "max_epochs": opt.max_epochs,
        "initial_lr": opt.learning_rate,
        "weight_decay": opt.weight_decay,
        "loss_threshold": opt.loss_threshold,
        "break_patience": opt.break_patience,
        "scheduler_family": opt.scheduler_family,
        "drop_factor": opt.drop_factor,
        "plateau_patience": opt.plateau_patience,
        # Dataset
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "positive_dataset": opt.positive_dataset,
        "shuffle": opt.shuffle == "y",
    }

    return options

"""Extract features from trained autoencoder"""
def extract_features(model, hparams, train_set, test_set):
    # Matrixify datasets
    Xtr, ytr = matrixify(train_set)
    Xtt, ytt = matrixify(test_set)
    # Evaluation object
    evl = EvaluateAutoEncoder(model, hparams["device"], ytr, ytt, train_set, test_set, subsample=True)
    # Get features
    Ftr = evl.original_attrs["training_features"]
    Ftt = evl.original_attrs["testing_features"]
    np.savez(os.path.join(cfg["data_dir"], "../numpy_files/autoencoder_features"),
        autoencoder_Ftr=Ftr,
        autoencoder_Ftt=Ftt,
    )
    # return features
    return Ftr, Ftt

def main():
    options = parse_options()

    # Baseline tr
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    positive_dataset="organamnist"

    """training dataset"""
    train_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=data_transform,
                        split="train")

    """val dataset"""
    val_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=data_transform,
                        split="val")
    """testing dataset"""
    test_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=data_transform,
                        split="test")

    """matrixify dataset"""
    Xtr, ytr = matrixify(train_set)
    Xvl, yvl = matrixify(val_set)
    Xtt, ytt = matrixify(test_set)

    """train model"""
    model, tr_h, val_h = train_autoencoder(train_set, val_set, options)

    """extract (and save) features"""
    Ftr, Ftt = extract_features(model, options, train_set, test_set)

if __name__ == "__main__":
    main()
