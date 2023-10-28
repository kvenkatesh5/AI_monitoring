"""
Train a CNN on in-modality OOD detection
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
from torchvision.models import resnet18

from medmnist_datasets import AbnominalCTDataset
from features import EvaluateFeatureSpace
from features import matrixify
from utils import save_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)

"""Forward hook for ResNet"""
cnn_layers = {}
def get_inputs(name):
    def hook(model, inpt, output):
        cnn_layers[name] = inpt[0].detach()
    return hook

"""Load model"""
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

"""Train CNN"""
def train_cnn(tr_set, vl_set, hparams):
    # loaders
    train_loader = DataLoader(tr_set, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    val_loader = DataLoader(vl_set, batch_size=hparams["batch_size"], shuffle=hparams["shuffle"])
    # Init a model with forward hook for extracting features
    model = resnet18(pretrained=hparams["pretrained"])
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=2),
        torch.nn.Sigmoid()
    )
    h = model.fc.register_forward_hook(get_inputs('fts'))
    model = model.to(hparams["device"])   
    # loss and optim
    classification_loss_fxn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["initial_lr"], weight_decay=hparams["weight_decay"])
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
            # make labels into one-hot
            targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
            targets = targets.to(hparams["device"])
            # model output
            outputs = model(images)
            # compute loss
            batch_loss = classification_loss_fxn(outputs, targets)
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
                # make labels into one-hot
                targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
                targets = targets.to(hparams["device"])
                # model output
                outputs = model(images)
                # compute loss
                batch_loss = classification_loss_fxn(outputs, targets)
                val_loss += batch_loss.item() * n_images
        val_loss = val_loss / tot_images
        val_loss_history.append(val_loss)
        print("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f}".format(e, train_loss, val_loss))
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
    return model, h, train_loss_history, val_loss_history

"""Evaluate CNN"""
class EvaluateCNN(EvaluateFeatureSpace):
    def get_features(self, dset):
        self.model.eval()
        features = []
        ldr = DataLoader(dset, batch_size=32)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                features.append(cnn_layers['fts'].detach().cpu().numpy())
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
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--loss-threshold', type=float, default=1e-4,
                        help='min change in loss to update best model')
    parser.add_argument('--break_patience', type=int, default=5,
                        help='early breaking patience')

    # dataset
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--shuffle', type=str, default='y')
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                        help='which dataset is positive')

    # method
    parser.add_argument('--pretrained', type=str, default='y', help='is the resnet pretrained on ImageNet')

    # other setting
    parser.add_argument("--use-gpus", default='all', type=str, help='')

    # extract
    opt = parser.parse_args()

    # storage files
    opt.model_path = './saves'
    opt.model_name = 'resnet18_lr{}_decay{}_bsz{}_time{}'.\
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
        "pretrained": opt.pretrained == 'y',
        "max_epochs": opt.max_epochs,
        "initial_lr": opt.learning_rate,
        "weight_decay": opt.weight_decay,
        "loss_threshold": opt.loss_threshold,
        "break_patience": opt.break_patience,
        # Dataset
        "batch_size": opt.batch_size,
        "num_workers": opt.num_workers,
        "positive_dataset": opt.positive_dataset,
        "shuffle": opt.shuffle == "y",
    }

    return options

"""Extract features from trained CNN"""
def extract_features(model, hparams, train_set, test_set):
    # Matrixify datasets
    Xtr, ytr = matrixify(train_set)
    Xtt, ytt = matrixify(test_set)
    # Evaluation object
    evl = EvaluateCNN(model, hparams["device"], ytr, ytt, train_set, test_set, subsample=True)
    # Get features
    cnn_Ftr = evl.original_attrs["training_features"]
    cnn_Ftt = evl.original_attrs["testing_features"]
    np.savez(os.path.join(cfg["data_dir"], "../numpy_files/cnn_features"),
        cnn_Ftr=cnn_Ftr,
        cnn_Ftt=cnn_Ftt,
    )
    # return features
    return cnn_Ftr, cnn_Ftt


def main():
    options = parse_options()

    """CNN training with data augmentations"""
    cnn_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    positive_dataset="organamnist"

    """training dataset"""
    train_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=cnn_transform,
                        split="train")

    """val dataset"""
    val_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=cnn_transform,
                        split="val")
    """testing dataset"""
    test_set = AbnominalCTDataset(data_dir=cfg["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=positive_dataset, tfms=cnn_transform,
                        split="test")

    """train model"""
    model, tr_h, val_h = train_cnn(train_set, val_set, options)

    """extract (and save) features"""
    cnn_Ftr, cnn_Ftt = extract_features(model, options, train_set, test_set)

if __name__ == "__main__":
    main()