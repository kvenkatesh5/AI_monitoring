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
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import resnet18
from sklearn.metrics import roc_auc_score

from medmnist_datasets import AbnominalCTDataset
from medmnist_datasets import matrixify
from utils import save_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)

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
    model = model.to(hparams["device"])
    # loss and optim
    classification_loss_fxn = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams["initial_lr"],
                                  weight_decay=hparams["weight_decay"])
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
        train_loss = train_loss / len(train_loader) # trying here with train loader
        train_loss_history.append(train_loss)
        # validation
        model.eval()
        val_loss = 0.0
        tot_images = 0
        val_ys = []
        val_yhats = []
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
                # compute AUROC
                val_ys.extend(targets.detach().cpu().argmax(dim=1).tolist())
                val_yhats.extend(outputs[:,1].detach().squeeze().cpu().tolist())
        val_loss = val_loss / len(val_loader) # trying here with val loader
        val_loss_history.append(val_loss)
        val_auroc = roc_auc_score(val_ys, val_yhats)
        print("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f} | Validation AUROC {:.4f}".format(e, train_loss, val_loss, val_auroc))
        
        # saving best model
        if best_loss - val_loss > hparams["loss_threshold"]:
            stopping_step = 0
            best_loss = val_loss
            save_model(model, optimizer, e, hparams, hparams["save_path"])
        else:
            stopping_step += 1
        # early breaking
        if stopping_step >= hparams["break_patience"]:
            print("...Early breaking!")
            break
    print("Model saved at: {}".format(hparams["save_path"]))
    return model, train_loss_history, val_loss_history

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--max_epochs', type=int, default=1000,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--scheduler_family', type=str, default="drop",
                        choices=["drop", "step", "no-scheduler"], help='kind of lr scheduler')
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
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
    print(f"Device: {device} | # {dev_n}")

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
        "scheduler_family": opt.scheduler_family,
        "drop_factor": opt.drop_factor,
        "plateau_patience": opt.plateau_patience,
        "weight_decay": opt.weight_decay,
        "loss_threshold": opt.loss_threshold,
        "break_patience": opt.break_patience,
        # Dataset
        "batch_size": opt.batch_size,
        "positive_dataset": opt.positive_dataset,
        "shuffle": opt.shuffle == "y",
    }

    return options

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

if __name__ == "__main__":
    main()
