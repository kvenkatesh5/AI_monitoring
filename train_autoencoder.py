"""
Train a convolutional autoencoder
"""
import argparse
import os
import json
import time
from copy import deepcopy

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist_datasets import AbnominalCTDataset
from networks.conv_autoencoder import AutoEncoder
from utils import save_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)
    
"""Train model"""
def train_autoencoder(tr_set, vl_set, hparams):
    # loaders
    train_loader = DataLoader(tr_set, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(vl_set, batch_size=hparams["batch_size"], shuffle=True)
    # model
    model = AutoEncoder(c_hid=hparams["c_hid"], latent_dim=hparams["latent_dim"])
    model = model.to(hparams["device"])
    # loss and optim
    loss_fxn = torch.nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["initial_lr"])
    # lr scheduler
    gamma = 0.1
    milestones = [0.5 * hparams["max_epochs"], 0.75 * hparams["max_epochs"]]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # loop
    train_loss_history = []
    val_loss_history = []
    best_loss = np.inf
    best_epoch = -1
    for epoch_number in range(1, hparams["max_epochs"]+1):
        # training
        model.train()
        total_train_loss = []
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # clear gradients
            optimizer.zero_grad()
            # make grayscale into 3 channel
            images = torch.cat([images, images, images], dim=1)
            images = images.to(hparams["device"])
            # model output
            outputs = model(images)
            # compute loss
            batch_loss = loss_fxn(images, outputs)
            # optimizer step
            batch_loss.backward()
            optimizer.step()
            # training loss
            total_train_loss.append(batch_loss.item())
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)
        train_loss_history.append(epoch_train_loss)
        # validation
        model.eval()
        total_val_loss = []
        val_ys = []
        val_yhats = []
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # make grayscale into 3 channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(hparams["device"])
                # model output
                outputs = model(images)
                # compute loss
                batch_loss = loss_fxn(images, outputs)
                total_val_loss.append(batch_loss.item())
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)
        val_loss_history.append(epoch_val_loss)
        print("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f}"\
              .format(epoch_number, epoch_train_loss,\
                      epoch_val_loss))
        # lr scheduler
        lr_scheduler.step()
        # saving best model by loss
        if epoch_val_loss < best_loss:
            best_loss = epoch_val_loss
            best_epoch = epoch_number
            save_model(deepcopy(model), optimizer, best_epoch, hparams, hparams["save_path"])
    print("Model saved at: {}".format(hparams["save_path"]))
    return model, train_loss_history, val_loss_history

def parse_options():
    parser = argparse.ArgumentParser('command-line arguments for model training')

    # arguments
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.001)
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                        help='which dataset is in-distribution')
    parser.add_argument('--c_hid', type=int, default=16, help='number of hidden channels')
    parser.add_argument('--latent_dim', type=int, default=100, help='number of latent dims')
    parser.add_argument("--use-gpus", default='all', type=str, help='gpu device numbers')

    # extract
    opt = parser.parse_args()

    # get in-distribution CT view
    views = {
        "A": "Axial",
        "C": "Coronal",
        "S": "Sagittal"
    }
    id_view = views[opt.positive_dataset.replace("organ", "")[0].capitalize()]

    # storage files
    opt.model_path = './saves'
    opt.model_name = 'autoencoder_lr{}_bsz{}_nep{}_indist{}_time{}'.\
        format(opt.learning_rate, opt.batch_size, opt.max_epochs, id_view, time.time())
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
        "save_path": opt.save_path,
        "device": device,
        "c_hid": opt.c_hid,
        "latent_dim": opt.latent_dim,
        "max_epochs": opt.max_epochs,
        "initial_lr": opt.learning_rate,
        "batch_size": opt.batch_size,
        "positive_dataset": opt.positive_dataset,
    }

    return options

"""Main fxn"""
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

    """train model"""
    model, tr_h, val_h = train_autoencoder(train_set, val_set, options)

if __name__ == "__main__":
    main()
