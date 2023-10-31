"""Contrastive learning for MedMNIST Abnominal CT images"""
import argparse
import os
import time
import json
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

from medmnist_datasets import AbnominalCTDataset
from networks.resnet_big import SupConResNet
from utils import SupConLoss, TwoCropTransform
from medmnist_datasets import AbnominalCTDataset
from utils import save_model
from medmnist_datasets import load_default_data

with open("cfg.json", "r") as f:
    cfg = json.load(f)

def parse_options():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.05,
                        help='learning rate')
    parser.add_argument('--positive_dataset', type=str, default='organamnist',
                        help='which dataset is in-distribution')
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument('--model', type=str, default='resnet18')
    parser.add_argument('--projection', type=str, default='mlp',
                        choices=['linear', 'mlp'], help='choose projection head for CLR training')
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--pretrained', action='store_true',
                        help='using ImageNet-pretrained weights')
    parser.add_argument("--use-gpus", default='all', type=str, help='')

    # extract
    opt = parser.parse_args()

    # storage files
    opt.model_path = './saves'
    opt.model_name = '{}_{}_lr{}_decay{}_bsz{}_temp{}_time{}.pt'.\
        format(opt.method, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp, time.time())

    opt.save_path = os.path.join(opt.model_path, opt.model_name)
    
    # data transform
    if opt.data_transform != "default":
        raise NotImplementedError("data transform not implemented: {}".format(opt.data_transform))
    opt.data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

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
        "method": opt.method,
        "model": opt.model,
        "optimizer_family": opt.optimizer_family,
        "scheduler_family": opt.scheduler_family,
        "max_epochs": opt.max_epochs,
        "initial_lr": opt.learning_rate,
        "weight_decay": opt.weight_decay,
        "momentum": opt.momentum,
        "loss_threshold": opt.loss_threshold,
        "plateau_patience": opt.plateau_patience,
        "break_patience": opt.break_patience,
        "drop_factor": opt.drop_factor,
        "temp": opt.temp,
        "pretrained": opt.pretrained,
        "projection": opt.projection,
        # Dataset
        "batch_size": opt.batch_size,
        "positive_dataset": opt.positive_dataset,
        "data_transform": opt.data_transform,
    }

    return options

"""Set model architecture"""
def set_model(opt):
    if opt["model"] != "resnet18":
        raise NotImplementedError("model is not implemented: {}".format(opt["model"]))
    # create model
    model = SupConResNet(opt["model"], opt["projection"])
    # set loss fxn
    criterion = SupConLoss(temperature=opt["temp"])
    # GPU support
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    
    return model, criterion

def save_model(model, optimizer, opt, epoch, save_path):
    print('==> Saving...')
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_path)
    del state

def train_one_epoch(tr_ldr, vl_ldr, model, criterion, optimizer, epoch, opt):
    # training
    model.train()
    train_loss = 0.0
    for j, (images, labels) in enumerate(tqdm(tr_ldr)):
        # stack both images
        images = torch.cat([images[0], images[1]], dim=0)
        # make grayscale into 3 channel
        images = torch.cat([images, images, images], dim=1)
        # GPU
        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        # compute loss
        features = model(images)
        bsz = labels.shape[0]
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt["method"] == 'SupCon':
            batch_loss = criterion(features, labels)
        elif opt["method"] == 'SimCLR':
            batch_loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt["method"]))
        # optimizer
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
        # update
        train_loss += batch_loss.item() * bsz
    train_loss = train_loss / len(tr_ldr)
    # validation
    model.eval()
    val_loss = 0.0
    for j, (images, labels) in enumerate(tqdm(vl_ldr)):
        with torch.no_grad():
            # stack both images
            images = torch.cat([images[0], images[1]], dim=0)
            # make grayscale into 3 channel
            images = torch.cat([images, images, images], dim=1)
            # GPU
            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            # compute loss
            features = model(images)
            bsz = labels.shape[0]
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            if opt["method"] == 'SupCon':
                batch_loss = criterion(features, labels)
            elif opt["method"] == 'SimCLR':
                batch_loss = criterion(features)
            else:
                raise ValueError('contrastive method not supported: {}'.
                                format(opt["method"]))
            # update
            val_loss += batch_loss.item() * bsz
    val_loss = val_loss / len(vl_ldr)

    return model, optimizer, train_loss, val_loss 


def train_contrastive(model: torch.nn.Module, criterion: torch.nn.Module, options: dict):
    # make contrastive datasets (two crop transform)
    cnn_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28),
        transforms.RandomRotation(10),
        options["data_transform"]
    ])
    ctr_train_set = AbnominalCTDataset(data_dir=options["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=options["positive_dataset"],
                        tfms=TwoCropTransform(cnn_transform), split="train")
    ctr_train_loader = DataLoader(ctr_train_set, batch_size=options["batch_size"], shuffle=True)
    ctr_val_set = AbnominalCTDataset(data_dir=options["data_dir"], label_mode="cheap-supervised",
                        positive_dataset=options["positive_dataset"],
                        tfms=TwoCropTransform(cnn_transform), split="val")
    ctr_val_loader = DataLoader(ctr_val_set, batch_size=options["batch_size"], shuffle=True)

    # tracking variables
    best_val_loss = np.inf
    best_epoch = -1

    # optimizer selection
    if options['optimizer_family'] == 'SGD':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=options['initial_lr'],
            momentum=options['momentum'],
            weight_decay=options['weight_decay']
        )
    elif options['optimizer_family'] == 'AdamW':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=options['initial_lr'],
            # weight_decay=options['weight_decay']
        )
    else:
        raise NotImplementedError('optimizer family not supported: {}'.format(
            options["optimizer_family"]
        ))

    # lr scheduler
    gamma = 0.1
    milestones = [0.5 * options["max_epochs"], 0.75 * options["max_epochs"]]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # training loop
    for epoch in range(1, options["max_epochs"]+1):
        # training
        time1 = time.time()
        model, optimizer, train_loss, val_loss = train_one_epoch(ctr_train_loader, ctr_val_loader,\
                                               model, criterion, optimizer, epoch, options)
        print('Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f} | Total Time {:.2f}'.\
              format(epoch, train_loss, val_loss, time.time() - time1))

        lr_scheduler.step()
        # saving best model by val loss
        if val_loss < best_val_loss:
            # update trackers
            best_val_loss = val_loss
            best_epoch = epoch
            # save
            save_model(deepcopy(model), optimizer, options, epoch, options["save_path"])
    
    print("Model saved at: {}".format(options["save_path"]))  
    return model

def main():
    # extract options
    opt = parse_options()
    print(opt)

    # model
    model, loss_fxn = set_model(opt)

    # contrastive training
    trained_model = train_contrastive(model, loss_fxn, opt)

    # extract features
    train_set, val_set, test_set = load_default_data(opt)

if __name__ == "__main__":
    main()
