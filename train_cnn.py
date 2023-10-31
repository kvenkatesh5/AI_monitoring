"""
Train a ResNet on supervised OOD detection
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
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score

from medmnist_datasets import AbnominalCTDataset
from utils import save_model

with open("cfg.json", "r") as f:
    cfg = json.load(f)

"""Train CNN"""
def train_cnn(tr_set, vl_set, hparams):
    # loaders
    train_loader = DataLoader(tr_set, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(vl_set, batch_size=hparams["batch_size"], shuffle=True)
    # Model
    if hparams["pretrained"]:
        model = resnet18(weights=ResNet18_Weights.DEFAULT)
    else:
        model = resnet18()
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(in_features=512, out_features=2),
        torch.nn.Sigmoid()
    )
    model = model.to(hparams["device"])
    # loss function
    classification_loss_fxn = torch.nn.BCELoss()
    # optimizer
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=hparams["initial_lr"])
    # lr scheduler
    gamma = 0.1
    milestones = [0.5 * hparams["max_epochs"], 0.75 * hparams["max_epochs"]]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    # loop
    train_loss_history = []
    val_loss_history = []
    best_auroc = 0
    best_epoch = -1
    for epoch_number in range(1, hparams["max_epochs"] + 1):
        # training
        model.train()
        total_train_loss = []
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # clear gradients
            optimizer.zero_grad()
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
                # make labels into one-hot
                targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
                targets = targets.to(hparams["device"])
                # model output
                outputs = model(images)
                # compute loss
                batch_loss = classification_loss_fxn(outputs, targets)
                total_val_loss.append(batch_loss.item())
                # compute AUROC
                val_ys.extend(targets.detach().cpu().argmax(dim=1).tolist())
                val_yhats.extend(outputs[:,1].detach().squeeze().cpu().tolist())
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)
        val_loss_history.append(epoch_val_loss)
        epoch_val_auroc = roc_auc_score(val_ys, val_yhats)
        val_ys = np.array(val_ys)
        # If yhat > 0.5, then prediction is assigned as 1 (because outputs[:,1] were chosen as yhat)
        val_preds = np.array(val_yhats) > 0.5
        epoch_val_acc = np.sum(val_ys == val_preds) / val_ys.shape[0]
        print("Epoch {} | Training Loss {:.4f} | Validation Loss {:.4f} | Validation AUROC {:.4f} | Validation Acc {:.4f}"\
              .format(epoch_number, epoch_train_loss,\
                      epoch_val_loss, epoch_val_auroc, epoch_val_acc))
        lr_scheduler.step()
        # saving best model by AUROC
        if epoch_val_auroc > best_auroc:
            best_auroc = epoch_val_auroc
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
                        help='which dataset is in-distribution (assigned y=1)')
    parser.add_argument('--pretrained', type=str, default='y',\
                        help='is the resnet pretrained on ImageNet')
    parser.add_argument("--use-gpus", default='all', type=str, help='GPU device numbers')
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
    opt.model_name = 'resnet18_lr{}_bsz{}_nep{}_indist{}_time{}'.\
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
        "pretrained": opt.pretrained == 'y',
        "max_epochs": opt.max_epochs,
        "initial_lr": opt.learning_rate,
        "batch_size": opt.batch_size,
        "positive_dataset": opt.positive_dataset,
    }

    return options

def main():
    options = parse_options()

    """CNN training with standard normalization"""
    cnn_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    """in-distribution CT view"""
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
