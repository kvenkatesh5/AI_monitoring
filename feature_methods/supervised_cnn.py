"""
Implementation of supervised CNN feature extraction.
"""
from copy import deepcopy

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.models import resnet18
from torchvision.models import ResNet18_Weights
from sklearn.metrics import roc_auc_score

from .base import Model

class SupervisedCNN(Model):
    def init_model(self):
        model = resnet18(
            weights=(ResNet18_Weights.DEFAULT if self.options["pretrained"] else None)
        )
        model.fc = torch.nn.Sequential(
            torch.nn.Linear(in_features=512, out_features=2),
            torch.nn.Sigmoid()
        )
        model = model.to(self.device)
        return model
    
    def set_loss_function(self):
        return torch.nn.BCELoss()
    
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        # update epoch number
        self.current_epoch_number += 1
        # training phase
        self.model.train()
        total_train_loss = []
        for i, (images, labels) in enumerate(tqdm(train_loader)):
            # clear gradients
            self.optimizer.zero_grad()
            # make grayscale into 3 channel
            images = torch.cat([images, images, images], dim=1)
            images = images.to(self.device)
            # make labels into one-hot
            targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
            targets = targets.to(self.device)
            # model output
            outputs = self.model(images)
            # compute loss
            batch_loss = self.loss_function(outputs, targets)
            # optimizer step
            batch_loss.backward()
            self.optimizer.step()
            # training loss
            total_train_loss.append(batch_loss.item())
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)
        # validation phase
        self.model.eval()
        total_val_loss = []
        val_ys = []
        val_yhats = []
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # make grayscale into 3 channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                # make labels into one-hot
                targets = torch.nn.functional.one_hot(labels, num_classes=2).float()
                targets = targets.to(self.device)
                # model output
                outputs = self.model(images)
                # compute loss
                batch_loss = self.loss_function(outputs, targets)
                total_val_loss.append(batch_loss.item())
                # compute AUROC
                val_ys.extend(targets.detach().cpu().argmax(dim=1).tolist())
                val_yhats.extend(outputs[:,1].detach().squeeze().cpu().tolist())
        # compute epoch val loss
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)
        # compute epoch val AUROC
        epoch_val_auroc = roc_auc_score(val_ys, val_yhats)
        # compute epoch val accuracy
        val_ys = np.array(val_ys)
        val_preds = np.array(val_yhats) > 0.5 # yhat > 0.5 --> 1
        epoch_val_acc = np.sum(val_ys == val_preds) / val_ys.shape[0]
        # update lr
        self.lr_scheduler.step()
        # update model
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            self.best_epoch_number = self.current_epoch_number
            self.save_model(
                model_copy=deepcopy(self.model),
                options=self.options,
                optimizer=self.optimizer,
                epoch_number=self.current_epoch_number,
                epoch_val_loss=epoch_val_loss,
                save_path=self.options["save_path"]
            )
        # save loss history
        self.train_loss_history.append(epoch_train_loss)
        self.val_loss_history.append(epoch_val_loss)
        # print
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " + 
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f} | " +
                f"Val AUROC: {epoch_val_auroc:.4f} | Val Acc: {epoch_val_acc:.4f}"
            )

    @staticmethod
    def _key():
        return "supervised-cnn"
