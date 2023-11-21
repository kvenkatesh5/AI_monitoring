"""
Implementations of base abstract `Model` class.
"""
from abc import ABC
from abc import abstractmethod

import numpy as np
import torch
from torch.utils.data import DataLoader

class Model(ABC):
    def __init__(self, options: dict):
        self.options = options
        # set device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # set model
        self.model = self.init_model()
        self.model = self.model.to(self.device)
        # loss function
        self.loss_function = self.set_loss_function()
        # optimizer
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.options["learning_rate"])
        # lr scheduler
        milestones = [0.5 * self.options["max_epochs"], 0.75 * self.options["max_epochs"]]
        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=0.1)
        # tracking variables
        self.best_loss = np.inf
        self.best_epoch_number = -1
        self.current_epoch_number = 0
        self.train_loss_history = []
        self.val_loss_history = []

    @abstractmethod
    def init_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def set_loss_function(self):
        pass

    def load_state(self, model_path: str):
        print("==> Loading!")
        d = torch.load(model_path)
        # load model
        self.model.load_state_dict(d["model"])
        # set device
        self.device = d["device"]
        self.model = self.model.to(self.device)
        # load optimizer
        self.optimizer.load_state_dict(d["optimizer"])
        # load last epoch & loss
        self.current_epoch_number = d["epoch"]
        self.best_epoch_number = d["epoch"]
        self.best_loss = d["val_loss"]
        # load options
        self.options = d["options"]

    @abstractmethod
    def train_one_epoch(self, train_loader: DataLoader, val_loader: DataLoader):
        pass

    def train_model(self, train_loader: DataLoader, val_loader: DataLoader):
        for epoch in range(1, self.options["max_epochs"]+1):
            self.train_one_epoch(train_loader, val_loader)

    # source: https://github.com/HobbitLong/SupContrast/blob/master/util.py
    def save_model(self, model_copy, options, optimizer, epoch_number, epoch_val_loss, save_path):
        print("==> Saving!")
        state = {
            "model": model_copy.state_dict(),
            "device": self.device,
            "options": options,
            "optimizer": optimizer.state_dict(),
            "epoch": epoch_number,
            "val_loss": epoch_val_loss
        }
        torch.save(state, save_path)
        del state
    
    @staticmethod
    @abstractmethod
    def _key(self):
        pass