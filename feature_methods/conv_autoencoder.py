"""
Implementation of convolutional autoencoder architecture + model class
Source: https://lightning.ai/docs/pytorch/stable/notebooks/course_UvA-DL/08-deep-autoencoders.html
"""
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from .base import Model
from .base import FeatureSpace

class Encoder(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(Encoder, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * 16 * c_hid, latent_dim),
        )
    def forward(self, x):
        return self.network(x)
    
class Decoder(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(Decoder, self).__init__()
        self.linear = nn.Sequential(nn.Linear(latent_dim, 2 * 16 * c_hid), nn.ReLU())
        self.network = nn.Sequential(
            nn.ConvTranspose2d(
                2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2
            ),
            nn.ReLU(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3, output_padding=1, padding=1, stride=2),
            nn.ReLU(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(
                c_hid, 3, kernel_size=3, output_padding=1, padding=1, stride=2
            ),
            nn.Tanh(),
        )
    def forward(self, x):
        z = self.linear(x)
        z = z.reshape(z.shape[0], -1, 4, 4)
        return self.network(z)

class ConvAutoEncoderCore(nn.Module):
    def __init__(self, c_hid=16, latent_dim=100):
        super(ConvAutoEncoderCore, self).__init__()
        self.encoder = Encoder(c_hid, latent_dim)
        self.decoder = Decoder(c_hid, latent_dim)
    def forward(self, x):
        return self.decoder(self.encoder(x))
    def features(self, x: torch.Tensor):
        return self.encoder(x)


# child of `Model` class
class ConvAutoEncoder(Model):
    def init_model(self) -> ConvAutoEncoderCore:
        return ConvAutoEncoderCore(
            c_hid = self.options["c_hid"],
            latent_dim = self.options["latent_dim"],
        ).to(self.device)

    def set_loss_function(self):
        return torch.nn.MSELoss()
    
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
            # model output
            outputs = self.model(images)
            # compute loss
            batch_loss = self.loss_function(images, outputs)
            # optimizer step
            batch_loss.backward()
            self.optimizer.step()
            # training loss
            total_train_loss.append(batch_loss.item())
        epoch_train_loss = sum(total_train_loss) / len(total_train_loss)
        # validation phase
        self.model.eval()
        total_val_loss = []
        for i, (images, labels) in enumerate(tqdm(val_loader)):
            with torch.no_grad():
                # make grayscale into 3 channel
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.device)
                # model output
                outputs = self.model(images)
                # compute loss
                batch_loss = self.loss_function(images, outputs)
                total_val_loss.append(batch_loss.item())
        epoch_val_loss = sum(total_val_loss) / len(total_val_loss)
        # update lr
        self.lr_scheduler.step()
        # update model
        if self.best_loss > epoch_val_loss:
            self.best_loss = epoch_val_loss
            self.best_epoch_number = self.current_epoch_number
            self.save_model(
                epoch_val_loss=epoch_val_loss,
            )
        # save loss history
        self.train_loss_history.append(epoch_train_loss)
        self.val_loss_history.append(epoch_val_loss)
        # print
        if self.options["print_mode"]:
            print(
                f"Epoch #{self.current_epoch_number} | Train Loss: {epoch_train_loss:.4f} | " + 
                f"Val Loss: {epoch_val_loss:.4f} | Best Loss: {self.best_loss:.4f}"
            )

    @staticmethod
    def _key():
        return "conv-autoencoder"


# child of `FeatureSpace` class
class ConvAutoEncoderFeatureSpace(FeatureSpace):
    def get_features(self, dset: Dataset):
        self.feature_model.model.eval()
        features = []
        ldr = DataLoader(dset, batch_size=32, shuffle=False)
        for j, (images, labels) in enumerate(tqdm(ldr)):
            with torch.no_grad():
                # make grayscale image into 3 channels
                images = torch.cat([images, images, images], dim=1)
                images = images.to(self.feature_model.device)
                # extract features using `features()` method from ConvAutoEncoderCore
                fts = self.feature_model.model.features(images).detach().cpu().numpy()
                features.append(fts)
        return np.vstack(features)
