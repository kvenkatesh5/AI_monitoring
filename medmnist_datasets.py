import json
import os
import random

import medmnist
from medmnist import INFO, Evaluator
from torchvision import datasets, transforms
from torch.utils.data import Dataset

class AbnominalCTDataset(Dataset):
    def __init__(self, data_dir, split, label_mode="unsupervised", positive_dataset="organamnist", tfms=None, image_size=28, random_seed=2023):
        self.data_flags = [
            "organamnist", "organcmnist", "organsmnist"
        ]
        self.data_order = {
            "organamnist":0, "organcmnist":1, "organsmnist":2
        }
        self.data_list = [
            getattr(medmnist, INFO[self.data_flags[i]]['python_class'])\
                (split=split, transform=tfms, download=True, root=data_dir, as_rgb=False)\
                for i in range(len(self.data_flags))
        ]
        # all three datasets have the same task & label
        self.label_dict = INFO[self.data_flags[0]]['label']
        self.n_classes = len(self.label_dict)

        self.l1, self.l2, self.l3 = [len(x) for x in self.data_list]
        self.length = self.l1 + self.l2 + self.l3
        self.random_order = random.sample(list(range(self.length)), k=self.length)

        if label_mode not in ["unsupervised", "cheap-supervised", "full-supervised"]:
            raise ValueError("label mode not supported: {}".format(label_mode))
        self.label_mode = label_mode
        if self.label_mode == "cheap-supervised":
            if positive_dataset not in self.data_flags:
                raise ValueError("dataset not supported: {}".format(positive_dataset))
            self.positive = self.data_order[positive_dataset]
            self.positive_dataset = positive_dataset
    
    def __getitem__(self, index):
        ri, ai = self.reduce_index(self.random_order[index])
        image, class_label = self.data_list[ai][ri]
        if self.label_mode == "unsupervised":
            return image
        elif self.label_mode == "cheap-supervised":
            dataset_label = (1 if ai==self.positive else 0)
            return image, dataset_label
        else:
            return image, class_label, self.data_flags[ai] 

    def reduce_index(self, j):
        """reduce 'global' index to the correct 'internal dataset' index"""
        if j < self.l1:
            return j, 0
        elif (j < self.l1 + self.l2):
            return j - self.l1, 1
        elif (j < self.length):
            return j - self.l1 - self.l2, 2
        else:
            raise ValueError("index is out of range: {}".format(j))

    def __len__(self):
        return self.length
    
    def get_n_classes(self):
        if self.label_mode == "cheap-supervised":
            return 3
        else:
            # n of image labels
            return self.n_classes
    
    def get_positive_name(self):
        return self.positive_dataset

def debug():
    with open("cfg.json", "r") as f:
        cfg = json.load(f)
    d = AbnominalCTDataset(cfg["data_dir"], "train", label_mode="cheap-supervised")
    print(d[60000])

def calc():
    with open("cfg.json", "r") as f:
        cfg = json.load(f)
    d = AbnominalCTDataset(cfg["data_dir"], "train", label_mode="cheap-supervised")
    # Calculate per-channel (here we have only 1 channel) mean and stdev


if __name__ == "__main__":
    debug()