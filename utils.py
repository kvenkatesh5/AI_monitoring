"""
Helper functions
"""
import random

import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset

def set_seed(random_seed: int):
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

