# %%
from pathlib import Path
import random

import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

from utils import set_seed

def prep_dataset(adult_datadir, peds_datadir, train_pct=0.5,split='train'):
    all_adult_fnames = list(adult_datadir.rglob('images_0*/images/*.png'))

    if (split == 'train') or (split == 'val'):
        with open(adult_datadir/'train_val_list.txt') as f:
            adult_train_val = list(map(lambda x: x.strip(), f.readlines()))

        peds_fnames = list(peds_datadir.rglob('train/NORMAL/*.jpeg'))

    if split == 'train':
        print('Prepping dataset: filtering adult dataset into training')
        adult_train = adult_train_val[:round(len(adult_train_val)*train_pct)] 
        adult_fnames = [f for f in all_adult_fnames if f.name in adult_train]

        peds_fnames = peds_fnames[:round(len(peds_fnames)*train_pct)]
    
    elif split == 'val':
        print('Prepping dataset: filtering adult dataset into validation')
        adult_val = adult_train_val[round(len(adult_train_val)*train_pct):]
        adult_fnames = [f for f in all_adult_fnames if f.name in adult_val]

        peds_fnames = peds_fnames[round(len(peds_fnames)*train_pct):] 

    elif split == 'test':
        print('Prepping dataset: filtering adult dataset into testing')
        with open(adult_datadir/'test_list.txt') as f:
            adult_test = list(map(lambda x: x.strip(), f.readlines()))
        adult_fnames = [f for f in all_adult_fnames if f.name in adult_test]

        peds_fnames = list(peds_datadir.rglob('test/NORMAL/*.jpeg'))
    else:
        ValueError('Invalid split: options [`train`, `val`, `test`]')
   
    return {'adult': adult_fnames, 'peds': peds_fnames}

class PediatricCXRDataset(Dataset):
    '''
    self.positive indicates pediatric ood, self.negative indicates adult data
    '''
    def __init__(self, data_dir, adult_data_dir, peds_data_dir, split, positive_dataset="pediatric", \
                 tfms=None, random_seed=1001):
        # set random seeds
        set_seed(random_seed)
        
        # set dataset transform
        self.tfms = tfms

        self.data_flags = [
            "adult", "pediatric"
        ]
        self.data_order = {
           "adult":0,"pediatric":1, 
        }
        cache_fname = Path(f'{data_dir}/kaggle_ped_cxr_{split}.npz')
        if not cache_fname.exists():
            print('Preparing dataset for the first time...')
            dset_dict = prep_dataset(Path(adult_data_dir), Path(peds_data_dir), split=split)
            if split == 'train':
                self.data_list = []
                self.data_list += list(zip(dset_dict['adult'], len(dset_dict['adult'])*[self.data_order['adult']]))#adult_fnames_train)
                self.data_list += list(zip(dset_dict['peds'], len(dset_dict['peds'])*[self.data_order['pediatric']]))#peds_fnames_train)
            elif split == 'val':
                self.data_list = []
                self.data_list += list(zip(dset_dict['adult'], len(dset_dict['adult'])*[self.data_order['adult']]))#adult_fnames_val)
                self.data_list += list(zip(dset_dict['peds'], len(dset_dict['peds'])*[self.data_order['pediatric']]))#peds_fnames_val)
            elif split == 'test':
                self.data_list = []
                self.data_list += list(zip(dset_dict['adult'], len(dset_dict['adult'])*[self.data_order['adult']])) #adult_fnames_test)
                self.data_list += list(zip(dset_dict['peds'], len(dset_dict['peds'])*[self.data_order['pediatric']]))#peds_fnames_test)
            else:
                ValueError(f'{split} does not match available options: [`train`, `test`]')

            im_array = []
            labels = []
            for idx, fname in enumerate(tqdm(self.data_list)):
                img, lbl = self.__getitem__(idx)
                im_array.append(img)
                labels.append(lbl)
            im_array = np.array(im_array)
            labels = np.array(labels)
            np.savez(cache_fname, im_array=im_array, labels=labels)
        
        dset = np.load(cache_fname)
        self.data_list = list(zip(dset['im_array'], dset['labels']))
        # self.random_order = random.sample(list(range(len(self.data_list))), k=len(self.data_list))


    def __getitem__(self, index):
        im_data, lbl = self.data_list[index]
        image = plt.imread(im_data) if isinstance(im_data, (str, Path)) else im_data
        if image.ndim > 2:
            image = image.mean(axis=-1) #some images are rgb or rgba so convert to grayscale
        if self.tfms:
            image = self.tfms(image)
        return image, lbl

    def __len__(self):
        return len(self.data_list)
    
    @staticmethod
    def get_default_transform() -> transforms.Compose:
        dataset_transforms = transforms.Compose([
            transforms.ToTensor(), transforms.Resize(size=(28, 28)), transforms.Normalize(mean=[.5], std=[.5]) #28 x 28 is way too small!
        ])
        return dataset_transforms


# %%

if __name__ == '__main__':
    adult_datadir = Path('/gpfs_projects/brandon.nelson/AI_monitoring/NIH_CXR')
    peds_datadir = Path('/gpfs_projects/brandon.nelson/AI_monitoring/Pediatric Chest X-ray Pneumonia')
    dset = PediatricCXRDataset(adult_data_dir=adult_datadir, peds_data_dir=peds_datadir, split='train')
# %%
    dset[0]
# %%
