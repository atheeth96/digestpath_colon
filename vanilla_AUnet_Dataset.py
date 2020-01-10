from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np



class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, img_dir, mask_dir, transform=None):
     
        self.img_dir=img_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.img_list=[]
        self.img_list=os.listdir(img_dir)
#Returns length of data-set unlike its keras counter part that returns no_batches
    def __len__(self):
        return len([x for x in os.listdir(self.img_dir) if x.split('.')[-1]=='jpg'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        

        img_name = os.path.join(self.img_dir,
                                self.img_list[idx])
   
        mask_name = os.path.join(self.mask_dir,
                                [x for x in os.listdir(self.mask_dir) if img_name.split('/')[-1].split('.')[0] in x][-1])
        
        image = imread(img_name)
        
        mask=imread(mask_name)
        
        sample = {'image': image, 'mask': mask}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Scale(object):
    """Convert ndarrays in sample to Tensors."""
  

    def __call__(self,sample):
        image, mask = sample['image'], sample['mask']

        
        return {'image': image/np.amax(image),
                'mask': mask/np.amax(mask)}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, mask = sample['image'], sample['mask']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        
        mask = np.expand_dims(mask,axis=2).transpose((2, 0, 1))
        
        return {'image': torch.from_numpy(image).type(torch.FloatTensor),
                'mask': torch.from_numpy(mask).float()}
    
