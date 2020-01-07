from skimage.io import imread
import numpy as np
import os
import skimage
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import random
import numpy as np
def Sort_Tuple(tup):  
      
    # getting length of list of tuples 
    lst = len(tup)  
    for i in range(0, lst):  
          
        for j in range(0, lst-i-1):  
            if (tup[j][0] > tup[j + 1][0]):  
                temp = tup[j]  
                tup[j]= tup[j + 1]  
                tup[j + 1]= temp  
    return tup  


class DataSet(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, map1_dir, map2_dir,mask_dir,gray=True, transform=None):
        
        self.map1_dir=map1_dir
        self.map2_dir=map2_dir
        self.mask_dir = mask_dir
        
        self.gray=gray
        self.transform = transform
        self.img_list=os.listdir(map1_dir)
        
    def __len__(self):
        return len([x for x in os.listdir(self.map1_dir) if x.split('.')[-1]=='jpg'])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
      

        map_1_name = os.path.join(self.map1_dir,
                                self.img_list[idx])
        
        map_2_name = os.path.join(self.map2_dir,
                                self.img_list[idx])
        
        mask_name = os.path.join(self.mask_dir,
                                self.img_list[idx].split('.')[0]+'_mask.jpg')
        #print(img_name)
        
        map_1_image = imread(map_1_name)
        map_2_image = imread(map_2_name)
        if self.gray:
            map_1_image = np.expand_dims(map_1_image,axis=2)
            map_2_image = np.expand_dims(map_2_image,axis=2)
        
        mask_image=np.expand_dims(imread(mask_name),axis=2)
        
        sample = {'map1': map_1_image, 'map2': map_2_image,'mask':mask_image}

        if self.transform:
            sample = self.transform(sample)

        return sample
class Scale(object):
    """Convert ndarrays in sample to Tensors."""
  

    def __call__(self,sample):
        map1, map2,mask = sample['map1'], sample['map2'],sample['mask']
        
        scale=255

        map1 = map1/scale
        
        map2 = map2/scale
        mask=mask/scale
        return {'map1': map1,
                'map2': map2,
               'mask': mask}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
  
    def __call__(self, sample):
        map1, map2,mask = sample['map1'], sample['map2'],sample['mask']
        

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        
        
        map1 = map1.transpose((2, 0, 1))
        map2 = map2.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        return {'map1': torch.from_numpy(map1).type(torch.FloatTensor),
                'map2': torch.from_numpy(map2).float(),
               'mask': torch.from_numpy(mask).float()}
    



    
class  Color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'