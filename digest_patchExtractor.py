

import numpy as np
import glob
import os
import sys
import scipy

import matplotlib.pyplot as plt
#import cv2
import imutils
import re


import zipfile

from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from sklearn import metrics
from sklearn.utils import shuffle
from sklearn.feature_extraction import image


import skimage
from skimage.util import pad
from skimage import draw
from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.filters import threshold_otsu

import time
import math

import random
from tqdm import tqdm_notebook as tqdm 



class DualPatchExtractor():
    def __init__(self,map1_dir,map2_dir,mask_dir,map1_patch_dir,map2_patch_dir\
    ,mask_patch_dir,gray=True,overlap=False,progress_bar=True):
    
        
        self.map1_dir=map1_dir
        self.map2_dir=map2_dir
        self.mask_dir=mask_dir
        
        self.map1_patch_dir=map1_patch_dir
        self.map2_patch_dir=map2_patch_dir
        self.mask_patch_dir=mask_patch_dir
        
        self.gray=gray
        self.overlap=overlap
        self.progress_bar=progress_bar
        

    def extract_patches(self):
        if self.overlap:
            step=256
        else:
            step=512
        NO_PATCHES=0
        
#       for tag in ['train','test']:
        map1_patch_dir=self.map1_patch_dir
        map2_patch_dir=self.map2_patch_dir
        mask_patch_dir=self.mask_patch_dir

        if not os.path.exists(map1_patch_dir):
            os.mkdir(map1_patch_dir)
            if self.progress_bar:
                print("Map1 patch directory made")
        if not os.path.exists(map2_patch_dir):
            os.mkdir(map2_patch_dir)
            if self.progress_bar:
                print("Map2 patch directory made")
        if not os.path.exists(mask_patch_dir):
            os.mkdir(mask_patch_dir)
            if self.progress_bar:
                print("Mask patch directory made")
      
        exception_list=[]
        
       
        slides=[x for x in os.listdir(self.map1_dir) if '.jpg' in x\
                and x in os.listdir(self.map2_dir) and x.split('.')[0]+'_mask.jpg' in os.listdir(self.mask_dir)]

        if self.progress_bar:
            loop=tqdm(slides)
        else:
            loop=slides
        for slide in loop:
 
            try:

                map1_path=os.path.join(self.map1_dir,slide)
                map2_path=os.path.join(self.map2_dir,slide)
                mask_path=os.path.join(self.mask_dir,slide.split('.')[0]+'_mask.jpg')






                map1_img=imread(map1_path)
                map2_img=imread(map2_path)
                mask_img=imread(mask_path)
                mask_img=mask_img>0
                mask_img=mask_img.astype(np.uint8)*255


                r,c=map1_img.shape[:2]

                new_r_count=(math.ceil((r-512)/512)+1)
                new_c_count=(math.ceil((c-512)/512)+1)


                pad_r1=((new_r_count-1)*512-r+512)//2 
                pad_r2=((new_r_count-1)*512-r+512)-pad_r1 
                pad_c1=((new_c_count-1)*512-c+512)//2 
                pad_c2=((new_c_count-1)*512-c+512)-pad_c1 

                if self.gray:
                    map1_padded=np.pad(map1_img, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
                    map2_padded=np.pad(map2_img, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
                    window_shape=(512,512)
                else:
                    map1_padded=np.pad(map1_img, [(pad_r1,pad_r2),(pad_c1,pad_c2),(0,0)], 'constant', constant_values=0)
                    map2_padded=np.pad(map2_img, [(pad_r1,pad_r2),(pad_c1,pad_c2),(0,0)], 'constant', constant_values=0)#/np.amax(input_image)
                    window_shape=(512,512,3)
                mask_padded=np.pad(mask_img, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)


                window_shape_mask=(512,512)

                map1_patches=skimage.util.view_as_windows(map1_padded, window_shape, step=step)
                map2_patches=skimage.util.view_as_windows(map2_padded, window_shape, step=step)
                mask_patches=skimage.util.view_as_windows(mask_padded, window_shape_mask, step=step)

                if self.gray:
                    map1_patches=map1_patches.reshape((-1,512,512))
                    map2_patches=map2_patches.reshape((-1,512,512))
                else:
                    map1_patches=map1_patches.reshape((-1,512,512,3))
                    map2_patches=map2_patches.reshape((-1,512,512,3))


                mask_patches=mask_patches.reshape((-1,512,512))


                for i,(map1_patch,map2_patch,mask_patch) in enumerate(zip(map1_patches,map2_patches,mask_patches)):
                    if len(np.where(mask_patch==255)[0])>=3000:
                        NO_PATCHES+=1


                        imsave(os.path.join(map1_patch_dir,slide.split('.')[0]+'_{}.jpg'.format(i+1)),map1_patch)
                        imsave(os.path.join(map2_patch_dir,slide.split('.')[0]+'_{}.jpg'.format(i+1)),map2_patch)
                        imsave(os.path.join(mask_patch_dir,slide.split('.')[0]+'_{}_mask.jpg'.format(i+1)),mask_patch)

            except:
                exception_list.append(slide.split('.')[0])
        if self.progress_bar:
            print("NUMBER OF PATCHES ARE {}".format(NO_PATCHES),'\nException\n',*exception_list,sep='\n')