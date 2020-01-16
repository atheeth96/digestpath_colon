import skimage
from skimage.io import imread,imsave
from skimage import morphology
from skimage.filters import threshold_otsu
from skimage import img_as_uint
from tqdm import tqdm_notebook as tqdm
import os 
import numpy as np
import math
from skimage import img_as_ubyte


import imageio.core.util

def ignore_warnings(*args, **kwargs):
    pass

imageio.core.util._precision_warn = ignore_warnings



import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import random
import matplotlib.pyplot as plt
import cv2


class EvaluatePrediction():
    
    def __init__(self,MAP1_PATH,MAP2_PATH,MASK_PATH,model):
        
        
        #IMAGE_PATH : Path to the images main dir
        #DAPI_PATH : Path to dapi images
        #biomarker : biomarker for which to perfom count prediction
        #model : pytorch model
        
        
        self.MAP1_PATH=MAP1_PATH
        self.MAP2_PATH=MAP2_PATH
        self.MASK_PATH=MASK_PATH
        self.model=model
        
  
        

    def whole_dice_metric(self,y_pred,y_true):
        smooth = 10e-16
        # single image so just roll it out into a 1D array

        m1 =np.reshape(y_pred,(-1))/255
        m2 =np.reshape(y_true,(-1))/255


        intersection = (m1 * m2)

        score =  (2*(np.sum(intersection)) + smooth) / (np.sum(m1) +(np.sum(m2) + smooth))

        return score
    
    def track_iu(self,y_pred,y_true):
       
        m1 =np.reshape(y_pred,(-1))/255
        m2 =np.reshape(y_true,(-1))/255


        intersection = np.sum(m1 * m2)

        union= np.sum(m1) +(np.sum(m2))
        return intersection,union
    


    def whole_img_pred(self,img_list,pred_dir_name,threshold=0.5,print_prompt=False):

        
        #img_list : List of images in the IMAGE_PATH for which to predict masks
        #pred_dir_name : directory (which will be created in pwd if necessary) where the predictions will be saved
        
        #print_prompt : boolean whether to print prompts
        MAP1_PATH=self.MAP1_PATH               
        MAP2_PATH=self.MAP2_PATH
        MASK_PATH=self.MASK_PATH
        


        pred_dir=os.path.join(os.getcwd(),pred_dir_name)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        model=self.model.to(device)
        model.eval()

        if not os.path.exists(pred_dir):
            os.mkdir(pred_dir)
            if print_prompt:
                print("Made {} directory".format(pred_dir.split('/')[-1]))
        else:
            if print_prompt:
                print("{} directory already exists in {}".format(pred_dir.split('/')[-1],'/'.join(pred_dir.split('/')[:-1])))

        step=512
        avg_dice=0
        loop=tqdm(img_list)
        intersection=0
        union=0
        for count,img_name in enumerate(loop):


            map1_image=imread(os.path.join(MAP1_PATH+'/Biomarkers/{}'.format(self.biomarker),img_name))
            map2_image=imread(os.path.join(MAP2_PATH+'/Biomarkers/{}'.format(self.biomarker),img_name))
            


            mask_name=img_name.split('.')[0]+'_mask.jpg'
            mask_image=imread(os.path.join(MASK_PATH,mask_name))
            


            r,c=map1_image.shape[:2]#4663,3881

            new_r_count=(math.ceil((r-512)/512)+1)#5
            new_c_count=(math.ceil((c-512)/512)+1)#5


            pad_r1=((new_r_count-1)*512-r+512)//2 #200
            pad_r2=((new_r_count-1)*512-r+512)-pad_r1 #200
            pad_c1=((new_c_count-1)*512-c+512)//2 #0
            pad_c2=((new_c_count-1)*512-c+512)-pad_c1#0

            map1_image_padded=np.pad(map1_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)
            map2_image_padded=np.pad(map2_image, [(pad_r1,pad_r2),(pad_c1,pad_c2)], 'constant', constant_values=0)


            window_shape=(512,512)

            map1_patches=skimage.util.view_as_windows(image_padded, window_shape, step=step)
            map1_patches=map1_patches.reshape((-1,512,512)).astype(np.float32)
            map1_patches=map1_patches.transpose((0,2,1))/255
            
            
            map2_patches=skimage.util.view_as_windows(image_padded, window_shape, step=step)
            map2_patches=map2_patches.reshape((-1,512,512)).astype(np.float32)
            map2_patches=map2_patches.transpose((0,2,1))/255
            
            
    
            mask_temp=[]


            for i in range(new_r_count):

                temp_map1_patches=torch.from_numpy(map1_patches[i*new_r_count:(i+1)*new_r_count]).type(torch.FloatTensor).to(device)
                temp_map2_patches=torch.from_numpy(map2_patches[i*new_r_count:(i+1)*new_r_count]).type(torch.FloatTensor).to(device)
                
                mask=torch.sigmoid(model(temp_map1_patches,temp_map2_patches))
                del temp_map1_patches,temp_map2_patches

                mask=mask.detach().cpu().numpy()

                mask=np.squeeze(mask,axis=1).transpose((0,2,1))
                mask=np.concatenate(mask,axis=1)

                mask_temp.append(mask)



            mask_temp=np.array(mask_temp)
            mask_temp=np.concatenate(mask_temp,axis=0)


            mask_temp=mask_temp[pad_r1:mask_temp.shape[0]-pad_r2,pad_c1:mask_temp.shape[1]-pad_c2]*255

            mask_temp=mask_temp.astype(np.uint8)

            dice_score=self.whole_dice_metric(mask_temp,mask_image*255)
            avg_dice+=dice_score
            
  
            if print_prompt:
                loop.set_postfix(Dice_score =dice_score,average=avg_dice/(count+1))




            imsave(pred_dir_name+img_name.split('.')[0]+'_pred.jpg',mask_temp)




        if print_prompt:
            print("DONE\nAverage Dice Score = {}".format((2*(intersection)+1e-15)/(union+1e-15)))



