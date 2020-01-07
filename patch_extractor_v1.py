import os 
from tqdm import tqdm_notebook as tqdm 
import cv2
import random
import numpy as np

def gen_patches(list_images,image_write_path,mask_write_path,discard_path):
    MASK_PATH='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1_mask'
    IMAGE_PATH_NOR='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1'
    
    for image_ID in tqdm(list_images):
        # Format of image_ID : D20180056701_2019-05-21 09_50_58-lv1-27237-28645-8099-3942.jpg
        image_name=os.path.join(IMAGE_PATH_NOR,image_ID)
        gray_img=cv2.imread(image_name,0)
        bgr_img=cv2.imread(image_name,1)
        mask_ID=image_ID.split('.')[0]+'_mask.'+image_ID.split('.')[-1]
        # Format of mask_ID: D20180056701_2019-05-21 09_50_58-lv1-27237-28645-8099-3942_mask.jpg
        mask_name=os.path.join(MASK_PATH,mask_ID)
        mask_img=cv2.imread(mask_name,0)
        ret,mask_img=cv2.threshold(mask_img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)



        th,img_otsu=cv2.threshold(gray_img,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        x_range=(img_otsu.shape[0]-512)//512 +1
        y_range=(img_otsu.shape[1]-512)//512 +1
    
        for count_x in (range(x_range)):
            start_x_pos=count_x*512
            end_x_pos=start_x_pos+512
            for count_y in (range(y_range)):
                start_y_pos=count_y*512
                end_y_pos=start_y_pos+512
                

                temp_otsu=img_otsu[start_x_pos:end_x_pos,start_y_pos:end_y_pos]
                temp_mask=mask_img[start_x_pos:end_x_pos,start_y_pos:end_y_pos]


                if len(np.where(temp_otsu==0)[0])<0.6*len(np.where(temp_otsu==255)[0]) and len(np.where(temp_mask==255)[0])>=5000:
                    img_write=bgr_img[start_x_pos:end_x_pos,start_y_pos:end_y_pos,:]
                    #print(img_write.shape)

                    write_path_image=os.path.join(image_write_path,image_ID.split('.')[0]+\
                                              "_{}_{}.jpg".format(count_x,count_y))
            # Format of image_ID now: D20180056701_2019-05-21 09_50_58-lv1-27237-28645-8099-3942_<count_x>_<count_y>.jpg
                    cv2.imwrite(write_path_image,img_write)
                    write_path_mask=os.path.join(mask_write_path,mask_ID.split('_mask')[0]+\
                                                  "_{}_{}".format(count_x,count_y)+'_mask.jpg')
            # Format of mask_ID now: D20180056701_2019-05-21 09_50_58-lv1-27237-28645-8099-3942_<count_x>_<count_y>_mask.jpg
                    cv2.imwrite(write_path_mask,temp_mask)
#                 else:
                    
# #                     write_path_discard=os.path.join(discard_path,mask_ID.split('.')[0]+\
# #                                                   "{}_{}.jpg".format(count_x,count_y))
# #                     cv2.imwrite(write_path_discard,temp_mask)
                    



def extract_non_overlap_patches(MASK_PATH,IMAGE_PATH,IMAGE_WRITE_PATH,MASK_WRITE_PATH,DISCARD_WRITE_PATH):
    must_include=['18-00530B_2019-05-07 23_56_22-lv1-11712-16122-7372-7686.jpg',\
        '18-01080B_2019-05-07 21_33_52-lv1-15262-19621-5715-4803.jpg',\
        '18-03912A_2019-05-07 22_55_07-lv1-16713-11566-3121-5791.jpg',\
        '18-04842A_2019-05-07 23_36_39-lv1-21930-4017-5709-5966.jpg',\
        '18-09926A_2019-05-08 00_06_27-lv1-23990-14292-4181-5408.jpg',\
        '18-13347A_2019-05-08 01_31_58-lv1-14057-17758-5693-6244.jpg',\
        '2018_68099_1-1_2019-02-20 23_21_13-lv1-7948-15919-4988-5294.jpg',\
        '2018_72876_1-1_2019-02-21 00_28_52-lv1-31103-26670-3231-3747.jpg',\
        '2018_73834_1-1_2019-02-21 00_16_28-lv1-43278-10364-7406-7672.jpg',\
        '2018_74969_1-1_2019-02-21 00_48_39-lv1-39175-17160-3764-3967.jpg',\
        '2018_83220_1-1_2019-02-20 18_33_11-lv1-58151-37497-5712-3460.jpg',\
        '2019_01246_1-1_2019-02-20 19_23_52-lv1-41847-6127-5995-4660.jpg',\
        '2019_02170_1-2_2019-02-20 19_37_17-lv1-18315-29727-7652-5581.jpg',\
        '2019_03867_1-1_2019-02-20 20_00_32-lv1-36172-15430-6105-5822.jpg',\
        '2019_05944_1-1_2019-02-20 20_42_06-lv1-31069-20357-7018-5934.jpg',\
        '1901940-1_2019-04-30 10_30_47-lv1-39947-11074-3848-5785.jpg',\
        '1800883002_2019-04-30 09_57_31-lv1-28885-30819-3472-3263.jpg',\
        'D201710920_2019-05-21 11_52_57-lv1-33661-31501-7087-7660.jpg',\
        'D201711541_2019-05-21 11_28_51-lv1-39189-27851-3979-5553.jpg',\
        'D201802733_2019-05-14 15_41_01-lv1-18828-14936-7756-5310.jpg']
    
    image_path_list=[IMAGE_WRITE_PATH+'_train',IMAGE_WRITE_PATH+'_val',IMAGE_WRITE_PATH+'_test']
    for write_path in image_path_list:
        file_name=write_path.split('/')[-1]
        if not os.path.exists(write_path):
            os.mkdir(write_path)
            print('{} directory made'.format(file_name))
        else:
            print('{} directory exists'.format(file_name))
        
   
    mask_path_list=[MASK_WRITE_PATH+'_train',MASK_WRITE_PATH+'_val',MASK_WRITE_PATH+'_test']

    for write_path in mask_path_list:
        file_name=write_path.split('/')[-1]
        if not os.path.exists(write_path):
            os.mkdir(write_path)
            print('{} directory made'.format(file_name))
        else:
            print('{} directory exists'.format(file_name))

    if not os.path.exists(DISCARD_WRITE_PATH):
        os.mkdir(DISCARD_WRITE_PATH)
        print('discard directory made')
    else:
        print('discard directory exists')

    image_list=os.listdir(IMAGE_PATH)
    for x in must_include:
        image_list.remove(x)
    
    
    random.shuffle(image_list)
    train_list_temp=image_list[:int(0.80*len(image_list))]

    train_list=train_list_temp[:int(0.80*len(train_list_temp))]
    train_list.extend(must_include)
    validation_list=train_list_temp[int(0.80*len(train_list_temp)):]
    test_list=image_list[int(0.80*len(image_list)):]
    print('TRAIN PATCHES')
    gen_patches(train_list,IMAGE_WRITE_PATH+'_train',MASK_WRITE_PATH+'_train',DISCARD_WRITE_PATH)
    print('VALIDATION PATCHES')
    gen_patches(validation_list,IMAGE_WRITE_PATH+'_val',MASK_WRITE_PATH+'_val',DISCARD_WRITE_PATH)
    print('TEST PATCHES')
    gen_patches(test_list,IMAGE_WRITE_PATH+'_test',MASK_WRITE_PATH+'_test',DISCARD_WRITE_PATH)


                    
MASK_PATH='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1_mask'
IMAGE_PATH_NOR='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1_color_normalized'
IMAGE_PATH_NN='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/tissue-train-pos-v1'
IMAGE_WRITE_PATH='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/input_cn_non_overlap'
MASK_WRITE_PATH='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/mask_cn_non_overlap'
DISCARD_WRITE_PATH='/home/vahadaneabhi01/datalab/digest/Colonoscopy_tissue_segment_dataset/discard_cn_non_overlap'

 
extract_non_overlap_patches(MASK_PATH,IMAGE_PATH_NOR,IMAGE_WRITE_PATH,MASK_WRITE_PATH,DISCARD_WRITE_PATH)
