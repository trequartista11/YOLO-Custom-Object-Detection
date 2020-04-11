#!/usr/bin/env python
# coding: utf-8

# # Define Directory (Source, Converted and Result)

# In[35]:


import shutil
import  random
from data_aug.data_aug import *
from data_aug.bbox_util import *
import numpy as np 
import cv2 
import matplotlib.pyplot as plt
import scipy.misc
from tqdm import tqdm

#Define source data
src_img_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master\dataset_awal/images'
src_label_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master\dataset_awal/label'

#main directory to replicate data
main_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master'
label_dir = main_dir + '/replicate/labels'
img_dir = main_dir + '/replicate/images'
bbox_dir = main_dir + '/replicate/labels2'

#result directory
return_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master'
dest_img_dir = return_dir + '/aug_result/images'
dest_label_dir = return_dir + '/aug_result/labels'


# # Create Directory

# In[40]:


def create_dir(label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir):
    list_dir = list([label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir])
    for i in list_dir:
        if not os.path.exists(i):
            os.makedirs(i)
            
create_dir(label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir)


# # Copy Random non-reapeated Image from Source

# In[41]:


# number of data to be copied from source
n_data = 500
# number of agmentation result from 1 data (include original one)
n_augment = 15
#range of rescale
range_scaling = (-0.9, -0.2)

files = os.listdir(src_img_dir)
files = [i for i in files if i.split('.')[1] == 'jpg']
idx = []
for i in range(n_data):
    r = random.randint(0,len(files)-1)
    if r not in idx: idx.append(r)
        
for ids in idx:
    shutil.copy(src_img_dir+ '/'+ files[ids], img_dir)
    shutil.copy(src_label_dir+ '/'+ files[ids].split('.')[0] + '.txt', label_dir)


# # Create YOLO to BBOX Converter

# In[42]:


import os
from PIL import Image

def convert_yolo_to_bbox(data, image_shape):
    
    c, center_x, center_y, width, height = data

    denorm_width = width * image_shape[0]
    denorm_height = height * image_shape[1]
    x1, x2 = center_x * image_shape[0] - denorm_width/2 , center_x * image_shape[0] + denorm_width/2
    y1, y2 = center_y * image_shape[1] - denorm_height/2, center_y * image_shape[1] + denorm_height/2

    new_data = list([round(x1,2), round(y1,2), round(x2,2), round(y2,2), int(c)])
    
    return new_data


# # Create BBOX to YOLO Converter

# In[43]:


def convert_bbox_to_yolo(bbox):
    
    #convert bbox format to yolo (normalization with image shape)
    bboxes_ = bbox[0]
    x_center_norm = (bboxes_[0]+(bboxes_[2]-bboxes_[0])/2)/img_.shape[1]
    y_center_norm = (bboxes_[1]+(bboxes_[3]-bboxes_[1])/2)/img_.shape[0]
    width_norm = (bboxes_[2]-bboxes_[0])/img_.shape[1]
    height_norm = (bboxes_[3]-bboxes_[1])/img_.shape[0]
    obj_class = bboxes_[-1]
    
    #create list as yolo format and return as string
    yolo_list = list([int(obj_class),round(x_center_norm, 2), round(y_center_norm, 2), round(width_norm, 2), round(height_norm,2)])
    yolo_list = [str(i) for i in yolo_list]
    str_yolo = str(yolo_list[0]+ ' ' + yolo_list[1] + ' ' + yolo_list[2] + ' ' + yolo_list[3] + ' ' + yolo_list[4])
    return str_yolo


# # Create New BBOX label file

# In[44]:


for file in os.listdir(label_dir):
    image = Image.open(img_dir+'\\'+file.split('.txt')[0]+'.jpg')
    img_size = image.size
    
    content = open(label_dir+'//'+file, 'r').read().split()
    float_a = [float(i) for i in content]
    new_data = convert_yolo_to_bbox(float_a, img_size)
    string = [str(i) for i in new_data]
    string = str(string[0]+' '+string[1]+' '+string[2]+' '+string[3]+' '+string[4])
    f = open(bbox_dir+'\\'+file, "w")
    f.write(string)
    f.close()


# # Do Augmentation and Store to Result Directory

# In[45]:


for file in os.listdir(img_dir):
    for i in range(n_augment):
        try:
            file_name = file.split('.jpg')[0]        
            img = cv2.imread(img_dir + '/' + file)[:,:,::-1]

            

            #opencv loads images in bgr. the [:,:,::-1] does bgr -> rgb
            bboxes = open(bbox_dir + '/' + file_name + '.txt', 'r').read().split()
            bboxes = np.array([[float(i) if x <= 3 else int(i) for x,i in enumerate(bboxes) ]])

            if i == 0:
                scipy.misc.imsave(dest_img_dir + '/' + file_name + '.jpg', img)
                f = open(dest_label_dir + '/' + file_name + '.txt', "w")
                str_yolo = convert_bbox_to_yolo(bboxes)
                f.write(str_yolo)
                f.close()
                continue
            
            #augment the images
            seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(scale = range_scaling), RandomTranslate(0.15), RandomRotate(10), RandomShear()])
            img_, bboxes_ = seq(img.copy(), bboxes.copy())

            #return to yolo format
            str_yolo = convert_bbox_to_yolo(bboxes_)

            #save augmented result (image and labels)
            scipy.misc.imsave(dest_img_dir + '/' + file_name + '_'+ str(i) +'.jpg', img_)
            f = open(dest_label_dir + '/' + file_name + '_' + str(i) +'.txt', "w")
            f.write(str_yolo)
            f.close()
        
        except:
            pass


# In[ ]:




