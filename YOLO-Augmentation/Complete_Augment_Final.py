#!/usr/bin/env python
# coding: utf-8

# # Define Directory (Source, Converted and Result)

# In[1]:


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
src_img_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master\dataset_awal/resource_img'
src_label_dir = 'D:\Project\Machine Learning\Projects\CV/Ultralitycs\yolov3-master\dataset_awal/resource_label'

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

# In[2]:


def create_dir(label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir):
    list_dir = list([label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir])
    for i in list_dir:
        if not os.path.exists(i):
            os.makedirs(i)
            
create_dir(label_dir, img_dir, bbox_dir, dest_img_dir, dest_label_dir)


# # Copy Random non-reapeated Image from Source

# In[3]:


# number of data to be copied from source
n_data = 607
# number of agmentation result from 1 data (include original one)
n_augment = 7
#range of rescale
range_scaling = (-0.7, 0.3)
img_resize = 360

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

# In[4]:


import os
from PIL import Image

def convert_yolo_to_bbox(data, image_shape):
    new_data = []
    for line in data:
        line = line.split(' ')
        line = [float(i) for i in line]
        c, center_x, center_y, width, height = line

        denorm_width = width * image_shape[0]
        denorm_height = height * image_shape[1]

        x1, x2 = center_x * image_shape[0] - denorm_width/2 , center_x * image_shape[0] + denorm_width/2
        y1, y2 = center_y * image_shape[1] - denorm_height/2, center_y * image_shape[1] + denorm_height/2
        
        list_bbox = list([round(x1,3), round(y1,3), round(x2,3), round(y2,3), int(c)])      
        new_data.append(" ".join(str(i) for i in list_bbox))
    
    strings = "\n".join(new_data) 
    return strings


# # Create BBOX to YOLO Converter

# In[5]:


def convert_bbox_to_yolo(data, image_shape):
    new_data = []
    #convert bbox format to yolo (normalization with image shape)
    for bbox in data:
        bboxes_ = [float(i) for i in bbox]
        x_center_norm = (bboxes_[0]+(bboxes_[2]-bboxes_[0])/2)/image_shape[1]
        y_center_norm = (bboxes_[1]+(bboxes_[3]-bboxes_[1])/2)/image_shape[0]
        width_norm = (bboxes_[2]-bboxes_[0])/image_shape[1]
        height_norm = (bboxes_[3]-bboxes_[1])/image_shape[0]
        obj_class = bboxes_[-1]

        #create list as yolo format and return as string
        list_yolo = list([int(obj_class),round(x_center_norm, 3), round(y_center_norm, 3), round(width_norm, 3), round(height_norm,3)])
        new_data.append(" ".join(str(i) for i in list_yolo))

    strings = "\n".join(new_data) 
    return strings


# # Create New BBOX label file

# In[6]:


for file in os.listdir(label_dir):
    image = Image.open(img_dir+'\\'+file.split('.txt')[0]+'.jpg')
    img_size = image.size
    
    content = label_dir+'//'+file
    lines = [line.strip() for line in open(content)]
    
    strings = convert_yolo_to_bbox(lines, img_size)
    
    f = open(bbox_dir+'\\'+file, "w")
    f.write(strings)
    f.close()


# # Do Augmentation and Store to Result Directory

# In[7]:


for file in os.listdir(img_dir):
    
    for i in range(n_augment):
        try:
            file_name = file.split('.jpg')[0]
            img = cv2.imread(img_dir + '/' + file)[:,:,::-1]
            
            file_label = bbox_dir + '/' + file_name + '.txt'
            lines = [line.strip() for line in open(file_label)]
            bboxes = []
            for a in [line.split(' ') for line in lines]:
                bboxes.append([float(i) for i in a])
            bboxes = np.array(bboxes)
            
            if i == 0:
                img_, bboxes_ = Resize(img_resize)(img.copy(), bboxes.copy())
                img_size = img_.shape[:2]
                
                scipy.misc.imsave(dest_img_dir + '/' + file_name + '.jpg', img_)
                f = open(dest_label_dir + '/' + file_name + '.txt', "w")
                
                str_yolo = convert_bbox_to_yolo(bboxes_,img_size)
                f.write(str_yolo)
                f.close()
                continue
            
            #augment the images
            seq = Sequence([RandomHSV(40, 40, 30),RandomHorizontalFlip(), RandomScale(scale = range_scaling), RandomTranslate(0.15), RandomRotate(10), RandomShear(), Resize(img_resize)])
            img_, bboxes_ = seq(img.copy(), bboxes.copy())
            img_size = img_.shape[:2]

            str_yolo = convert_bbox_to_yolo(bboxes_,img_size)

            #save augmented result (image and labels)
            scipy.misc.imsave(dest_img_dir + '/' + file_name + '_'+ str(i) +'.jpg', img_)
            f = open(dest_label_dir + '/' + file_name + '_' + str(i) +'.txt', "w")
            f.write(str_yolo)
            f.close()

        
        except:
            pass


# In[ ]:




