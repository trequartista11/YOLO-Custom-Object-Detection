#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from os import path

image_path = 'D:\\Project\\Machine Learning\\Projects\\CV\\Ultralitycs\\yolov3-master/aug_result/images'
file_train = open('./file_train.txt', "w")

for file in os.listdir(image_path):
    file_train.write(image_path+ '/' + file + "\n")
file_train.close()


# In[ ]:




