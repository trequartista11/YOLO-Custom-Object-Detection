#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_cell_magic('time', '', "\nimport matplotlib.image as im\nfrom PIL import Image\nimport scipy.misc\nimport os\nfrom os import path\nimport shutil\nimport random\nimport torchvision\nimport numpy as np\nfrom skimage.util import random_noise\n\n\n\ndef augment(img, img_name, dest_path, new_name_ext,n_output):\n\n    grad_gamma = list((0.3,1,2,3,4))\n    grad_sigma = list((0.1,0.2, 0.3))\n    grad_bright = list((0.25,0.5,1,2,3,4))\n    new_files = []\n        \n    #create n augmented images (exclude original one)\n    for i in range(n_output+1):\n        \n        #reproduce original image\n        if i == 0:\n            pict3 = np.array(img)\n            file_name = img_name + new_name_ext + str(i)\n            scipy.misc.imsave(dest_path + '/' + file_name + '.jpg', pict3)\n            new_files.append(file_name)\n            continue\n            \n        #Get random element of gamma, brightness and sigma (for noise)\n        gamma_fct = grad_gamma[random.randint(0,len(grad_gamma)-1)]\n        bright_fct = grad_bright[random.randint(0,len(grad_bright)-1)]\n        sigma_fct = grad_sigma[random.randint(0,len(grad_sigma)-1)]\n        \n        #implement brightness manipulation\n        pict1 = torchvision.transforms.functional.adjust_brightness(img, bright_fct)\n        \n        #change the pict to numpy array for noise manipulation implementation\n        pix = np.array(pict1)\n        pict3 = random_noise(pix,var=sigma_fct**2)\n        \n        file_name = img_name + new_name_ext + str(i)\n        scipy.misc.imsave(dest_path + '/' + file_name + '.jpg', pict3)\n        \n        #save list of new images name to be consumed in main code for label creation\n        new_files.append(file_name)\n\n    \n    return new_files\n\n#create function to replicate label\ndef copy_rename(old_name, new_name, source_path, dest_path):\n    src_file = source_path + old_name\n    shutil.copy(src_file, dest_path)\n    os.rename(dest_path + old_name, dest_path + new_name)\n\n\n\nimage_path = 'D:\\Project\\Machine Learning\\Projects\\\\CV\\\\Ultralitycs\\yolov3-master\\\\truck\\images\\\\train/'\n#image_path = 'D:\\Project\\Machine Learning\\Projects\\CV/Ultralitycs\\yolov3-master\\dataset_awal/tes aug/augment_result/'\nlabel_path = 'D:\\Project\\Machine Learning\\Projects\\\\CV\\\\Ultralitycs\\yolov3-master\\\\truck\\labels\\\\train/'\ndest_path_img = './augmented/images/'\ndest_path_lab = './augmented/labels/'\nnew_name_ext = '_aug_'\nnum_output = 5\n\nif not path.exists(dest_path_img):\n        os.makedirs(dest_path_img)\n        \nif not path.exists(dest_path_lab):\n        os.makedirs(dest_path_lab)\n\nfor image in os.listdir(image_path):\n    img = image_path+'\\\\'+image\n    img = Image.open(img)\n    img_name = image.split('.')[0]\n    file_list = augment(img, img_name, dest_path_img, new_name_ext, num_output)\n    \n    #create data label with new name\n    for file in file_list:\n        old_name = img_name + '.txt'\n        new_name = file + '.txt'\n        \n        copy_rename(old_name, new_name, label_path, dest_path_lab)")


# In[ ]:




