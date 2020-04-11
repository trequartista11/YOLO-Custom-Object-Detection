# YOLO-Custom-Object-Detection-With Augmentation
Workflow of doing custom dataset training using YOLO framework



# 1. Prepare your dataset
Prepare your image dataset to be annotated

# 2. Annotate your images
Use free labelling an annotation tool to create anntotation and label, more information about this tool will be available from its original source https://github.com/tzutalin/labelImg

# 3. Augmentation
Here i developed my customized image and label augmentation tools from folder "YOLO-Augmentation". This tools will convert YOLO normalized bounding-box format (class, x_center, y_center, width, height) to casual bounding-box format (x1,y1,x2,y2,class), augment the images and labels then return it to YOLO format. 
So before you execute this tool, you better prepare your training images folder and YOLO formatted label and adjust the path to your own image and label ddirectory

# 4. Training
After you've done with 3rd step, go create image file list as instructed on ultralytics repository on https://github.com/ultralytics/yolov3/wiki/Train-Custom-Data by using "create_image_file_list" code by also adjusting your file directory path and follow the rest training and detection instructions from ultralytics repo.
