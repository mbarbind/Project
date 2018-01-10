import os
import cv2
import numpy as np 
from random import shuffle

#preprocessing part 1 - greyscale and resizing images

#sample dataset taken here- caltech faces

#change row and cols value
IMG_ROWS = 480
IMG_COLS = 320

#path specific to Rahul Gore, change here
DATA_DIR = r"C:\Users\Rahul Gore\Documents\Sem7\project\codes\dataset"
NEW_DATA_DIR = r"C:\Users\Rahul Gore\Documents\Sem7\project\codes\new_set"

#lists all the images
list_data = os.listdir(DATA_DIR)

#real conversion
for img in list_data:
	in_img = cv2.imread(DATA_DIR+"\\"+img)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    op_img = cv2.resize(in_img, (IMG_ROWS,IMG_COLS))
    img_name = r"\img"+'{:0>3}'.format(str(i))+r".jpg"
    img_path = os.path.join(NEW_DATA_DIR,img_name)
    cv2.imwrite(img_path, op_img)
	i = i+1


#labeling the data, worst case scenario
new_list_data = os.listdir(NEW_DATA_DIR)

#new loop for each label
#here one label is given
#here, 20 images having same label are considered
for img in new_list_data[0:20]:
	label_name = "adams"+img[3:6]+".jpg"
	os.rename(os.path.join(NEW_DATA_DIR,img),os.path.join(NEW_DATA_DIR,label_name))

