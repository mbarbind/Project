import os
import cv2
import numpy as np 
from random import shuffle

IMG_ROWS = 480
IMG_COLS = 320
DATA_DIR = "C:\Users\Rahul Gore\Documents\Sem7\project\codes\dataset"
NEW_DATA_DIR = "C:\Users\Rahul Gore\Documents\Sem7\project\codes\new_set"

list_data = os.listdir(DATA_DIR)
for img in list_data:
	in_img = cv2.imread(DATA_DIR+"\\"+img)
    in_img = cv2.cvtColor(in_img, cv2.COLOR_BGR2GRAY)
    op_img = cv2.resize(in_img, (480,320))
    img_name = r"\img"+str(i)+r".jpg"
    img_path = os.path.join(NEW_DATA_DIR,img_name)
    cv2.imwrite(img_path, op_img)
	i = i+1


