import os
import numpy as np 
from random import shuffle
import pickle
from keras import backend as bk 
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#preprocessing part 2 - data modelling
#dataset - caltech faces

IMG_ROWS = 480
IMG_COLS = 320

#optional, if pickle used in part 1
'''with open('imgdata.pickle','rb') as f:
	img_data = pickle.load(f)'''


#here tensorflow backend and single channel is used
#change AXIS to 1 for theano
AXIS = 4
img_data = np.expand_dims(img_data, axis=AXIS)
print(img_data.shape)

NUM_CLASSES = 11
NUM_SAMPLES = 170
labels = np.ones((NUM_SAMPLES,), dtype='int64')

labels[0:20] = 1
#and so on

names = ['adam', 'amy', 'bella', 'ben', 'chloe', 'chris', 'dan', 'diana', 'evan', 'felix', 'george']

#pickle the labels

#convert to one-hot encoding
one_hot = np_utils.to_categorical(labels, NUM_CLASSES)

#OR
one_hot = np.zeros((170, 11))
one_hot[np.arange(170), a] = 1

#combine this data and shuffle it
x, y = shuffle(img_data, one_hot, random_state=2)
x_train, y_train, x_test, y_test = train_test_split(x, y, test_size=0.2, random_state=4)

#the complete training and testing models are:
training_comp = zip(x_train, y_train)
training_comp = list(training_comp)

testing_comp = zip(x_test, y_test)
testing_comp = list(testing_comp)

#pickle this

#this is the complete preprocessing model