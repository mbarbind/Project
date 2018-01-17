import os
import numpy as np 
from random import shuffle
import pickle
from keras import backend as bk
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import adam, SGD, RMSprop

# x_train_new.npy & y_train_new.npy are files in which my LFW data is currently stored
xtd = np.load('x_train_new.npy')
ytd = np.load('y_train_new.npy')

num_class = ytd[0].shape
# >> num_class
# out: [7,]
ip_shape = xtd[0].shape
# >> ip_shape
# out: [50, 37, 1]

# play around with this s**t
model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=in_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.4))
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.output_shape
# out: [None, 10, 7, 32]

model.add(Flatten())
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(7))
model.add(Activation('softmax'))

model.output_shape
# out: [None, 7]

model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

# here we don't have validation data, so using first option. If you have it, use second option
hist = model.fit(xtd, ytd, batch_size=32, num_epoch=20, verbose=1, validation_split=0.2)
#or
hist = model.fit(xtd, ytd, batch_size=32, num_epoch=20, verbose=1, validation_data=(x_val,y_val))


#done with the training!
with open('trained_nn.pickle','wb') as f:
	pickle.dump(hist, f)

#for testing, uncomment print statement if you want to check accuracy
score = model.evaluate(xtd[0:100], ytd[0:100], show_accuracy=True, verbose=0)
loss, accuracy = score[0], score[1]
# >> print(loss, accuracy)

#again, i have testing data in 'x_test.npy'
test_images = npy.load('x_test.npy')
for img in test_images:
	print(model.predict(img))
	print(model.predict_classes(img))
	print('\n')






