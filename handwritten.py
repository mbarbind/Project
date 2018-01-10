import math
from tensorflow.examples.tutorials.mnist import input_data
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from datetime import timedelta


#step 1 tuning hyperparameters
#convoluitonal layer1
filter_size1 = 5
num_filters1 = 16

#convoluitonal layer2
filter_size2 = 5
num_filters2 = 36

#fully connected layer
fc_size = 128



#data diamensions
img_size = 28

#image diamension height by width
img_size_flat = img_size * img_size

#number of channels
num_channels = 1

#number of classes
num_classes = 10

#step 2 load data
data = input_data.read_data_sets('data/MNIST/', one_hot = True)


#helper functions
def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):

    #shape is 4-D tensor
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    #weights and biases used in the convoluitonal layer of nn
    weights = new_weights(shape=shape)
    biases = new_biases(length=num_filters)

    #convoluitonal nn function
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1],padding='SAME')

    #add the biases to the output of convoluitonal nn
    layer += biases

    #max_pool reduces the diamensionality of the matrix and easier for training
    if use_pooling:
        layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #apply rectilinear unit for introducing non-linearity and hence nn can learn more complex functions
    layer = tf.nn.relu(layer)

    return layer,weights

def flattern_layer(layer):
    #get shape of input layer
    #layer shape = [num_images, img_height, img_width, num_channels]
    layer_shape = layer.get_shape()

    #number of features = img_height * img_width * num_channels
    num_features = layer_shape[1:4].num_elements()

    #function to reshape the input layer
    layer_flat = tf.reshape(layer, [-1, num_features])

    return layer_flat, num_features

def new_fc_layer(input, num_inputs, num_outputs, use_relu=True):

    #create weights and biases
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    #use o/p=input*weights+biases formula
    layer = tf.matmul(input, weights) + biases

    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# step 3 setting placeholders(inputs and labels)
# input images of shape [num_images, img_size_flat]
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')

# but input to convolutional neural network is 4-D vector
x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])

# true labels associated with the placeholder x
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

# class-number
y_true_cls = tf.argmax(y_true, dimension=1)

# step 4 creating actual neural network pipeline using helper functions
# call method for creating convoluitonal neural network 1 and next line for checking if correct tensor operatioon is created or not
layer_conv1, weights_conv1 = new_conv_layer(input=x_image, num_input_channels=num_channels, filter_size=filter_size1, num_filters=num_filters1, use_pooling=True)
layer_conv1

# call method for creating convoluitonal neural network 2
layer_conv2, weights_conv2 = new_conv_layer(input=layer_conv1, num_input_channels=num_filters1, filter_size=filter_size2, num_filters=num_filters2, use_pooling=True)
layer_conv2

# flattern_layer used to convert 4-D tensor which is the output of convoluitonal neural network to 2-D tensor
layer_flat, num_features = flattern_layer(layer_conv2)
layer_flat
num_features

# fully connected layer 1
layer_fc1 = new_fc_layer(input=layer_flat, num_inputs=num_features, num_outputs=fc_size, use_relu=True)
layer_fc1

# fully connected layer 2
layer_fc2 = new_fc_layer(input=layer_fc1, num_inputs=fc_size, num_outputs=num_classes, use_relu=False)
layer_fc2

# step 5 prediction
# prediction of class
y_pred = tf.nn.softmax(layer_fc2)

# class-number is index of the largest element
y_pred_cls = tf.argmax(y_pred, dimension=1)

# step 6 error calculation
# output of the layer_fc2 is directly used as input of the softmax cross_entropy function since it performs softmax internally and y_pred has applied the softmax already
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2, labels=y_true)

# calculate avg of cross_entropy for all the image classifications
cost = tf.reduce_mean(cross_entropy)

# step 7 optimization to reduce the error
# AdamOptimizer is advanced form of gradient descent optimization
optimizer = tf.train.AdamOptimizer(learning_rate=1e-4,).minimize(cost)

# step 8 performance measure
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# step 9 session creation
session = tf.Session()
session.run(tf.global_variables_initializer())


#helper functions for optimization iterations
train_batch_size = 64

total_iterations = 0
def optimize(num_iterations):
    global total_iterations

    start_time = time.time()

    for i in range(total_iterations, total_iterations + num_iterations):
        #get batch of training examples
        #x_batch holds a batch of images
        #y_true_batch holds the true labels for those images
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)

        #put batch in dictionary
        feed_dict_train = {x: x_batch, y_true: y_true_batch}

        session.run(optimizer, feed_dict=feed_dict_train)

        if i%100 == 0:
            #calculate accuracy over training-set
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            msg = "Optimization Iteration: {0:>6}, Training Accuracy: {1:>6.1%}"

            # Print it.
            print(msg.format(i + 1, acc))

    total_iterations +=num_iterations
    end_time = time.time()
    time_dif = end_time - start_time

    print("Time usage: "+ str(timedelta(seconds = int(round(time_dif)))))


#split the test set into smaller batches of the size 256
test_batch_size = 256

def print_test_accuracy(show_examples_errors=False,show_confusion_matrix=False):
    #number of test examples
    num_test = len(data.test.images)

    #array of the predicted class for each exmples
    cls_pred = np.zeros(shape = num_test, dtype = np.int)

    i=0
    while i < num_test:
        #holds the start of next iteration
        j=min(i + test_batch_size, num_test)


        images = data.test.images[i:j, :]
        labels = data.test.labels[i:j, :]

        feed_dict = {x: images, y_true: labels}

        cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        i=j

    cls_true = np.argmax(data.test.labels, axis =1)
    correct = (cls_true==cls_pred)

    correct_sum = correct.sum()

    acc = float(correct_sum)/num_test

    msg = "Accuracy on test-set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum,num_test))


#performance before any optimization
print_test_accuracy()

#performance after 1 optimization
optimize(num_iterations=1)
print_test_accuracy()

#performance after 100 optimizations
optimize(num_iterations=100)
print_test_accuracy()

#performance after 1000 optimizations
optimize(num_iterations=900)
print_test_accuracy()

#performance after 10000 optimizations
optimize(num_iterations=9000)
print_test_accuracy()

optimize(num_iterations=45000)
print_test_accuracy()
