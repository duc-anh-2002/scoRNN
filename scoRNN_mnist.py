'''
A Recurrent Neural Network (RNN) implementation using TensorFlow library.  Can
be used to run the scoRNN architecture and an LSTM.  Uses the MNIST database of 
handwritten digits (http://yann.lecun.com/exdb/mnist/)
'''


# Import modules
from __future__ import print_function
import tensorflow as tf
import numpy as np
import matplotlib.pylab as plt
import sys
from tensorflow.keras.datasets import mnist
from scoRNN import *
tf.compat.v1.disable_eager_execution()
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

'''
To classify images using a recurrent neural network, we consider every image
as a sequence of single pixels. Because MNIST image shape is 28*28px, we will 
then handle 784 steps of single pixels for every sample.
'''

# Network parameters
model = 'scoRNN'
n_input = 1             # MNIST data input (single pixel at a time)
n_steps = 784           # Number of timesteps (img shape: 28*28)
n_hidden = 170          # Hidden layer size
n_neg_ones = 17         # No. of -1's to put on diagonal of scaling matrix 
n_classes = 10          # MNIST total classes (0-9 digits)
permuteflag = False     # Used for permuted MNIST
training_epochs = 70
batch_size = 50


# Input/Output parameters
in_out_optimizer = 'rmsprop'
in_out_lr = 1e-3


# Hidden to hidden parameters
A_optimizer = 'rmsprop'
A_lr = 1e-4


# COMMAND LINE ARGS: MODEL HIDDENSIZE IO-OPT IO-LR AOPT ALR NEG-ONES PFLAG
try:
    model = sys.argv[1]
    n_hidden = int(sys.argv[2])
    in_out_optimizer = sys.argv[3]
    in_out_lr = float(sys.argv[4])
    A_optimizer = sys.argv[5]
    A_lr = float(sys.argv[6])
    n_neg_ones = int(sys.argv[7])
    permuteflag = sys.argv[8]
    if permuteflag == 'True':
        permuteflag = True
    else:
        permuteflag = False
except IndexError:
    pass

# Setting the random seed
tf.random.set_seed(5544)
np.random.seed(5544)

# MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape and normalize the data
x_train = x_train.reshape((-1, 28, 28)) / 255.0
x_test = x_test.reshape((-1, 28, 28)) / 255.0

# Convert labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Split test data into 10 parts
test_data = np.split(x_test, 10)
test_label = np.split(y_test, 10)


# Name of save string/scaling matrix
if model == 'LSTM':
    savestring = '{:s}_{:d}_{:s}_{:.1e}_permute_{:s}'.format(model, n_hidden, \
                 in_out_optimizer, in_out_lr, str(permuteflag))
if model == 'scoRNN':
    savestring = '{:s}_{:d}_{:d}_{:s}_{:.1e}_{:s}_{:.1e}_permute_{:s}'.format(model, \
                 n_hidden, n_neg_ones, in_out_optimizer, in_out_lr, \
                 A_optimizer, A_lr, str(permuteflag))
    D = np.diag(np.concatenate([np.ones(n_hidden - n_neg_ones), \
        -np.ones(n_neg_ones)]))
print('\n')
print(savestring)
print('\n')


# Creating fixed permutation, if applicable
if permuteflag:
    permute = np.random.RandomState(92916)
    xpermutation = permute.permutation(784)   


# Defining RNN architecture
def RNN(x):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)
    
    
    # Permuting batch_size and n_steps
    x = tf.transpose(x, [1, 0, 2])
   
    # Reshaping to (n_steps*batch_size, n_input)
    x = tf.reshape(x, [-1, n_input])
    
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    x = tf.split(x, n_steps, 0)
    
    
    # Create RNN cell
    if model == 'LSTM':
        rnn_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden, activation=tf.nn.tanh)
    
    if model == 'scoRNN':
        rnn_cell = scoRNNCell(n_hidden, D = D)
    
    # Place RNN cell into RNN, take last timestep as output    
    # outputs, states = tf.contrib.rnn.static_rnn(rnn_cell, x, dtype=tf.float32)
    outputs, states = tf.compat.v1.nn.static_rnn(rnn_cell, x, dtype=tf.float32)
    rnnoutput = outputs[-1]
    
        
    # Last layer, linear
    # output = tf.layers.Dense(inputs=rnnoutput, units=n_classes, activation=None)
    output = tf.keras.layers.Dense(units=n_classes, activation=None)(rnnoutput)
    return output



# Used to calculate Cayley Transform derivative
def Cayley_Transform_Deriv(grads, A, W):
    
    # Calculate update matrix
    I = np.identity(grads.shape[0])
    Update = np.linalg.lstsq((I + A).T, np.dot(grads, D + W.T))[0]
    DFA = Update.T - Update
    
    return DFA


# Used to make the hidden to hidden weight matrix
def makeW(A):
    # Computing hidden to hidden matrix using the relation 
    # W = (I + A)^-1*(I - A)D
    
    I = np.identity(A.shape[0])
    W = np.dot(np.linalg.lstsq(I+A, I-A)[0],D)  

    return W


# Plotting loss & accuracy
def graphlosses(xax, tr_loss, te_loss, tr_acc, te_acc):
       
    plt.subplot(2,1,1)
    plt.plot(xax, tr_loss, label='training loss')
    plt.plot(xax, te_loss, label='testing loss')
    plt.ylim([0,3])
    plt.legend(loc='lower left', prop={'size':6})
    plt.subplot(2,1,2)
    plt.plot(xax, tr_acc, label='training acc')
    plt.plot(xax, te_acc, label='testing acc')
    plt.ylim([0,1])
    plt.legend(loc='lower left', prop={'size':6})
    plt.savefig(savestring + '.png')
    plt.clf()
    
    return


# Graph input
# x = tf.Variable(tf.zeros([None, n_steps, n_input]), dtype=tf.float32)
x = tf.compat.v1.placeholder("float", [None, n_steps, n_input])
y = tf.compat.v1.placeholder("float", [None, n_classes])
# x = tf.keras.Input(shape=(n_steps, n_input))
print('x = ', x)
# y = tf.keras.Input(shape=(n_classes,))
print('y = ', y)
# Assigning to RNN function
pred = RNN(x) 

    
# Define loss object
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, \
#       labels=y))
class SoftmaxCrossEntropy(tf.keras.layers.Layer):
    def __init__(self):
        super(SoftmaxCrossEntropy, self).__init__()
    def call(self, pred, labels):
        return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=labels))

cost = SoftmaxCrossEntropy()(pred, y)
    
# Evaluate model
# correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))

# New code
class Accuracy(tf.keras.layers.Layer):
    def __init__(self):                
        super(Accuracy, self).__init__()
    def call(self, pred, labels):
        return tf.equal(tf.argmax(pred, 1), tf.argmax(labels, 1))

correct_pred = Accuracy()(pred, y)

# Define a custom Keras layer for casting and averaging
class CastAndAverage(tf.keras.layers.Layer):
    def __init__(self):
        super(CastAndAverage, self).__init__()

    def call(self, inputs):
        casted_inputs = tf.cast(inputs, tf.float32)
        return tf.reduce_mean(casted_inputs)

# Use the custom layer
accuracy = CastAndAverage()(correct_pred)

# Optimizers/Gradients
optimizer_dict = {'adam' : tf.keras.optimizers.Adam,
                  'adagrad' : tf.keras.optimizers.Adagrad,
                  'rmsprop' : tf.keras.optimizers.RMSprop,
                  'sgd' : tf.keras.optimizers.SGD}

opt1 = optimizer_dict[in_out_optimizer](learning_rate=in_out_lr)


# LSTM training operations
if model == 'LSTM':
    LSTMtrain = opt1.minimize(cost)

# scoRNN training operations
if model == 'scoRNN':
    opt2 = optimizer_dict[A_optimizer](learning_rate=A_lr)
    # trainable_vars = model.trainable_variables
    # Wvar = [v for v in trainable_vars if 'W:0' in v.name][0]
    # Avar = [v for v in trainable_vars if 'A:0' in v.name][0]
    
    # for v in tf.compat.v1.trainable_variables():
    #     print(v.name)
        
    Wvar = [v for v in tf.compat.v1.trainable_variables() if 'W:0' in v.name][0]
    Avar = [v for v in tf.compat.v1.trainable_variables() if 'A:0' in v.name][0]
    othervarlist = [v for v in tf.compat.v1.trainable_variables() if v not in \
                   [Wvar, Avar]]
    
    # Getting gradients
    grads = tf.gradients(cost, othervarlist + [Wvar])
  
    # Applying gradients to input-output weights
    with tf.control_dependencies(grads):
        applygrad1 = opt1.apply_gradients(zip(grads[:len(othervarlist)], \
                     othervarlist))  
    
    # Updating variables
    newW = tf.compat.v1.placeholder(tf.float32, Wvar.get_shape())
    updateW = tf.compat.v1.assign(Wvar, newW)
    # newW = tf.Variable(tf.zeros_like(Wvar), trainable=False)
    # gradA = tf.Variable(tf.zeros_like(Avar), trainable=False)
    gradA = tf.compat.v1.placeholder(tf.float32, Avar.get_shape())
    # Applying hidden-to-hidden gradients
    applygradA = opt2.apply_gradients([(gradA, Avar)])
    
    # applygradA = opt2.apply_gradients([(gradA, Avar)])

# Plotting lists
epochs_plt = []
train_loss_plt = []
test_loss_plt = []
train_accuracy_plt = []
test_accuracy_plt = []


# Training
with tf.Session() as sess:
    
    # Initializing the variables
    init = tf.global_variables_initializer()
    sess.run(init)
    
    # Get initial A and W
    if model == 'scoRNN':
        A, W = sess.run([Avar, Wvar])

    
    # Keep training until reach number of epochs
    epoch = 1
    while epoch <= training_epochs:
        step = 1
        # Keep training until reach max iterations
        while step * batch_size <= mnist.train.images.shape[0]:
        
            # Getting input data
            batch_x, batch_y = mnist.train.next_batch(batch_size)
            # Reshape data to get 784 seq of 1 pixel
            batch_x = batch_x.reshape((batch_size, n_steps, n_input))
               
            # Updating weights
            if model == 'LSTM':
                sess.run(LSTMtrain, feed_dict={x: batch_x, y: batch_y})         


            if model == 'scoRNN':
                _, hidden_grads = sess.run([applygrad1, grads[-1]], \
                                  feed_dict = {x: batch_x, y: batch_y})
                DFA = Cayley_Transform_Deriv(hidden_grads, A, W)            
                sess.run(applygradA, feed_dict = {gradA: DFA})
                A = sess.run(Avar)
                W = makeW(A)
                sess.run(updateW, feed_dict = {newW: W})

            step += 1

        # Evaluating average epoch accuracy/loss of model
        test_acc, test_loss = map(list, zip(*[sess.run([accuracy, cost], \
                   feed_dict={x: tbatch, y: tlabel}) \
                   for tbatch, tlabel in zip(test_data, test_label)]))
        test_acc, test_loss = np.mean(test_acc), np.mean(test_loss)
 

        # Evaluating training accuracy/loss of model on random training batch               
        train_index = np.random.randint(0, \
                      mnist.train.images.shape[0]//batch_size + 1)
        train_x = mnist.train.images[train_index:train_index + batch_size,:]
        train_y = mnist.train.labels[train_index:train_index + batch_size,:]
        train_x = train_x.reshape((batch_size, n_steps, n_input))
        train_acc, train_loss = sess.run([accuracy, cost], \
                                feed_dict={x: train_x, y: train_y})
                                
                        
        # Printing results
        print('\n')
        print("Completed Epoch: ", epoch)
        print("Testing Accuracy:", test_acc)
        print("Testing Loss:", test_loss)
        print("Training Accuracy:", train_acc)
        print("Training Loss:", train_loss)
        print('\n')
        
        # Plotting results
        epochs_plt.append(epoch)
        train_loss_plt.append(train_loss)
        test_loss_plt.append(test_loss)
        train_accuracy_plt.append(train_acc)
        test_accuracy_plt.append(test_acc)
               
        graphlosses(epochs_plt, train_loss_plt, test_loss_plt, \
                    train_accuracy_plt, test_accuracy_plt)
        
        # Saving files
        np.savetxt(savestring + '_train_loss.csv', \
                   train_loss_plt, delimiter = ',')
        np.savetxt(savestring + '_test_loss.csv', \
                   test_loss_plt, delimiter = ',')
        np.savetxt(savestring + '_train_acc.csv', \
                   train_accuracy_plt, delimiter = ',')
        np.savetxt(savestring + '_test_acc.csv', \
                test_accuracy_plt, delimiter = ',')
                   
        epoch += 1
    
    print("Optimization Finished!")        
