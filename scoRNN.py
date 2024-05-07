from tensorflow.python.ops.rnn_cell_impl import RNNCell 

import tensorflow as tf
import numpy as np

from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import variable_scope as vs



class scoRNNCell(RNNCell):
    """Scaled Cayley Orthogonal Recurrent Network (scoRNN) Cell

    """

    def __init__(self, hidden_size, D = None, activation='modReLU'):
        
        self._hidden_size = hidden_size
        self._activation = activation
        if D is None:
            self._D = np.identity(hidden_size, dtype=np.float32)
        else:
            self._D = D.astype(np.float32)
        
        # Initialization of skew-symmetric matrix
        s = np.random.uniform(0, np.pi/2.0, \
        size=int(np.floor(self._hidden_size/2.0)))
        s = -np.sqrt((1.0 - np.cos(s))/(1.0 + np.cos(s)))
        z = np.zeros(s.size)
        if self._hidden_size % 2 == 0:
            diag = np.hstack(list(zip(s, z)))[:-1]
        else:
            diag = np.hstack(list(zip(s,z)))
        A_init = np.diag(diag, k=1)
        A_init = A_init - A_init.T
        A_init = A_init.astype(np.float32)
        
        self._A = tf.Variable(tf.random.normal([self._hidden_size, self._hidden_size]), name="A", \
                  initializer = init_ops.constant_initializer(A_init))
        
        # Initialization of hidden to hidden matrix
        I = np.identity(hidden_size)
        Z_init = np.linalg.lstsq(I + A_init, I - A_init)[0].astype(np.float32)
        W_init = np.matmul(Z_init, self._D)
        
        self._W = tf.Variable(tf.random.normal([self._hidden_size, self._hidden_size]), name="W", \
                  initializer = init_ops.constant_initializer(W_init))        
	#self._W = tf.Variable(W_init, name="W", dtype=tf.float32)
	
	    # Initialization of bias
        self._bias = tf.Variable(tf.zeros([self._hidden_size]), name="b", \
	    initializer= init_ops.constant_initializer())
   
    @property
    def state_size(self):
        return self._hidden_size

    @property
    def output_size(self):
        return self._hidden_size

    def __call__(self, inputs, state, scope=None):
        with vs.variable_scope(scope or "scoRNNCell"):
            
            # Initialization of input matrix
            U_init = init_ops.random_uniform_initializer(-0.01, 0.01)
            #U = vs.get_variable("U", [inputs.get_shape()[-1], \
            #    self._hidden_size], initializer= U_init)
            U = tf.Variable(U_init(shape=(input_shape, self._hidden_size), dtype=tf.float32), name="U") 

            # Forward pass of graph           
            res = math_ops.matmul(inputs, U) + math_ops.matmul(state, self._W)
            if self._activation == 'modReLU':
                output = tf.nn.relu(nn_ops.bias_add(tf.abs(res), self._bias))\
                *(tf.sign(res))
            else:
                output = self._activation(nn_ops.bias_add(res, self._bias))

        return output, output

