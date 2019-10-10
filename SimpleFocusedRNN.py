# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 17:29:32 2019

@author: talha
"""
import warnings
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import activations, regularizers, constraints
from tensorflow.keras import initializers
from tensorflow.keras.initializers import constant
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.layers import Dense
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
import numpy as np
import numpy

class SimpleFocusedRNNCell(Layer):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 init_mu_current = 'spread',
                 init_sigma_current=0.1,
                 init_mu_prev = 'spread',
                 init_sigma_prev=0.1,
                 verbose=False,
                 si_regularizer=None,
                 train_mu=True,train_sigma=True, 
                 train_weights=True,
                 normed=2,
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 **kwargs):
        super(SimpleFocusedRNNCell, self).__init__(**kwargs)
        self.units = units
        self.activation = activations.get(activation)
        self.use_bias = use_bias

        self.kernel_initializer = initializers.get(kernel_initializer)
        self.recurrent_initializer = initializers.get(recurrent_initializer)
        self.bias_initializer = initializers.get(bias_initializer)

        self.kernel_regularizer = regularizers.get(kernel_regularizer)
        self.recurrent_regularizer = regularizers.get(recurrent_regularizer)
        self.bias_regularizer = regularizers.get(bias_regularizer)

        self.kernel_constraint = constraints.get(kernel_constraint)
        self.recurrent_constraint = constraints.get(recurrent_constraint)
        self.bias_constraint = constraints.get(bias_constraint)

        self.dropout = min(1., max(0., dropout))
        self.recurrent_dropout = min(1., max(0., recurrent_dropout))
        self.state_size = self.units
        self.output_size = self.units
        self._dropout_mask = None
        self._recurrent_dropout_mask = None
        self.si_regularizer = regularizers.get(si_regularizer)
        self.init_mu_current=init_mu_current
        self.init_mu_prev=init_mu_prev
        self.init_sigma_current=init_sigma_current
        self.init_sigma_prev=init_sigma_prev
        self.verbose = verbose
        self.input_spec = InputSpec(min_ndim=2)
        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.train_weights = train_weights
        self.normed = normed

    def build(self, input_shape):
        
        self.input_dim = input_shape[-1]
        
        self.input_spec = InputSpec(min_ndim=2, axes={-1: self.input_dim}) #never used for another layer

        mu_curent, si_current = mu_si_initializer(self.init_mu_current, self.init_sigma_current, self.input_dim,
                                   self.units, verbose=self.verbose)
        #print("mu_curent",mu_curent)
        
        self.mu_current = self.add_weight(shape=(self.units,), 
                                  initializer=constant(mu_curent), 
                                  name="Mu_current", 
                                  trainable=self.train_mu)
        self.sigma_current = self.add_weight(shape=(self.units,), 
                                     initializer=constant(si_current), 
                                     name="Sigma_current", 
                                     regularizer=self.si_regularizer,
                                     trainable=self.train_sigma)
        
        mu_prev, si_prev = mu_si_initializer(self.init_mu_prev, self.init_sigma_prev, self.input_dim,
                                   self.units, verbose=self.verbose)
        
        #print("mu_prev",mu_prev)
        
        self.mu_prev = self.add_weight(shape=(self.units,), 
                                  initializer=constant(mu_prev), 
                                  name="Mu_prev", 
                                  trainable=self.train_mu)
        self.sigma_prev = self.add_weight(shape=(self.units,), 
                                     initializer=constant(si_prev), 
                                     name="Sigma_prev", 
                                     regularizer=self.si_regularizer,
                                     trainable=self.train_sigma)
        
        idxs_current = np.linspace(0, 1.0,self.input_dim)   #  current previ lazım mı
        idxs_current = idxs_current.astype(dtype='float32')
        
        self.idxs_current = K.constant(value=idxs_current, shape=(self.input_dim,), 
                                   name="idxs_current")
        
        idxs_prev = np.linspace(0, 1.0,self.units)   #  current previ lazım mı
        idxs_prev = idxs_prev.astype(dtype='float32')
        
        self.idxs_prev = K.constant(value=idxs_prev, shape=(self.units,), #outputa göre 
                                   name="idxs_prev")
        
        MIN_SI = 0.01  # zero or below si will crashed calc_u
        MAX_SI = 1.0 
        
        # create shared vars.
        self.MIN_SI = np.float32(MIN_SI)#, dtype='float32')
        self.MAX_SI = np.float32(MAX_SI)#, dtype='float32')
        
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=self.kernel_initializer,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=self.recurrent_initializer,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint)
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        name='bias',
                                        initializer=self.bias_initializer,
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, states, training=None):
        u_current = self.calc_U_current()
        u_previous = self.calc_U_prev()
        print("u_current",u_current)
        print("u_previous",u_previous)
        prev_output = states[0]
        if 0 < self.dropout < 1 and self._dropout_mask is None:
            self._dropout_mask = _generate_dropout_mask(
                K.ones_like(inputs),
                self.dropout,
                training=training)
        if (0 < self.recurrent_dropout < 1 and
                self._recurrent_dropout_mask is None):
            self._recurrent_dropout_mask = _generate_dropout_mask(
                K.ones_like(prev_output),
                self.recurrent_dropout,
                training=training)

        dp_mask = self._dropout_mask
        rec_dp_mask = self._recurrent_dropout_mask
        
        kernel_current = self.kernel*u_current #u'lar la çarğınca sapıtıyo
        kernel_previous = self.recurrent_kernel*u_previous

        if dp_mask is not None:
            h = K.dot(inputs * dp_mask, kernel_current)
        else:
            h = K.dot(inputs, kernel_current)
        if self.bias is not None:
            h = K.bias_add(h, self.bias)

        if rec_dp_mask is not None:
            prev_output *= rec_dp_mask
        output = h + K.dot(prev_output, kernel_previous)
        if self.activation is not None:
            output = self.activation(output)

        # Properly set learning phase on output tensor.
        if 0 < self.dropout + self.recurrent_dropout:
            if training is None:
                output._uses_learning_phase = True
        return output, [output]
    
    def calc_U_current(self,verbose=False):
        """
        function calculates focus coefficients. 
        normalizes and prunes if
        """
        up= (self.idxs_current - K.expand_dims(self.mu_current,1))**2
        #print("up =",K.eval(up))
        #print("up.shape", up.shape)
        #up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.sigma_current,self.MIN_SI,self.MAX_SI)
        #print("sigma= ",sigma)
        
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        #print("result= ",K.eval(result))
        #kernel= K.eval(result)
        #print("RESULT max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
  
        
        
      
        #print("U shape :",result.shape)
        #print("inputs shape",inputs.shape)

        #sum normalization each filter has sum 1
        #sums = K.sum(masks**2, axis=(0, 1), keepdims=True)
        #print(sums)
        #gain = K.constant(self.gain, dtype='float32')
        
        #Normalize to 1
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))

            #if verbose:
             #   kernel= K.eval(result)
              #  print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            #
        #Normalize to get equal to WxW Filter
        #masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
        # make norm sqrt(filterw x filterh x self.incoming_channel)
        # the reason for this is if you take U all ones(self.kernel_size[0],kernel_size[1], num_channels)
        # its norm will sqrt(wxhxc)
        #print("Vars: ",self.input_channels,self.kernel_size[0],self.kernel_size[1])
        
        #print("out")
        return K.transpose(result)
    
    def calc_U_prev(self,verbose=False):
        """
        function calculates focus coefficients. 
        normalizes and prunes if
        """
        up= (self.idxs_prev - K.expand_dims(self.mu_prev,1))**2
        #print("up.shape", up.shape)
        #up = K.expand_dims(up,axis=1,)
        #print("up.shape",up.shape)
        # clipping scaler in range to prevent div by 0 or negative cov. 
        sigma = K.clip(self.sigma_prev,self.MIN_SI,self.MAX_SI)
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        #kernel= K.eval(result)
        #print("RESULT max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
  
        
        
      
        #print("U shape :",result.shape)
        #print("inputs shape",inputs.shape)

        #sum normalization each filter has sum 1
        #sums = K.sum(masks**2, axis=(0, 1), keepdims=True)
        #print(sums)
        #gain = K.constant(self.gain, dtype='float32')
        
        #Normalize to 1
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))

            #if verbose:
                #kernel= K.eval(result)
                #print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            #
        #Normalize to get equal to WxW Filter
        #masks *= K.sqrt(K.constant(self.input_channels*self.kernel_size[0]*self.kernel_size[1]))
        # make norm sqrt(filterw x filterh x self.incoming_channel)
        # the reason for this is if you take U all ones(self.kernel_size[0],kernel_size[1], num_channels)
        # its norm will sqrt(wxhxc)
        #print("Vars: ",self.input_channels,self.kernel_size[0],self.kernel_size[1])
        
        #print("out prev")
        return K.transpose(result)

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleFocusedRNNCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def _generate_dropout_mask(ones, rate, training=None, count=1):
    def dropped_inputs():
        return K.dropout(ones, rate)

    if count > 1:
        return [K.in_train_phase(
            dropped_inputs,
            ones,
            training=training) for _ in range(count)]
    return K.in_train_phase(
        dropped_inputs,
        ones,
        training=training)  
    
    
def mu_si_initializer(initMu, initSi, num_incoming, num_units, verbose=True):
    '''
    Initialize focus centers and sigmas with regards to initMu, initSi
    
    initMu: a string, a value, or a numpy.array for initialization
    initSi: a string, a value, or a numpy.array for initialization
    num_incoming: number of incoming inputs per neuron
    num_units: number of neurons in this layer
    '''
    
    if isinstance(initMu, str):
        if initMu == 'middle':
            #print(initMu)
            mu = np.repeat(.5, num_units)  # On paper we have this initalization                
        elif initMu =='middle_random':
            mu = np.repeat(.5, num_units)  # On paper we have this initalization
            mu += (np.random.rand(len(mu))-0.5)*(1.0/(float(20.0)))  # On paper we have this initalization                
            
        elif initMu == 'spread':
            mu = np.linspace(0.2, 0.8, num_units)  #paper results were taken with this
            #mu = np.linspace(0.1, 0.9, num_units)
        else:
            print(initMu, "Not Implemented")
            
    elif isinstance(initMu, float):  #initialize it with the given scalar
        mu = np.repeat(initMu, num_units)  # 

    elif isinstance(initMu,np.ndarray):  #initialize it with the given array , must be same length of num_units
        if initMu.max() > 1.0:
            print("Mu must be [0,1.0] Normalizing initial Mu value")
            initMu /=(num_incoming - 1.0)
            mu = initMu        
        else:
            mu = initMu
    
    #Initialize sigma
    if isinstance(initSi,str):
        if initSi == 'random':
            si = np.random.uniform(low=0.05, high=0.25, size=num_units)
        elif initSi == 'spread':
            si = np.repeat((initSi / num_units), num_units)

    elif isinstance(initSi,float):  #initialize it with the given scalar
        si = np.repeat(initSi, num_units)# 
        
    elif isinstance(initSi, np.ndarray):  #initialize it with the given array , must be same length of num_units
        si = initSi
        
    # Convert Types for GPU
    mu = mu.astype(dtype='float32')
    si = si.astype(dtype='float32')

    if verbose:
        print("mu init:", mu)
        print("si init:", si)
  
    return mu, si

from tensorflow.keras.layers import RNN

class SimpleRNN(RNN):
    
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer='glorot_uniform',
                 recurrent_initializer='orthogonal',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 recurrent_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 recurrent_constraint=None,
                 bias_constraint=None,
                 dropout=0.,
                 recurrent_dropout=0.,
                 return_sequences=False,
                 return_state=False,
                 go_backwards=False,
                 stateful=False,
                 unroll=False,
                 **kwargs):
        if 'implementation' in kwargs:
            kwargs.pop('implementation')
            warnings.warn('The `implementation` argument '
                          'in `SimpleRNN` has been deprecated. '
                          'Please remove it from your layer call.')
        if K.backend() == 'theano' and (dropout or recurrent_dropout):
            warnings.warn(
                'RNN dropout is no longer supported with the Theano backend '
                'due to technical limitations. '
                'You can either set `dropout` and `recurrent_dropout` to 0, '
                'or use the TensorFlow backend.')
            dropout = 0.
            recurrent_dropout = 0.

        cell = SimpleFocusedRNNCell(units,
                             activation=activation,
                             use_bias=use_bias,
                             kernel_initializer=kernel_initializer,
                             recurrent_initializer=recurrent_initializer,
                             bias_initializer=bias_initializer,
                             kernel_regularizer=kernel_regularizer,
                             recurrent_regularizer=recurrent_regularizer,
                             bias_regularizer=bias_regularizer,
                             kernel_constraint=kernel_constraint,
                             recurrent_constraint=recurrent_constraint,
                             bias_constraint=bias_constraint,
                             dropout=dropout,
                             recurrent_dropout=recurrent_dropout)
        super(SimpleRNN, self).__init__(cell,
                                        return_sequences=return_sequences,
                                        return_state=return_state,
                                        go_backwards=go_backwards,
                                        stateful=stateful,
                                        unroll=unroll,
                                        **kwargs)
        self.activity_regularizer = regularizers.get(activity_regularizer)

    def call(self, inputs, mask=None, training=None, initial_state=None):
        self.cell._dropout_mask = None
        self.cell._recurrent_dropout_mask = None
        return super(SimpleRNN, self).call(inputs,
                                           mask=mask,
                                           training=training,
                                           initial_state=initial_state)

    @property
    def units(self):
        return self.cell.units

    @property
    def activation(self):
        return self.cell.activation

    @property
    def use_bias(self):
        return self.cell.use_bias

    @property
    def kernel_initializer(self):
        return self.cell.kernel_initializer

    @property
    def recurrent_initializer(self):
        return self.cell.recurrent_initializer

    @property
    def bias_initializer(self):
        return self.cell.bias_initializer

    @property
    def kernel_regularizer(self):
        return self.cell.kernel_regularizer

    @property
    def recurrent_regularizer(self):
        return self.cell.recurrent_regularizer

    @property
    def bias_regularizer(self):
        return self.cell.bias_regularizer

    @property
    def kernel_constraint(self):
        return self.cell.kernel_constraint

    @property
    def recurrent_constraint(self):
        return self.cell.recurrent_constraint

    @property
    def bias_constraint(self):
        return self.cell.bias_constraint

    @property
    def dropout(self):
        return self.cell.dropout

    @property
    def recurrent_dropout(self):
        return self.cell.recurrent_dropout

    def get_config(self):
        config = {'units': self.units,
                  'activation': activations.serialize(self.activation),
                  'use_bias': self.use_bias,
                  'kernel_initializer':
                      initializers.serialize(self.kernel_initializer),
                  'recurrent_initializer':
                      initializers.serialize(self.recurrent_initializer),
                  'bias_initializer': initializers.serialize(self.bias_initializer),
                  'kernel_regularizer':
                      regularizers.serialize(self.kernel_regularizer),
                  'recurrent_regularizer':
                      regularizers.serialize(self.recurrent_regularizer),
                  'bias_regularizer': regularizers.serialize(self.bias_regularizer),
                  'activity_regularizer':
                      regularizers.serialize(self.activity_regularizer),
                  'kernel_constraint': constraints.serialize(self.kernel_constraint),
                  'recurrent_constraint':
                      constraints.serialize(self.recurrent_constraint),
                  'bias_constraint': constraints.serialize(self.bias_constraint),
                  'dropout': self.dropout,
                  'recurrent_dropout': self.recurrent_dropout}
        base_config = super(SimpleRNN, self).get_config()
        del base_config['cell']
        return dict(list(base_config.items()) + list(config.items()))

    @classmethod
    def from_config(cls, config):
        if 'implementation' in config:
            config.pop('implementation')
        return cls(**config)

'''
model = tf.keras.Sequential()
model.add(layers.Embedding(input_dim=1000, output_dim=64))

# The output of GRU will be a 3D tensor of shape (batch_size, timesteps, 256)
model.add(layers.GRU(256, return_sequences=True))

# The output of SimpleRNN will be a 2D tensor of shape (batch_size, 128)
model.add(SimpleRNN(128))

model.add(layers.Dense(10, activation='softmax'))

model.summary() 
'''

numpy.random.seed(7)
# load the dataset but only keep the top n words, zero the rest
top_words = 5000
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# truncate and pad input sequences
max_review_length = 500
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# create the model
embedding_vecor_length = 32
model = Sequential()
model.add(layers.Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
model.add(SimpleRNN(100))#cell  #layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))



