# -*- coding: utf-8 -*-
"""
Created on Fri Oct 25 15:25:25 2019

@author: talha
"""

import warnings
import tensorflow as tf
import keras as keras
from keras import backend as K
from keras.layers import Layer, InputSpec
from keras import activations, regularizers, constraints
from keras import initializers
from keras.initializers import constant
from keras import layers
from keras.preprocessing import sequence
from keras.layers import Dense
from keras.datasets import imdb
from keras.models import Sequential
from keras.callbacks import Callback
import numpy as np
import numpy

class SimpleFocusedRNNCell(Layer):

    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer=None,
                 recurrent_initializer=None,
                 bias_initializer='zeros',
                 init_mu_current = 'spread',
                 init_sigma_current=0.,
                 init_mu_prev = 'spread',
                 init_sigma_prev=0.,
                 gain=1.0,
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
        self.gain = gain

    def build(self, input_shape):
        
        print("input_shape",input_shape)
        self.input_dim = input_shape[-1]
        print("self.input_dim= ",self.input_dim)
        
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
        
        w_init_currennt = self.weight_initializer_fw_bg_current
        w_init_prev =  self.weight_initializer_fw_bg_prev
        #w_init_currennt = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_fw_bg_current
        #w_init_prev = initializers.get(self.kernel_initializer) if self.kernel_initializer else self.weight_initializer_fw_bg_prev
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      name='kernel',
                                      initializer=w_init_currennt,
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint,
                                      trainable=self.train_weights)
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            name='recurrent_kernel',
            initializer=w_init_prev,
            regularizer=self.recurrent_regularizer,
            constraint=self.recurrent_constraint,
            trainable=self.train_weights)
        
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
        sigma = K.clip(self.sigma_current,self.MIN_SI,self.MAX_SI)
        #print("sigma= ",sigma)
        
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))

            #if verbose:
             #   kernel= K.eval(result)
              #  print("RESULT after NORMED max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            #
        return K.transpose(result)
    
    def calc_U_prev(self,verbose=False):
        """
        function calculates focus coefficients. 
        normalizes and prunes if
        """
        up= (self.idxs_prev - K.expand_dims(self.mu_prev,1))**2
        sigma = K.clip(self.sigma_prev,self.MIN_SI,self.MAX_SI)
        #cov_scaler = self.cov_scaler
        dwn = K.expand_dims(2 * ( sigma ** 2), axis=1)
        #scaler = (np.pi*self.cov_scaler**2) * (self.idxs.shape[0])
        #print("down shape :",dwn.shape)
        result = K.exp(-up / dwn)
        
        if self.normed==1:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
        
        elif self.normed==2:
            result /= K.sqrt(K.sum(K.square(result), axis=-1,keepdims=True))
            result *= K.sqrt(K.constant(self.input_dim))
        return K.transpose(result)
    
    def weight_initializer_current(self,shape):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U_current())
        W = np.zeros(shape=shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        
        for c in range(W.shape[1]):
            fan_in = np.sum((kernel[:,c])**2)
            
            #fan_in *= self.input_channels no need for this in repeated U. 
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
            
            std = np.float32(std)
            if c == 0 and verbose:
                print("Std here: ",std, type(std),W.shape[0],
                      " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
            if distribution == 'uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:-1])
            elif distribution == 'normal':
                std = std/ np.float32(.87962566103423978)           
                w_vec = np.random.normal(scale=std, size=W.shape[0])
                
            W[:,c] = w_vec.astype('float32')
            
        return W
    
    def weight_initializer_prev(self,shape):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'uniform'
        
        kernel = K.eval(self.calc_U_prev())
        W = np.zeros(shape=shape, dtype='float32')
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        
        for c in range(W.shape[1]):
            fan_in = np.sum((kernel[:,c])**2)
            
            #fan_in *= self.input_channels no need for this in repeated U. 
            if initer == 'He':
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
            else:
                std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
            
            std = np.float32(std)
            if c == 0 and verbose:
                print("Std here: ",std, type(std),W.shape[0],
                      " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
            if distribution == 'uniform':
                std = std * sqrt32(3.0)
                std = np.float32(std)
                w_vec = np.random.uniform(low=-std, high=std, size=W.shape[:-1])
            elif distribution == 'normal':
                std = std/ np.float32(.87962566103423978)           
                w_vec = np.random.normal(scale=std, size=W.shape[0])
                
            W[:,c] = w_vec.astype('float32')
            
        return W
    
    def weight_initializer_fw_bg_prev(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'normal'
        print("in")
        kernel = K.eval(self.calc_U_prev())
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
                    print(r,",",c," Fan in ", fan_in, " Fan_out:", fan_out, W[r,c])
                    
                if distribution == 'uniform':
                    std = std * sqrt32(3.0)
                    std = np.float32(std)
                    w_vec = np.random.uniform(low=-std, high=std, size=1)
                elif distribution == 'normal':
                    std = std/ np.float32(.87962566103423978)           
                    w_vec = np.random.normal(scale=std, size=1)
                    
                W[r,c] = w_vec.astype('float32')
                
        return W
    
    def weight_initializer_fw_bg_current(self,shape, dtype='float32'):
        #only implements channel last and HE uniform
        initer = 'He'
        distribution = 'normal'
        
        kernel = K.eval(self.calc_U_current())
        
        W = np.zeros(shape=shape, dtype=dtype)
        # for Each Gaussian initialize a new set of weights
        verbose=self.verbose
        if verbose:
            print("Kernel max, mean, min: ", np.max(kernel), np.mean(kernel), np.min(kernel))
            print("kernel shape:", kernel.shape, ", W shape: ",W.shape)
        
        fan_out = self.units
        sum_over_domain = np.sum(kernel**2,axis=1) # r base
        sum_over_neuron = np.sum(kernel**2,axis=0)
        for c in range(W.shape[1]):
            for r in range(W.shape[0]):
                fan_out = sum_over_domain[r]
                fan_in = sum_over_neuron[c]
                
                #fan_in *= self.input_channels no need for this in repeated U. 
                if initer == 'He':
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in)
                else:
                    std = self.gain * sqrt32(2.0) / sqrt32(fan_in+fan_out)
                
                std = np.float32(std)
                if c == 0 and verbose:
                    print("Std here: ",std, type(std),W.shape[0],
                          " fan_in", fan_in, "mx U", np.max(kernel[:,:,:,c]))
                    print(r,",",c," Fan in ", fan_in, " Fan_out:", fan_out, W[r,c])
                    
                if distribution == 'uniform':
                    std = std * sqrt32(3.0)
                    std = np.float32(std)
                    w_vec = np.random.uniform(low=-std, high=std, size=1)
                elif distribution == 'normal':
                    std = std/ np.float32(.87962566103423978)           
                    w_vec = np.random.normal(scale=std, size=1)
                    
                W[r,c] = w_vec.astype('float32')
                
        return W
    

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
    
def sqrt32(x):
        return np.sqrt(x,dtype='float32')    
    
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

from keras.layers import RNN

class SimpleFocusedRNN(RNN):
    
    def __init__(self, units,
                 activation='tanh',
                 use_bias=True,
                 kernel_initializer=None,
                 recurrent_initializer='orthogonal',#orthogonal
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
                 init_sigma_current=0.,
                 init_sigma_prev=0.,
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
            dropout = 0.2
            recurrent_dropout = 0.2

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
                             recurrent_dropout=recurrent_dropout,
                             init_sigma_current=init_sigma_current,
                             init_sigma_prev=init_sigma_prev)
        super(SimpleFocusedRNN, self).__init__(cell,
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
        return super(SimpleFocusedRNN, self).call(inputs,
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
        base_config = super(SimpleFocusedRNN, self).get_config()
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
    

class PrintLayerVariableStats(Callback):
    def __init__(self,name,var,stat_functions,stat_names,num):
        self.layername = name
        self.varname = var
        self.stat_list = stat_functions
        self.stat_names = stat_names
        self.num=num

    def setVariableName(self,name, var):
        self.layername = name
        self.varname = var
    def on_train_begin(self, logs={}):
        all_params = self.model.get_layer(self.layername)._trainable_weights
        all_weights = self.model.get_layer(self.layername).get_weights()
        #print("self.model",self.model)
        #print("all_params",all_params)
        #print("self.layername",self.layername)
        #print("self.varname",self.varname)
        i=self.num
        if(i == 0):
            stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("\nStats for kernel:0 ", stat_str)
        if(i == 1):
            stat_str_1 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for Sigma_current:0 ", stat_str_1)
        if(i == 2):
            stat_str_2 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for Mu_current:0 ", stat_str_2)
        if(i == 3):
            stat_str_3 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for recurrent_kernel:0 ", stat_str_3)
        
        
        

        #def on_batch_end(self, batch, logs={}):
        #    self.record.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        all_weights = self.model.get_layer(self.layername).get_weights()
        i=self.num
        if(i == 0):
            stat_str = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("\nStats for kernel:0 ", stat_str)
        if(i == 1):
            stat_str_1 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for Sigma_current:0 ", stat_str_1)
        if(i == 2):
            stat_str_2 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for Mu_current:0 ", stat_str_2)
        if(i == 3):
            stat_str_3 = [n+str(s(all_weights[i])) for s,n in zip(self.stat_list,self.stat_names)]
            print("Stats for recurrent_kernel:0 ", stat_str_3)

#from keras_utils import SGDwithLR
'''
mod='focused'
for i in range(0,5):
  numpy.random.seed(7)
  K.clear_session()
  # load the dataset but only keep the top n words, zero the rest
  top_words = 5000
  (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
  # truncate and pad input sequences
  max_review_length = 500
  X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
  X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
  #lr_dict = {'all':0.1}
  #mom_dict = {'all':0.9}           
  #decay_dict = {'all':0.9}
  #e_i=X_train.shape[0]
  #decay_epochs =np.array([e_i*100, e_i*150], dtype='int64')
  #clip_dict ={}
  #opt= SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict, decay_epochs)
  # create the model
  embedding_vecor_length = 32
  model = Sequential()
  model.add(layers.Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
  model.add(SimpleFocusedRNN(100,dropout=0.2,recurrent_dropout=0.2,init_sigma_current=0.02,init_sigma_prev=0.02))#cell  #layer
  model.add(Dense(1, activation='sigmoid'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  print(model.summary())
  stat_func_name = ['max: ', 'mean: ', 'min: ', 'var: ', 'std: ']
  stat_func_list = [np.max, np.mean, np.min, np.var, np.std]
  #callbacks = [tb]
  callbacks = []
  pr_0 = PrintLayerVariableStats("simple_focused_rnn","kernel:0",stat_func_list,stat_func_name,0)
  pr_1 = PrintLayerVariableStats("simple_focused_rnn","Sigma_current:0",stat_func_list,stat_func_name,1)
  pr_2 = PrintLayerVariableStats("simple_focused_rnn","Mu_current:0",stat_func_list,stat_func_name,2)
  pr_3 = PrintLayerVariableStats("simple_focused_rnn","recurrent_kernel:0",stat_func_list,stat_func_name,3)
  #pr_5 = PrintLayerVariableStats("simple_focused_rnn","Sigma_prev:0",stat_func_list,stat_func_name)
  #pr_6 = PrintLayerVariableStats("simple_focused_rnn","Mu_prev:0",stat_func_list,stat_func_name)

  #print("pr_1",pr_1.on_train_begin)
  #print("pr_1",pr_1.on_epoch_end)
  callbacks+=[pr_0,pr_1,pr_2,pr_3]
  #print("callbacks",callbacks)
  model.fit(X_train, y_train, epochs=15, batch_size=64,verbose=1,callbacks=callbacks)
  # Final evaluation of the model
  scores = model.evaluate(X_test, y_test, verbose=0)
  print("Accuracy: %.2f%%" % (scores[1]*100))
'''


