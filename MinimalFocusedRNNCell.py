# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 23:59:42 2019

@author: talha
"""
import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras
from tensorflow.keras.layers import RNN
from tensorflow.python.keras.layers import Layer, InputSpec
from tensorflow.keras import activations, regularizers, constraints
from tensorflow.keras import initializers
from tensorflow.keras.models import Sequential
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras.initializers import constant
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.datasets import imdb
import numpy as np
import numpy

class MinimalRNNFocusedCell(keras.layers.Layer):

    def __init__(self, units,
                 activation=None,
                 activity_regularizer=None,
                 init_mu_current = 'spread',
                 init_sigma_current=0.1,
                 init_mu_prev = 'spread',
                 init_sigma_prev=0.1,
                 verbose=True,
                 si_regularizer=None,
                 train_mu=True,train_sigma=True, 
                 train_weights=True,
                 normed=2,
                 init_w=initializers.glorot_uniform(),
                 **kwargs):
        self.units = units
        self.state_size = units
        self.verbose = verbose
        self.si_regularizer = regularizers.get(si_regularizer)
        self.init_mu_current=init_mu_current
        self.init_mu_prev=init_mu_prev
        self.init_sigma_current=init_sigma_current
        self.init_sigma_prev=init_sigma_prev
        self.train_mu = train_mu
        self.train_sigma = train_sigma
        self.train_weights = train_weights
        self.normed = normed
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.input_spec = InputSpec(min_ndim=2)
        self.init_weights =init_w
        self.activation = activations.get(activation)
        super(MinimalRNNFocusedCell, self).__init__(**kwargs)
    

    def build(self, input_shape):
        #assert len(input_shape) >= 2
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
        
        
        print("input_shape= ",input_shape[-1])
        print("units= ",self.units)
        self.W_current = self.add_weight(shape=(input_shape[-1],self.units),
                                      initializer='uniform',
                                      name='W_current')
    
        
        
        self.W_previous = self.add_weight(
            shape=(self.units,self.units),
            initializer='uniform',
            name='W_previous')
        
   
        self.built = True

    def call(self, inputs, states):
        #print("here")
        u_current = self.calc_U_current()
        u_previous = self.calc_U_prev()
        #print(u_current)
        #print(u_previous)
        if self.verbose:
            print("weight_current shape", self.W_current.shape)
            print("weight_prev shape", self.W_previous.shape)
        prev_output = states[0]
        kernel_current = self.W_current*u_current
        kernel_previous = self.W_previous*u_previous
        
        #self.kernel_current = self.W_current*u_current
        #self.kernel_previous = self.W_previous*u_previous
        
        
        h = K.dot(inputs, kernel_current)
        #print("h= ",h)
        output = h + K.dot(prev_output, kernel_previous)  #kernel_prev mi W_previous
        #print("output= ",output)
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
        config = {
            'units': self.units,
            'activation': activations.serialize(self.activation),
            'use_bias': self.use_bias,
            'kernel_initializer': initializers.serialize(self.kernel_initializer),
            'bias_initializer': initializers.serialize(self.bias_initializer),
            'kernel_regularizer': regularizers.serialize(self.kernel_regularizer),
            'bias_regularizer': regularizers.serialize(self.bias_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'kernel_constraint': constraints.serialize(self.kernel_constraint),
            'bias_constraint': constraints.serialize(self.bias_constraint),
            'focusing vars': "not here"
        }
        base_config = super(MinimalRNNFocusedCell, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    
    
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

# Let's use this cell in a RNN layer:

cell = MinimalRNNFocusedCell(32)
x = keras.Input((None, 5))
layer = RNN(cell)
y = layer(x)

# Here's how to use the cell to build a stacked RNN:

cells = [MinimalRNNFocusedCell(32), MinimalRNNFocusedCell(64)]
x = keras.Input((None, 5))
layer = RNN(cells)
y = layer(x)

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
model.add(MinimalRNNFocusedCell(32))#cell  #layer
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=3, batch_size=64)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))


"""
from tensorflow.keras.datasets import boston_housing

(train_data, train_targets), (test_data, test_targets) = boston_housing.load_data()

# take a look at the data

print(f'Training data : {train_data.shape}')
print(f'Test data : {test_data.shape}')
print(f'Training sample : {train_data[0]}')
print(f'Training target sample : {train_targets[0]}')
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

pad = 5
train_data2 = np.zeros(shape=(train_data.shape[0],train_data.shape[1]+2*pad), dtype=train_data.dtype)
test_data2 = np.zeros(shape=(test_data.shape[0],test_data.shape[1]+2*pad), dtype=train_data.dtype)
train_data2[:,pad:-pad]=train_data
test_data2[:,pad:-pad]=test_data

from keras_utils import SGDwithLR, RMSpropwithClip
lr_dict = {'all':0.01,
           'focus-1/Sigma:0': 0.01,'focus-1/Mu:0': 0.01,'focus-1/Weights:0': 0.01,
           'focus-2/Sigma:0': 0.01,'focus-2/Mu:0': 0.01,'focus-2/Weights:0': 0.01,
           'dense_1/Weights:0':0.01}
        
#lr_dict = {'all':0.0001}

mom_dict = {'all':0.9}
#decay_dict = {'all':0.9}
#mom_dict = {'all':0.9,'focus-1/Sigma:0': 0.25,'focus-1/Mu:0': 0.25,
#           'focus-2/Sigma:0': 0.25,'focus-2/Mu:0': 0.25}
    
decay_dict = {'all':0.9, 'focus-1/Sigma:0': 0.1,'focus-1/Mu:0':0.1,
              'focus-2/Sigma:0': 0.1,'focus-2/Mu:0': 0.1}

clip_dict = {'focus-1/Sigma:0':(0.05,1.0),'focus-1/Mu:0':(0.0,1.0),
             'focus-2/Sigma:0':(0.05,1.0),'focus-2/Mu:0':(0.0,1.0)}

def build_model(N=64,mod='dense', optimizer_s='SGDwithLR'):
    
    model = models.Sequential()
    if mod=='dense':
        model.add(layers.Dense(N, activation='relu', input_shape=(train_data.shape[1],),name='dense-1'))
        model.add(layers.Dense(N, activation='relu',name='dense-2'))
    elif mod=='focused':
        model.add(MinimalRNNFocusedCell(units=N,
                                  name='focus-1',
                                  activation='relu',
                                  init_sigma_current=0.25, 
                                  init_mu_current='spread',
                                  init_sigma_prev=0.25,
                                  init_mu_prev='spread',
                                  init_w= None,
                                  train_sigma=True, 
                                  train_weights=True,
                                  train_mu = True,normed=2))
        
        
        model.add(MinimalRNNFocusedCell(units=N,
                                  name='focus-2',
                                  activation='relu',
                                  init_sigma_current=0.25, 
                                  init_mu_current='spread',
                                  init_sigma_prev=0.25,
                                  init_mu_prev='spread',
                                  init_w= None,
                                  train_sigma=True, 
                                  train_weights=True,
                                  train_mu = True,normed=2))
        
    model.add(layers.Dense(1,name='dense-3'))
    

    if optimizer_s == 'SGDwithLR':
        opt = SGDwithLR(lr_dict, mom_dict,decay_dict,clip_dict)#, decay=None)
    elif optimizer_s=='RMSpropwithClip':
        opt = RMSpropwithClip(lr=0.001, rho=0.9, epsilon=None, decay=0.0,clips=clip_dict)
    else:
        opt= SGDwithLR(lr=0.01, momentum=0.9)#, decay=None)  #asıl kodda SGD olarak yazıyordu hatalı mı acaba.
    
    
    
    model.compile(optimizer=opt,
              loss='mse',
              metrics=['mae'])
    return model

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
mod='focused'
N=64
all_scores = []

for i in range(k):
    print(f'Processing fold # {i}')
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i+1) * num_val_samples]
    
    partial_train_data = np.concatenate(
                            [train_data[:i * num_val_samples],
                            train_data[(i+1) * num_val_samples:]],
                            axis=0)
    partial_train_targets = np.concatenate(
                            [train_targets[:i * num_val_samples],
                            train_targets[(i+1)*num_val_samples:]],
                            axis=0)
    model = build_model(N,mod)
    model.fit(partial_train_data,
              partial_train_targets,
              epochs=num_epochs,
              batch_size=1,
              verbose=0)
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)
"""
