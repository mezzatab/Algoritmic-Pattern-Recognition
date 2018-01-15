import quandl as qn
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model
from keras import backend as K
from keras.callbacks import TensorBoard
#from sliding_window import sliding_window  ## Defined functions ## 
import warnings
warnings.filterwarnings("ignore")

def AutoEncoder(nCompanies, nDays, encDim): ## This function use time series of data and turn them into autoencoders ##
    # this is the size of our encoded representations
    encoding_dim = encDim  # 32 floats -> compression of factor 24.5, assuming the input is 784 floats

    # this is our input placeholder
    (x,y)=(nDays, nCompanies)
    inputSize=x*y
    input_img = Input(shape=(x*y,))
    #print input_img
    # "encoded" is the encoded representation of the input
    encoded = Dense(encoding_dim, activation='relu' )(input_img)
    #print encoded.shape[1:]
    decoded = Dense(inputSize, activation='tanh')(encoded)
    ## Start compiling CNN autoencoders 
    autoencoder = Model(input_img, decoded)
    encoder=Model(input_img, encoded)
#    decoder=Model(encoded, decoded)
    L=len(autoencoder.layers)
    autoencoder.compile(optimizer='adam', loss='mean_squared_error')#, metrics=['accuracy'])
    return (autoencoder,encoder) 
