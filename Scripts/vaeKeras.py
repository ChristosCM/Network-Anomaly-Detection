import os
import pandas as pd
import numpy as np
import scipy
import tensorflow as tf
from keras.layers import Lambda, Input, Dense, Activation
from keras.models import Model, Sequential, model_from_json
# from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K
import seaborn as sns

import matplotlib.pyplot as plt
import argparse

from numpy.random import seed
from sklearn import preprocessing
from sklearn.preprocessing import minmax_scale
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE

from keras import losses
import sys
def genSample(data):#function to generate sample for the autoencoder
    #firstly move all the sub attack categories in the sample
    df = pd.DataFrame(data[(data.attack_cat!="Generic")&(data.attack_cat!="Exploits")&(data.attack_cat!="Fuzzers")&(data.Label==1)])
    #df.reset_index(inplace=True)
    notNorm = pd.DataFrame(data[(data.attack_cat=="Generic")|(data.attack_cat=="Exploits")|(data.attack_cat=="Fuzzers")]) #hold just normal data and Generic attack categories
    notNorm = notNorm.sample(frac=0.1,random_state=1) #sample the normal data to be put back

    normGen = pd.DataFrame(data[(data.Label==0)])
    normGen = normGen.sample(frac=0.05,random_state=1)
    #normGen.reset_index(inplace=True)
    df = pd.concat([df,normGen,notNorm],ignore_index=True)
    df.dropna(inplace=True)
    
    return df


df = pd.read_csv("all_clean_data.csv")

df = genSample(df) #split now is 60(norm)-40(abnorm) with around 175,000 data points (still a lot for PC)
sam = df.sample(10,random_state=1)
saml=sam.pop("Label")
samc=sam.pop("attack_cat")

label = df.pop("Label")
cat = df.pop("attack_cat")
#normalize between 0 and 1 maybe to fix loss
x = df.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled)

x_train = df.sample(frac=0.7)#,random_state=0)
x_test = df.drop(x_train.index)
x_test = x_test.reset_index(drop=True)
trainLabel = label.sample(frac=0.7)
testLabel=label.drop(trainLabel.index)

# Setup the network parameters:
original_dim = x_train.shape[1]
input_shape = (original_dim, )
intermediate_dim = 50
batch_size = 128
latent_dim = 2
epochs = 100
OPTIMIZER = "adadelta"
ENC_ACTIVATION = "relu"
DEC_ACTIVATION = "relu"
OUT_ACTIVATION = "sigmoid"
# Map inputs to the latent distribution parameters:
# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
x = Dense(intermediate_dim, activation=ENC_ACTIVATION)(inputs)
z_mean = Dense(latent_dim, name='z_mean')(x)
z_log_var = Dense(latent_dim, name='z_log_var')(x)

# Use those parameters to sample new points from the latent space:
# reparameterization trick
# instead of sampling from Q(z|X), sample epsilon = N(0,I)
# z = z_mean + sqrt(var) * epsilon
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
  
# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])


# Instantiate the encoder model:
encoder = Model(inputs, z_mean)
# encoder.summary() #print summary of variables
 #plot_model(encoder, to_file='vae_mlp_encoder.png', show_shapes=True)
# Build the decoder model:
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(intermediate_dim, activation=DEC_ACTIVATION)(latent_inputs)
outputs = Dense(original_dim, activation=OUT_ACTIVATION)(x)
# Instantiate the decoder model:
decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
 #plot_model(decoder, to_file='vae_mlp_decoder.png', show_shapes=True)

# Instantiate the VAE model:
outputs = decoder(encoder(inputs))
vae = Model(inputs, outputs, name='vae_mlp')
 #plot_model(vae,to_file='vae_mlp.png',show_shapes=True)
# As in the Keras tutorial, we define a custom loss function:
def vae_loss(x, x_decoded_mean):
    xent_loss = losses.binary_crossentropy(x, x_decoded_mean)
    kl_loss = - 0.5 * K.mean(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
    return xent_loss + kl_loss
  
# We compile the model:
vae.compile(optimizer=OPTIMIZER, loss=vae_loss)

# Finally, we train the model:
results = vae.fit(x_train, trainLabel,
        shuffle=True,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=(x_test, testLabel))
def saveModel(model, name):
    modelJson = model.to_json()
    nameJSON=str(name)+".json"
    nameH5 = str(name)+".h5"
    with open(nameJSON,"w")as jsonF:
        jsonF.write(modelJson)
    #seraialize the wirghts to HDF5
    model.save_weights(nameH5)
    print("Model {} has been saved to disk".format(name))

def loasModel(name):
    nameJSON=str(name)+".json"
    nameH5 = str(name)+".h5"
    jsonFile = open(nameJSON,"r")
    loadedModel = jsonFile.read()
    jsonFile.close()
    loadedModel = model_from_json(loadedModel)
    loadedModel.load_weights(nameH5)
    return loadedModel
def plotLoss():
    plt.plot(results.history['loss'])
    plt.plot(results.history['val_loss'])
    plt.title('Model loss\nInformation: Batch Size: {} |ID: {} |LD: {} |Optimizer: {}'.format(batch_size,intermediate_dim,latent_dim,OPTIMIZER))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right',fancybox=True,framealpha=1,shadow=True,borderpad=1)
    plt.show()
#encoded = encoder.predict(x_test)
def applyTSNE(data):
    tsne = TSNE(n_components=2, verbose=1, perplexity=25, n_iter=500)
    #features = data.drop(['Label','attack_cat'],axis=1)

    tsne_results = tsne.fit_transform(data)
    target = cat.sample(frac=0.7)
    df_subset = pd.DataFrame()
    df_subset['tsne-2d-one'] = tsne_results[:,0]
    df_subset['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
        x="tsne-2d-one", y="tsne-2d-two",
        hue=target,
        palette=sns.color_palette("hls", len(target.unique().tolist())),
        data=df_subset,
        legend="full",
        alpha=0.3
    )
    
    plt.show()
# applyTSNE(encoded) #apply TSNE to the laten dimension of the test or train dataset 
# encoded = encoder.predict(x_test)
model=Sequential(layers=encoder.layers)

ynew = model.predict_classes(sam)
# print(ynew)
# yprob =encoder.predict(x)
for i in range(10):
    print ("Prediction is: {} and actual class is: {}".format(ynew[i],saml.iloc[i]))
