'''
This file is used to train the shape autoencoder model.

It uses cvae.py as the base model and many data functions from utils to make it simpler.

It also has various methods for exploring a trained model to see how well it can reconstruct models and
interpolate between various reconstructions.

At the end there is a method called 'journey' which extends on the idea of interpolating between 2 chosen models
and chooses the models automatically on repeat to create cool interpolation animations.
'''
#

#%% Imports
import numpy as np
import os
from shutil import copyfile
import subprocess
from sys import getsizeof, stdout
from scipy import spatial

import time
import json
import pandas as pd
import random
import inspect

import pickle
from tqdm import tqdm
import glob

import cvae as cv
import utils as ut
import logger
import configs as cf

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.offline import plot
import plotly.figure_factory as FF
import plotly.express as px
from sklearn.manifold import TSNE
import seaborn as sns
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE
VAL_FRAC = 20.0 / 100.0

# %reload_ext autoreload
# %autoreload 2

#%% Setup

random.seed(488)
tf.random.set_seed(488)

### currently these are dummies until i can re-write the interpolation functions
# TODO remove "categories" and fix the interpolatin tools
cf_cat_prefixes = ut.cf_cat_prefixes = ['goat','sns']
cf_num_classes = len(cf_cat_prefixes)
#######
cf_img_size = cf.IMG_SIZE
cf_latent_dim = cf.LATENT_DIM
cf_batch_size = 32
cf_learning_rate = 4e-4
cf_limits = [cf_img_size, cf_img_size]
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#
cf_kl_weight = cf.KL_WEIGHT

dfmeta = ut.read_meta()


#%%  are we GPU-ed?
tf.config.experimental.list_physical_devices('GPU') 


#%% model viz function
# can i make this a model function??
def plot_IO(model):
    print("\n Net Summary (input then output):")
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


#%% Make model and print info
model = cv.CVAE(cf_latent_dim, cf_img_size, learning_rate=cf_learning_rate, kl_weight=cf_kl_weight, training=True)
### instance of model used in GOAT blog
#model = cv.CVAE_EF(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)

#model.setLR(cf_learning_rate)
model.print_model_summary()
model.print_model_IO()
plot_IO(model.enc_model)
plot_IO(model.gen_model)

#%% Setup logger info
train_from_scratch = True
if train_from_scratch :
    lg = logger.logger(trainMode=True, txtMode=False)
else :
    shp_run_id = '0922-1614'  
    root_dir = os.path.join(cf.IMG_RUN_DIR, shp_run_id)
    lg = logger.logger(root_dir=root_dir, trainMode=True, txtMode=False)

lg.setup_checkpoint(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer)
lg.check_make_dirs()
lg.restore_checkpoint() # actuall reads in the  weights...



#%% define splitShuffleData

def split_shuffle_data(files, test_split):
    """[summary]

    Args:
        files ([type]): [description]
        test_split ([type]): [description]

    Returns:
        [test_files,val_files,is_val]: [description]
    """

    for _ in range(100):  # shuffle 100 times
       np.random.shuffle(files)

    data_size = files.shape[0]
    train_size = int((1.0 - test_split) * data_size)
    ceil = lambda x: int(-(-x // 1))
    val_size = ceil(test_split * data_size)

    is_val = np.zeros(data_size,dtype=int)
    is_val[train_size:]=1

    train_data = files[0:train_size]
    val_data = files[train_size:]

    assert len(train_data) == train_size , "split wasn't clean (train)"
    assert len(val_data) == val_size , "split wasn't clean (validate)"
    #all_data = zip(files,is_val)
    all_data = [list(z) for z in zip(files,is_val)]
    return train_data, val_data, all_data


#%% Set up the data..
# TODO:  logical on does train_data exist... if not make it.. else just load

img_in_dir = lg.img_in_dir
#img_in_dir = "/Users/ergonyc/Projects/DATABASE/SnkrScrpr/data/"
files = glob.glob(os.path.join(img_in_dir, "*/img/*"))
#shuffle the dataset (GOAT+SNS)
files = np.asarray(files)

train_data, val_data, all_data = split_shuffle_data(files,VAL_FRAC)

# # ## Save base train data to file  
np.save(os.path.join(cf.DATA_DIR, 'train_data.npy'), train_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'val_data.npy'), val_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'all_data.npy'), all_data, allow_pickle=True)


#%% Load base train data from finterval=5,curr_losses=curr_losses)
train_dat = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'))
val_dat = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'))
all_dat = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'))



#%% # LOAD & PREPROCESS the from list of filessudo apt install gnome-tweak-tool


train_dataset = tf.data.Dataset.from_tensor_slices(train_dat)
test_dataset = tf.data.Dataset.from_tensor_slices(val_dat)

train_dataset = ut.load_and_prep_data(cf_img_size, train_dataset, augment=True)
test_dataset = ut.load_and_prep_data(cf_img_size,  test_dataset, augment=False)

def batch_and_prep_dataset(dataset):
    dataset = dataset.batch(cf_batch_size, drop_remainder=False)  #this might mess stuff up....
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_dataset = batch_and_prep_dataset(train_dataset)
test_dataset = batch_and_prep_dataset(test_dataset)

#%% Load all data
# why do we need to do this??? to get a list of samples??
for train_samples, train_labels in train_dataset.take(1) : pass
for test_samples, test_labels in test_dataset.take(1) : pass


#%% Load all data get number of batches... 

total_train_batchs = 0
for _ in train_dataset :
    total_train_batchs += 1


# #%% Setup datasets
sample_index = 1

#%% Training methods
def get_test_set_loss(dataset, batches=0) :
    test_losses = []
    for test_x, test_label in (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
        #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
        #test_loss_batch = model.compute_loss(test_x)
        #test_cost_batch = model.vae_cost(test_x)
        test_cost_batch = model.compute_test_loss(test_x)  # this should turn off the dropout...
        
        test_losses.append(test_cost_batch)
    return np.mean(test_losses)


def train_model(epochs, display_interval=-1, save_interval=10, test_interval=10,current_losses=([],[])) :
    print('\n\nStarting training...\n')
    model.training=True
    elbo_test,elbo_train = current_losses

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        losses = []
        batch_index = 1

        # DO THE AUGMENTATION HERE...
        for train_x, label in train_dataset :
            #train_x = tf.cast(train_x, dtype=tf.float32)
            loss_batch = model.trainStep(train_x)
            losses.append(loss_batch)
            stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
            stdout.flush()  

            batch_index = batch_index + 1

        ## TRAIN LOSS
        elbo = np.mean(losses)
        print('Epoch: {}   Train loss: {:.1f}   Epoch Time: {:.2f}'.format(lg.total_epochs, 
                            float(elbo),
                            float(time.time() - start_time)) )

        lg.log_metric(elbo, 'train loss',test=False)
        elbo_train.append(elbo)

        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            if epoch == 1:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)    
            else:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=False, save_fig=True, limits=cf_limits)

        ## TEST LOSSin chekmakedirs
        if epoch % test_interval == 0:
            test_loss = get_test_set_loss(test_dataset, cf_batch_size)  # what should I do here??/ batch of 2???  shouldn't it be batch size??
            print('   TEST LOSS  : {:.1f}    for epoch: {}'.format(test_loss, 
                                                        lg.total_epochs))
            lg.log_metric(test_loss, 'test loss',test=True)
            elbo_test.append(test_loss)

        ## SAVE
        if epoch % save_interval == 0:
            lg.save_checkpoint()


        lg.increment_epoch()

        if (ut.check_stop_signal(dir_path=cf.IMG_RUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")
    
    out_losses = (elbo_train,elbo_test)
    return epoch, out_losses #(loss_batch2,loss_batchN)


#%% Training data save
start_time = time.time()
batch_index = 1
imgs = []
labels = []

for train_x, label in train_dataset :
    #train_x = tf.cast(train_x, dtype=tf.float32)
    #imgs.append(np.moveaxis(train_x.numpy(),0,-1)) # put the "batch" at the end so we can stack
    imgs.append(train_x.numpy()) # put the "batch" at the end so we can stack
    labs = [l.numpy().decode() for l in label]# decode makes this a simple string??
    labels.extend(labs)
    stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
    stdout.flush()
    batch_index = batch_index + 1


trainimgs = np.concatenate(imgs,axis=0)
trainlabs = labels # np.stack(labels)
False
print('Epoch Time: {:.2f}'.format( float(time.time() - start_time)))

#path = os.path.join(cf.IMG_RUN_DIR, "saved_data","train")
# #save training data
# tf.data.experimental.save(
#     train_dataset, path, compression=None, shard_func=None
# )
ut.dump_pickle(os.path.join(lg.saved_data,"train_agumented.pkl"), (trainimgs,trainlabs) )


#%% validation data save 
batch_index = 1
imgs = []
labels = []
for test_x, label in test_dataset :
    #train_x = tf.cast(train_x, dtype=tf.float32)
    #imgs.append(np.moveaxis(train_x.numpy(),0,-1)) # put the "batch" at the end so we can stack
    imgs.append(train_x.numpy()) # put the "batch" at the end so we can stack
    labs =  [l.numpy().decode() for l in label] # decode makes this a simple string??
    labels.extend(labs)

    stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, 16))
    stdout.flush()
    batch_index = batch_index + 1

flatten = lambda l: [item for sublist in l for item in sublist]

testlabs = labels # np.stack(labels)
testimgs = np.concatenate(imgs,axis=0)
print('Epoch Time: {:.2f}'.format( float(time.time() - start_time)))

ut.dump_pickle(os.path.join(lg.saved_data,"test.pkl"), (testimgs,testlabs) )
#%% log Config...
lg.writeConfig(locals(), [cv.CVAE, cv.CVAE.__init__])
lg.updatePlotDir()
tf.config.experimental.list_physical_devices('GPU') 

# copy to the current run train data to file
np.save(os.path.join(lg.saved_data, 'train_data.npy'), train_data, allow_pickle=True)
np.save(os.path.join(lg.saved_data, 'val_data.npy'), val_data, allow_pickle=True)
np.save(os.path.join(lg.saved_data, 'all_data.npy'), all_data, allow_pickle=True)

#%% 
n_epochs = 300
total_epochs = 0
epoch_n, curr_losses = train_model(n_epochs, display_interval=5, save_interval=20, test_interval=5,current_losses=([],[]))
#epoch_n,elbo_train,elbo_test = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n
if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
else:
    lg.reset(total_epochs=total_epochs)
model.save_model(lg.root_dir, lg.total_epochs )

ut.dump_pickle(os.path.join(lg.saved_data, "losses.pkl"),curr_losses)



#%%


#%% 

for test_samples, test_labels in test_dataset.take(1) : pass
for train_samples, train_labels in train_dataset.take(1) : pass

#%% 
sample_index = 1

for sample_index in range(10):
    title_text = f"trained n={sample_index}"
    ut.show_reconstruct(model, train_samples, title=title_text, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)

for sample_index in range(10):
    title_text = f"tested n={sample_index}"
    ut.show_reconstruct(model, test_samples, title=title_text, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)

###########################
############################
#
#  Now make some easy access databases...
#
############################
###########################
#%% 

# ut.make_gif_from_dir(gif_in_dir, name):

# model.save_model(lg.root_dir, 138)
# #%% 

# model.load_model(lg.root_dir,669)
# # Need to make methods to extract the pictures 

#%% Run model on all data to get latent vects and loss. Used for streamlit app and other places.
#preds,losses = ut.dumpReconstruct( model, train_dataset, test_dataset )
ds = ut.load_and_dump(cf_img_size, lg.img_in_dir)
#or _samples, _labels in ds.take(1) : pass
# remake this to simply go through all the data and calculate the embedding and loss... new functions probably...
#%%count our n
n_samples = 0
for _ in ds :
    n_samples += 1
#%% dump the vectors to a dictionary

snk2loss = {}
snk2vec = {}
for sample, label in tqdm(ds, 
                            unit_scale=True, 
                            desc="Saving shape 2 vec: ", 
                            unit=" encodes", 
                            total=n_samples ) :
    #sample = tf.cast(sample, dtype=tf.float32)
    key = label.numpy()  # maybe should have changed this to a string... but byte is good...
    snk2vec[key] = model.encode(sample[None,...], reparam=True).numpy()[0]
    snk2loss[key] = model.compute_loss(sample[None,...]).numpy()

ut.dump_pickle(os.path.join(lg.root_dir,"snk2vec.pkl"), snk2vec)
ut.dump_pickle(os.path.join(lg.root_dir,"snk2loss.pkl"), snk2loss)




#################
#################
