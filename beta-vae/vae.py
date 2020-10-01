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

import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

JUPYTER_NOTEBOOK = False

# if JUPYTER_NOTEBOOK:
# %reload_ext autoreload
# %autoreload 2

#%% Setup
#######
cf_img_size = cf.IMG_SIZE
cf_latent_dim = cf.LATENT_DIM
cf_batch_size = cf.BATCH_SIZE #32
cf_learning_rate = cf.IMGRUN_LR #4e-4
cf_limits = [cf_img_size, cf_img_size]
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#
cf_kl_weight = cf.KL_WEIGHT
cf_num_epochs = cf.N_IMGRUN_EPOCH
#dfmeta = ut.read_meta()
cf_val_frac = cf.VALIDATION_FRAC
#%%  are we GPU-ed?
tf.config.experimental.list_physical_devices('GPU') 


#%% Define Training methods
def get_test_set_loss(dataset, batches=0) :
    test_losses = []
    for test_x, test_label in (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
        #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
        test_cost_batch = model.compute_test_loss(test_x)  # this should turn off the dropout...
        test_losses.append(test_cost_batch)

    return np.mean(test_losses)


def train_model(epochs, display_interval=-1, save_interval=10, test_interval=10,current_losses=([],[])) :
    """
    custom training loops to enable dumping images of the progress
    """
    print('\n\nStarting training...\n')
    model.training=True
    elbo_test,elbo_train = current_losses
    if len(elbo_test)>0:
        print(f"test: n={len(elbo_test)}, last={elbo_test[-1]}")
        print(f"train: n={len(elbo_train)}, last={elbo_train[-1]}")

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
        if (ut.check_stop_signal(dir_path=cf.IMGRUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")
    
    out_losses = (elbo_train,elbo_test)
    return epoch, out_losses #(loss_batch2,loss_batchN)



#%% #################################################
##
##  LOAD/PREP data
##         - l if we've already been through this for the current database we'll load... otherwise process.
#####################################################



data_from_scratch = not ut.check_for_datafiles(cf.DATA_DIR,['train_data.npy','val_data.npy','all_data.npy'])
#data_from_scratch = True
random.seed(488)
tf.random.set_seed(488)

if data_from_scratch:
    #create
    files = glob.glob(os.path.join(cf.IMAGE_FILEPATH, "*/img/*"))
    files = np.asarray(files)
    train_data, val_data, all_data = ut.split_shuffle_data(files,cf_val_frac)
    # Save base train data to file  
    np.save(os.path.join(cf.DATA_DIR, 'train_data.npy'), train_data, allow_pickle=True)
    np.save(os.path.join(cf.DATA_DIR, 'val_data.npy'), val_data, allow_pickle=True)
    np.save(os.path.join(cf.DATA_DIR, 'all_data.npy'), all_data, allow_pickle=True)
else:
    #load
    print(f"loading train/validate data from {cf.DATA_DIR}")
    train_data = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'), allow_pickle=True)
    val_data = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'), allow_pickle=True)
    all_data = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'), allow_pickle=True)


#%% #################################################
##
##  Set up the model 
##         - load current state or
##         - train from scratch
#####################################################

model = cv.CVAE(cf_latent_dim, cf_img_size, learning_rate=cf_learning_rate, kl_weight=cf_kl_weight, training=True)
### instance of model used in GOAT blog
#model = cv.CVAE_EF(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)

model.print_model_summary()
model.print_model_IO()

if JUPYTER_NOTEBOOK:
    tf.keras.utils.plot_model(model.enc_model, show_shapes=True, show_layer_names=True)
    tf.keras.utils.plot_model(model.gen_model, show_shapes=True, show_layer_names=True)


#%% Setup logger info
train_from_scratch = ( cf.CURR_IMGRUN_ID is None )

if train_from_scratch:
    lg = logger.logger(trainMode=True, txtMode=False)
    lg.setup_checkpoint(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer) # sets up the writer
    #lg.restore_checkpoint() 
    lg.check_make_dirs() # makes all the direcotries
    # copy to the current run train data to file
    np.save(os.path.join(lg.saved_data, 'train_data.npy'), train_data, allow_pickle=True)
    np.save(os.path.join(lg.saved_data, 'val_data.npy'), val_data, allow_pickle=True)
    np.save(os.path.join(lg.saved_data, 'all_data.npy'), all_data, allow_pickle=True)
    total_epochs = 0
    curr_losses = ([],[])
else:
    root_dir = os.path.join(cf.IMGRUN_DIR, cf.CURR_IMGRUN_ID)
    lg = logger.logger(root_dir=root_dir, trainMode=True, txtMode=False)
    lg.setup_checkpoint(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer) # sets up the writer
    lg.restore_checkpoint() # actuall reads in the  weights...
    allfiles = os.listdir(lg.saved_data)
    print(f"allfiles: {allfiles}")
    total_epochs = [int(f.rstrip(".pkl").lstrip("losses_")) for f in allfiles if f.startswith("losses_")]
    total_epochs.sort(reverse=True)
    print(f"total_epochs = {total_epochs[0]}")
    total_epochs = total_epochs[0]
    curr_losses = ut.load_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"))



#%% # LOAD & PREPROCESS the from list of filessudo apt install gnome-tweak-tool
# could simplify this by making another "load_prep_batch_data(train_data,imagesize,augment=True,)"
train_dataset = ut.load_prep_and_batch_data(train_data, cf_img_size, cf_batch_size, augment=True)
test_dataset =  ut.load_prep_and_batch_data(  val_data, cf_img_size, cf_batch_size, augment=False)

# train_dataset = tf.data.Dataset.from_tensor_slices(train_data)
# test_dataset = tf.data.Dataset.from_tensor_slices(val_data)
# train_dataset = ut.load_and_prep_data(cf_img_size, train_dataset, augment=True)
# test_dataset = ut.load_and_prep_data(cf_img_size,  test_dataset, augment=False)
# train_dataset = ut.batch_data(train_dataset)
# test_dataset = ut.batch_data(test_dataset)

#%% Load all data
# get some samples
for train_samples, train_labels in train_dataset.take(1) : pass
for test_samples, test_labels in test_dataset.take(1) : pass

# count number of batches... 
total_train_batchs = 0
for _ in train_dataset :
    total_train_batchs += 1

# #%% Setup datasets
sample_index = 1

#%% Training & Validation data save?
# do we want to save the image data for the training set... i.e. the augmented bytes?
dump_image_data = False
if dump_image_data:

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

    ut.dump_pickle(os.path.join(lg.saved_data,"train_agumented.pkl"), (trainimgs,trainlabs) )

    # validation data save 
    batch_index = 1
    imgs = []
    labels = []
    for test_x, label in test_dataset :
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


#%% 
# #################################################
##
##  log the run and TRAIN!!
##    - train from scratch OR 
##    - start where we left off
##
#####################################################
# log Config...
lg.write_config(locals(), [cv.CVAE, cv.CVAE.__init__])
lg.update_plot_dir()
#tf.config.experimental.list_physical_devices('GPU') 

#%% 
n_epochs = cf_num_epochs
epoch_n, curr_losses = train_model(n_epochs, display_interval=5, save_interval=20, test_interval=5,current_losses=curr_losses)
#epoch_n,elbo_train,elbo_test = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n
if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
else:
    lg.reset(total_epochs=total_epochs)
model.save_model(lg.root_dir, lg.total_epochs )

ut.dump_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"),curr_losses)



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
