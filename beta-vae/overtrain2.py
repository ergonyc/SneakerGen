

#%% Imports
import numpy as np
import os
from sys import stdout
import time
import json
import pandas as pd
import random
import pickle
import matplotlib.pyplot as plt
from tqdm import tqdm
import glob

import tensorflow as tf
import tensorflow_probability as tfp

#import betavae as cv
#import pcvae as cv
import kcvae as kcv 

import utils as ut
import logger
import configs as cf

import importlib

AUTOTUNE = tf.data.experimental.AUTOTUNE

JUPYTER_NOTEBOOK = True

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
cf_pixel_dim = cf_img_size*cf_img_size*3
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#
cf_kl_weight = cf.KL_WEIGHT
cf_beta = cf_kl_weight
cf_num_epochs = cf.N_IMGRUN_EPOCH
#dfmeta = ut.read_meta()
cf_val_frac = cf.VALIDATION_FRAC
#%%  are we GPU-ed?
tf.config.experimental.list_physical_devices('GPU') 
#
#physical_devices = tf.config.list_physical_devices('GPU')
#print("Num GPUs:", len(physical_devices))
#%% helpers
latent_dim = 40
pix_dim = 160
batch_size = 64
epochs = 100
kl_weight = 5

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

#%% TODO:  move to utility.py
def make_dir(dirname):
    if os.path.isdir(dirname):
        return
    else:
        os.mkdir(dirname)
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


#%% # LOAD & PREPROCESS the from list of filessudo apt install gnome-tweak-tool
train_dataset = ut.load_prep_and_batch_data(train_data, params['x_dim'][0],params['batch_size'], augment=True)
test_dataset =  ut.load_prep_and_batch_data(  val_data, params['x_dim'][0],params['batch_size'], augment=False)
#%% ####################   load some data for easy access  ####################

for test_samples, test_labels in test_dataset.take(1): 
    pass
for train_samples, train_labels in train_dataset.take(1): 
    pass

x = test_samples


#%%
#importlib.reload(kcv) # works


#%%


#%%
model = kcv.K_PCVAE_BN
model_name = "K_PCVAE_BN"
# model = kcv.K_PCVAE
# model_name = "K_PCVAE"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
epochs = 400

def overtrain_vae(model, model_name, data_dir,params, epochs):
    make_dir(data_dir)
    
    vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                 learning_rate=0.0001, kl_weight=params['kl_weight'])

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=0, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"
    ut.dump_pickle(os.path.join(data_dir,filename), (history,params))


    sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")

    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)



#%%
overtrain_vae(model, model_name, data_dir,params,epochs)


#%%%

model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
#overtrain_vae(model, model_name, data_dir,params,epochs)
# %%

latent_dim = 40
pix_dim = 160
batch_size = 64
epochs = 100
kl_weight = 5

klws = [1,2,3,5,10]
latents = [32,40,64]
epochs = 400


for kl in klws:
    params['kl_weight'] = kl
    for l in latents:
        params['z_dim'] = l
        print(f"training beta={kl} z={l}")
        data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
        print(data_dir)
        overtrain_vae(model, model_name, data_dir,params,epochs)
        print("trained")



#


# %%

latent_dim = 40
pix_dim = 160
batch_size = 64
epochs = 100
kl_weight = 5

klws = [1,2,3,5,10]
latents = [32,40,64]
epochs = 400


for kl in klws:
    params['kl_weight'] = kl
    for l in latents:
        params['z_dim'] = l
        print(f"training beta={kl} z={l}")
        data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
        print(data_dir)
        overtrain_vae(model, model_name, data_dir,params,epochs)
        print("trained")



#


model = kcv.K_PCVAE_KL_Reg
model_name = "K_PCVAE_KL_Reg"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
#overtrain_vae(model, model_name, data_dir,params,epochs)

latent_dim = 40
pix_dim = 160
batch_size = 64
kl_weight = 5

klws = [1,2,3,5,10]
latents = [32,40,64]
epochs = 400

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

for kl in klws:
    params['kl_weight'] = kl
    for l in latents:
        params['z_dim'] = l
        print(f"training beta={kl} z={l}")
        data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
        print(data_dir)
        overtrain_vae(model, model_name, data_dir,params,epochs)
        print("trained")



#%%

latent_dim = 40
pix_dim = 160
batch_size = 64
kl_weight = 5

model = kcv.K_PCVAE
model_name = "K_PCVAE"
batch_size = 64
klws = [1,2,3,5,10]
latents = [32,40,64]
pix_dims = [192,224]
epochs = 400

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

# finished everything but the biggest KL_WEIHT


for pdim in pix_dims:
    params['x_dim']=(pdim,pdim,3)
    train_dataset = ut.load_prep_and_batch_data(train_data, params['x_dim'][0],params['batch_size'], augment=True)
    test_dataset =  ut.load_prep_and_batch_data(  val_data, params['x_dim'][0],params['batch_size'], augment=False)


    for test_samples, test_labels in test_dataset.take(1): 
        pass
    for train_samples, train_labels in train_dataset.take(1): 
        pass
    x = test_samples

    # if pdim == 192:
    #     klws = [10]
    # else:
    #     klws = [1,2,3,5,10]

    for kl in klws:
        params['kl_weight'] = kl
        for l in latents:
            params['z_dim'] = l
            print(f"training beta={kl} z={l}, x={pdim}")
            data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
            print(data_dir)
            overtrain_vae(model, model_name, data_dir,params,epochs)
            print("trained")




#%%  look to see if what/bach size has on things...

#batches = [16,32]

(pix_dim,pix_dim,3)








+# %%
filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"    
history,params = ut.load_pickle(os.path.join(data_dir,filename))

vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                learning_rate=0.0001, kl_weight=params['kl_weight'])

vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")
vae.load_model(sv_path, epochs)




# %%
x
# %%
x.shape
# %%
xhat = vae(x)
# %%
tf.math.squared_difference(x,xhat).numpy().mean()
# %%
xhat.numpy().min()
# %%
plt.imshow(xhat[0,].numpy().squeeze())
# %%
