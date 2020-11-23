

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
import umap

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

#%% TODO:  move to utility.py
def make_dir(dirname):
    if os.path.isdir(dirname):
        return 
    else:
        os.mkdir(dirname)
        print(f"made {dirname}")
        return

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


#%%
#importlib.reload(kcv) # works



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

def get_n_samples(ds):
    n_samples = 0
    for _ in ds :
        n_samples += 1
    return n_samples 


def get_ds(pix_dim):
    return ut.load_and_dump(pix_dim, cf.IMAGE_FILEPATH)


def create_imgStats(ds,n_samples):
    db = {}
    for sample, label in tqdm(ds, 
                                unit_scale=True, 
                                desc="Saving data stats: ", 
                                unit=" samples", 
                                total=n_samples ) :

        key = label.numpy().decode().split('/')[-3]
        if key in db.keys():
            imgs = db[key]
            db[key] = np.concatenate((imgs,sample[None,...]),axis=0)
        else:
            db[key] = sample[None,...]

    return db


def create_snk2dicts(ds,n_samples,vae):
    snk2loss = {}
    snk2vec = {}
    for sample, label in tqdm(ds, 
                                unit_scale=True, 
                                desc="Saving shape 2 vec: ", 
                                unit=" encodes", 
                                total=n_samples ) :
        key = label.numpy()  # maybe should have changed this to a (string... but byte is good...
        snk2vec[key] = vae.encoder(sample[None,...]).sample().numpy()[0] #assume that we don't have a batch
        snk2loss[key] = vae.partial_vae_loss(sample[None,...]).numpy()

    return snk2loss, snk2vec

def load_and_prime_model(model,model_name,params,data_dir,epochs):
    filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"    
    hist,params = ut.load_pickle(os.path.join(data_dir,filename))

    vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                learning_rate=0.0001, kl_weight=params['kl_weight'])

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
    sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")
    vae.load_model(sv_path, epochs)

    return vae,hist

def dump_pickle(filepath, item_to_save):
    f = open(filepath, "wb")
    pickle.dump(item_to_save, f)
    f.close()

def load_pickle(filepath):
    infile = open(filepath, "rb")
    item = pickle.load(infile)
    infile.close()
    return item

def load_snk2pickles(data_dir,kl_weight):
    load_dir = os.path.join(data_dir,f"kl_weight{kl_weight:03d}")
    snk2vec = load_pickle(os.path.join(load_dir,"snk2vec.pkl"))
    snk2loss = load_pickle(os.path.join(load_dir,"snk2loss.pkl"))
    return snk2loss, snk2vec 

def dump_snk2pickles(snk2loss,snk2vec,data_dir,kl_weight):
    load_dir = os.path.join(data_dir,f"kl_weight{kl_weight:03d}")
    dump_pickle(os.path.join(load_dir,"snk2vec.pkl"), snk2vec)
    dump_pickle(os.path.join(load_dir,"snk2loss.pkl"), snk2loss)


def load_snk2umap(data_dir,kl_weight):
    load_dir = os.path.join(data_dir,f"kl_weight{kl_weight:03d}")
    snk2umap = load_pickle(os.path.join(load_dir,"snk2umap.pkl"))
    return snk2umap 

def dump_snk2umap(snk2umap,data_dir,kl_weight):
    load_dir = os.path.join(data_dir,f"kl_weight{kl_weight:03d}")
    print(load_dir)
    dump_pickle(os.path.join(load_dir,"snk2umap.pkl"), snk2umap)

def get_umap_embedding(snk2vec):
    latents = np.array(list(snk2vec.values()))
    reducer = umap.UMAP(random_state=666)
    reducer.fit(latents)
    embedding = reducer.transform(latents)
    assert(np.all(embedding == reducer.embedding_))
    snk2umap = dict(zip(np.array(list(snk2vec.keys())),embedding))

    return snk2umap

def overtrain_vae_and_collate(model, model_name, data_dir,params, epochs):
    # assumes access to these toplevel vars:
    # -train_dataset
    # -test_dataset
    # -ds (for dumping)
    make_dir(data_dir)

    sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")
    make_dir(sv_path)
    
    filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"

    # check to see if we've already done this
    if os.path.exists(os.path.join(data_dir,filename)):
        vae, hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
    else:
        # train and save
        vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                    learning_rate=0.0001, kl_weight=params['kl_weight'])

        vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

        train_history = vae.fit(train_dataset,epochs=epochs, 
                                verbose=0, validation_data=test_dataset)
                                #, initial_epoch = 11 )
        history = train_history.history
        filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"
        ut.dump_pickle(os.path.join(data_dir,filename), (history,params))
        make_dir(sv_path)
        print('save model')
        vae.save_model(sv_path, epochs)

    if os.path.exists(os.path.join(sv_path,"snk2vec.pkl")):
        snk2loss,snk2vec = load_snk2pickles(data_dir,params['kl_weight'])
    else:
        snk2loss,snk2vec = create_snk2dicts(ds,n_samples,vae)
        dump_snk2pickles(snk2loss,snk2vec,data_dir,params['kl_weight'])
    
    if os.path.exists(os.path.join(sv_path,"snk2umap.pkl")):
        snk2umap = load_snk2umap(data_dir,params['kl_weight'])
    else:
        snk2umap = get_umap_embedding(snk2vec)
        dump_snk2umap(snk2umap,data_dir,params['kl_weight'])   



#%%

#%% defaults
latent_dim = 40
pix_dim = 160
batch_size = 64
kl_weight = 5

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

pix_dims = [128,160,192]
klws = [1,2,3,5,10]
latents = [32,40,64]
models = [kcv.K_PCVAE,kcv.K_PCVAE_BN, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_BN", "K_PCVAE_KL_Reg"]

epochs = 400




#%%


for pix_dim in pix_dims:
    params['x_dim']=(pix_dim,pix_dim,3)

    train_dataset = ut.load_prep_and_batch_data(train_data, params['x_dim'][0],params['batch_size'], augment=True)
    test_dataset =  ut.load_prep_and_batch_data(  val_data, params['x_dim'][0],params['batch_size'], augment=False)

    ds = get_ds(pix_dim)
    n_samples = get_n_samples(ds)

    # for test_samples, test_labels in test_dataset.take(1): 
    #     pass
    # for train_samples, train_labels in train_dataset.take(1): 
    #     pass
    # x = test_samples
    for model,model_name in zip(models,model_names):
    #for model in models:
        for lat in latents:
            params['z_dim'] = lat
            for kl in klws:
                params['kl_weight'] = kl
                
                print(f"training beta={kl} z={lat}, x={pix_dim}")
                data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
                print(data_dir)
                
                overtrain_vae_and_collate(model, model_name, data_dir,params,epochs)
                print("trained")





#%%  look to see if what/bach size has on things...

#batches = [16,32]





# %%
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
