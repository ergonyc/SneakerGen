

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


def overtrain_vae(model, model_name, data_dir,params, epochs):
    make_dir(data_dir)
    
    vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                 learning_rate=0.0001, kl_weight=params['kl_weight'])
    loss = vae.partial_vae_loss
    vae.compile(optimizer=vae.optimizer, loss = loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=1, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"
    ut.dump_pickle(os.path.join(data_dir,filename), (history,params))


    sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")

    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)



#%%
#overtrain_vae(model, model_name, data_dir,params,epochs)

# %%
model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"


latent_dim = 24
pix_dim = 160
batch_size = 64
epochs = 200
kl_weight = 4
params['kl_weight'] = kl_weight
params['z_dim'] = latent_dim
print(f"training beta={kl_weight} z={latent_dim}")
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
print(data_dir)
overtrain_vae(model, model_name, data_dir,params,epochs)
print("trained")






# %%

# %%
model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"


pix_dim = 160
batch_size = 64
epochs = 400
kl_weight = 5
latent_dim = 40


params['kl_weight'] = kl_weight
params['z_dim'] = latent_dim

print(f"training beta={kl_weight} z={latent_dim}")
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
print(data_dir)


model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"


filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"    
history,params = ut.load_pickle(os.path.join(data_dir,filename))

vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'], 
                learning_rate=0.0001, kl_weight=params['kl_weight'])

vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")
vae.load_model(sv_path, epochs)




# # %%
# x
# # %%
# x.shape
# # %%
# xhat = vae(x)
# # %%
# tf.math.squared_difference(x,xhat).numpy().mean()
# # %%
# xhat.numpy().min()
# # %%
# plt.imshow(xhat[0,].numpy().squeeze())
# # %%



#%% Run model on all data to get latent vects and loss. Used for streamlit app and other places.
#preds,losses = ut.dumpReconstruct( model, train_dataset, test_dataset )
ds = ut.load_and_dump(pix_dim, cf.IMAGE_FILEPATH)
#or _samples, _labels in ds.take(1) : pass
# remake this to simply go through all the data and calculate the embedding and loss... new functions probably...
#%%count our n
n_samples = 0
for _ in ds :
    n_samples += 1
#%% dump the vectors to a dictionary
make_dicts = False
if make_dicts:

    snk2loss = {}
    snk2vec = {}
    for sample, label in tqdm(ds, 
                                unit_scale=True, 
                                desc="Saving shape 2 vec: ", 
                                unit=" encodes", 
                                total=n_samples ) :
        #sample = tf.cast(sample, dtype=tf.float32)
        key = label.numpy()  # maybe should have changed this to a (string... but byte is good...
        snk2vec[key] = vae.encoder(sample[None,...]).sample().numpy()[0]
        snk2loss[key] = vae.partial_vae_loss(sample[None,...]).numpy()



    ut.dump_pickle(os.path.join(data_dir,"snk2vec.pkl"), snk2vec)
    ut.dump_pickle(os.path.join(data_dir,"snk2loss.pkl"), snk2loss)

else:
    snk2vec = ut.load_pickle(os.path.join(data_dir,"snk2vec.pkl"))
    snk2loss = ut.load_pickle(os.path.join(data_dir,"snk2loss.pkl"))

# %%

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import umap



dfdesc = pd.read_csv(cf.DESCRIPTIONS_CSV)
df_meta = pd.read_csv(cf.META_DATA_CSV)
# fix the root directory....


digits_df = pd.DataFrame.from_dict(snk2vec, orient='index')

#%%
sns.pairplot(digits_df, hue='digit', palette='Spectral')


reducer = umap.UMAP(random_state=42)
reducer.fit(digits_df.to_numpy())

# UMAP(a=None, angular_rp_forest=False, b=None,
#      force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
#      local_connectivity=1.0, low_memory=False, metric='euclidean',
#      metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
#      n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
#      output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
#      set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
#      target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
#      transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)




embedding = reducer.transform(digits_df.to_numpy())
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
embedding.shape

plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24);

# %%
