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

#import seaborn as sns
import umap

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

#%% Setup generic values...
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



# # %%
# #%%
# sns.pairplot(digits_df, hue='digit', palette='Spectral')


# reducer = umap.UMAP(random_state=42)
# reducer.fit(digits_df.to_numpy())

# #%% Run model on all data to get latent vects and loss. Used for streamlit app and other places.
# #preds,losses = ut.dumpReconstruct( model, train_dataset, test_dataset )
# ds = ut.load_and_dump(pix_dim, cf.IMAGE_FILEPATH)
# #or _samples, _labels in ds.take(1) : pass
# # remake this to simply go through all the data and calculate the embedding and loss... new functions probably...
# %%
# HELPERS:
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
    umap = load_pickle(os.path.join(load_dir,"snk2umap.pkl"))
    return snk2loss, snk2vec 

def dump_snk2umap(snk2umap,data_dir,kl_weight):
    load_dir = os.path.join(data_dir,f"kl_weight{kl_weight:03d}")
    dump_pickle(os.path.join(load_dir,"snk2umap.pkl"), snk2umap)

def get_umap_embedding(snk2vec):
    latents = np.array(list(snk2vec.values()))
    reducer = umap.UMAP(random_state=42)
    reducer.fit(latents)
    embedding = reducer.transform(latents)
    assert(np.all(embedding == reducer.embedding_))
    snk2umap = dict(zip(np.array(list(snk2vec.keys())),embedding))

    return snk2umap


#%% dump the vectors to a dictionary

# ds = get_ds(pix_dim)
# n_samples = get_n_samples(ds)

# model = kcv.K_PCVAE
# model_name = "K_PCVAE"
# epochs = 400

# data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
# vae,hist = load_and_prime_model(model,model_name,params,data_dir,epochs)

# make_dicts = False
# if make_dicts:
#     snk2loss,snk2vec,svecs = create_snk2dicts(ds,n_samples,vae, dump_dicts=True)
# else:
#     snk2loss,snk2vec = load_snk2pickles(data_dir)

# %%  define function to create snk2vec, snk2loss, and UMAP objects for each network / latents



# %%   
#############################
#
#  comparison 1:  KL_Reg model vs KL_loss model @x160
#
###############################
latent_dim = 40
pix_dim = 160
batch_size = 64
kl_weight = 5

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

ds = get_ds(pix_dim)
n_samples = get_n_samples(ds)

#loop over models
models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]
data_dirs = [f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}" for model_name in model_names]

klws = [1,2,3,5,10]
latents = [32,40,64]
epochs = 400

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

for model,model_name in zip(models,model_names):

    for kl in klws:
        params['kl_weight'] = kl
        for l in latents:

            params['z_dim'] = l
            data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
            print(data_dir)

            vae, hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
            snk2loss,snk2vec = create_snk2dicts(ds,n_samples,vae)

            dump_snk2pickles(snk2loss,snk2vec,data_dir,params['kl_weight'])



#

#############################
#
#  comparison 2:  KL_loss model across X_di, = @x124, @x160, @x192, @x224
#
###############################

model = models[0]
model_name = model_names[0]

# model = kcv.K_PCVAE
# model_name = "K_PCVAE"
klws = [1,2,3,5,10]
latent_dims = [32,40,64]
pix_dims = [128,160,192]
epochs = 400



# finished everything but the biggest KL_WEIHT


# loop over pix_dim
for pix_dim in pix_dims:
    ds = get_ds(pix_dim)
    n_samples = get_n_samples(ds)
    params['x_dim']=(pix_dim,pix_dim,3)
    for kl in klws:
        params['kl_weight'] = kl
        for l in latent_dims:

            params['z_dim'] = l
            data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
            print(data_dir)

            vae,hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
            snk2loss,snk2vec = create_snk2dicts(ds,n_samples,vae)

            dump_snk2pickles(snk2loss,snk2vec,data_dir,params['kl_weight'])


#%%
ds = get_ds(pix_dim)
n_samples = get_n_samples(ds)

#%%
latent_dim = 64
pix_dim = 160
batch_size = 64
kl_weight = 5

# model = kcv.K_PCVAE
# model_name = "K_PCVAE"
klws = [1,2,3,5,10]
latent_dims = [32,40,64]
pix_dims = [128,160,192]
epochs = 400


p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))


#loop over models
models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]
model = models[1]
model_name = model_names[0]


pix_dim = pix_dims[2]
params['x_dim']=(pix_dim,pix_dim,3)
kl = klws[4]
params['kl_weight'] = kl
l = latent_dims[2]
params['z_dim'] = l
data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
print(data_dir)
#vae,hist = load_and_prime_model(model,model_name,params,data_dir,epochs)

#%%
snk2loss, snk2vec = load_snk2pickles(data_dir,params['kl_weight'])


desc_df = pd.read_pickle(f"{cf.DESCRIPTIONS}.pkl")
meta_df = pd.read_pickle(f"{cf.META_DATA}.pkl")


# fix the root directory....
#%%
#############################
#
#  make comparison... for each network compare the UMAP...
#
###############################
latents = np.array(list(snk2vec.values()))
losses = np.array(list(snk2loss.values()))
labels = np.array(list(snk2vec.keys()))
reducer = umap.UMAP(random_state=42)
reducer.fit(latents)

embedding = reducer.transform(latents)

assert(np.all(embedding == reducer.embedding_))
embedding.shape

plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
#plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of VAE ', fontsize=24)
plt.savefig("media/umap_small.png")




latents = np.array(list(snk2vec.values()))
losses = np.array(list(snk2loss.values()))
labels = np.array(list(snk2vec.keys()))
reducer = umap.UMAP(random_state=42)
reducer.fit(latents)

embedding = reducer.transform(latents)

assert(np.all(embedding == reducer.embedding_))
embedding.shape


# %%   
#############################
#
#  comparison 1:  KL_Reg model vs KL_loss model @x160
#
###############################
latent_dim = 40
pix_dim = 160
batch_size = 64
kl_weight = 5

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))

ds = get_ds(pix_dim)
n_samples = get_n_samples(ds)

#loop over models
models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]
data_dirs = [f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}" for model_name in model_names]

klws = [1,2,3,5,10]
latents = [32,40,64]
epochs = 400

p_names = ['z_dim','x_dim','kl_weight','batch_size']
p_vals = [latent_dim, (pix_dim,pix_dim,3), kl_weight, batch_size]
params = dict(zip(p_names,p_vals))



for model,model_name in zip(models,model_names):
    for kl in klws:
        params['kl_weight'] = kl
        for l in latents:
            params['z_dim'] = l
            data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
            print(data_dir)
            # vae, hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
            # snk2loss,snk2vec = create_snk2dicts(ds,n_samples,vae)
            # dump_snk2pickles(snk2loss,snk2vec,data_dir,params['kl_weight'])
            snk2loss,snk2vec = load_snk2pickles(data_dir,params['kl_weight'])
            snk2umap = get_umap_embedding(snk2vec)
            dump_snk2umap(snk2umap,data_dir,kl_weight)   
#

#############################
#
#  comparison 2:  KL_loss model across X_di, = @x124, @x160, @x192, @x224
#
###############################

model = models[0]
model_name = model_names[0]
klws = [1,2,3,5,10]
latent_dims = [32,40,64]
pix_dims = [128,160,192]
epochs = 400


# loop over pix_dim
for pix_dim in pix_dims:
    ds = get_ds(pix_dim)
    n_samples = get_n_samples(ds)
    params['x_dim']=(pix_dim,pix_dim,3)
    for kl in klws:
        params['kl_weight'] = kl
        for l in latent_dims:
            params['z_dim'] = l
            data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
            print(data_dir)
            # vae, hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
            # snk2loss,snk2vec = create_snk2dicts(ds,n_samples,vae)
            # dump_snk2pickles(snk2loss,snk2vec,data_dir,params['kl_weight'])
            snk2loss,snk2vec = load_snk2pickles(data_dir,params['kl_weight'])
            snk2umap = get_umap_embedding(snk2vec)
            dump_snk2umap(snk2umap,data_dir,kl_weight) 

#%%


# %%
from io import BytesIO
from PIL import Image
import base64
def embeddable_image(label):
    # img_data = 255 - 15 * data.astype(np.uint8)
    # image = Image.fromarray(img_data, mode='L').resize((64, 64), Image.BICUBIC)
    # buffer = BytesIO()
    # image.save(buffer, format='jpg')
    # for_encoding = buffer.getvalue()
    # return 'data:image/png;base64,' + base64.b64encode(for_encoding).decode()
    
    # loads the image if string input

    #image = get_thumbnail(label)
    return image_formatter(label)
    # return 'data:image/jpeg;base64,' + base64.b64encode(for_encoding).decode()


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((64, 64), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f"data:image/png;base64,{image_base64(im)}"
    #return f'<img src="data:image/jpeg;base64,{image_base64(im)}">'


#%%

from bokeh.plotting import figure, show, output_notebook
from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
from bokeh.palettes import Spectral10

output_notebook()

digits_df = pd.DataFrame(embedding, columns=('x', 'y'))
digits_df['digit'] = [str(x.decode()) for x in labels]
digits_df['image'] = digits_df.digit.map(lambda f: embeddable_image(f))
digits_df['fname'] = digits_df.digit.map(lambda x: f"{x.split('/')[-3]} {x.split('/')[-1]}")
digits_df['db'] = digits_df.digit.map(lambda x: f"{x.split('/')[-3]}")
digits_df['loss'] = [f"{x:.1f}" for x in losses]

datasource = ColumnDataSource(digits_df)
color_mapping = CategoricalColorMapper(factors=["sns","goat"],
                                       palette=Spectral10)

plot_figure = figure(
    title='UMAP projection VAE latent',
    plot_width=1000,
    plot_height=1000,
    tools=('pan, wheel_zoom, reset')
)

plot_figure.add_tools(HoverTool(tooltips="""
<div>
    <div>
        <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
    </div>
    <div>
        <span style='font-size: 14px'>@fname</span>
        <span style='font-size: 14px'>@loss</span>
    </div>
</div>
"""))

plot_figure.circle(
    'x',
    'y',
    source=datasource,
    color=dict(field='db', transform=color_mapping),
    line_alpha=0.6,
    fill_alpha=0.6,
    size=4
)

show(plot_figure)
 












#%%%
vecs = np.array(list(snk2vec.values()))

labs = np.array(list(snk2vec.keys()))

reducer = umap.UMAP(random_state = 488)

embedding = reducer.transform(vecs)

reducer.fit(vecs)

plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
#plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)



#%%


snk2loss, snk2vec = load_snk2pickles(data_dir)


dfdesc = pd.read_pickle(f"{cf.DESCRIPTIONS}.pkl")
df_meta = pd.read_pickle(f"{cf.META_DATA}.pkl")
# fix the root directory....


digits_df = pd.DataFrame.from_dict(snk2vec, orient='index')

#

pix_dim = 160
batch_size = 64
epochs = 400
kl_weight = 5
latent_dim = 40


models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]




data_dirs = [f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}" for model_name in model_names]





params['kl_weight'] = kl_weight
params['z_dim'] = latent_dim

print(f"training beta={kl_weight} z={latent_dim}")
data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
print(data_dir)

model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = f"{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"

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



#%%

df_meta.head()






# UMAP(a=None, angular_rp_forest=False, b=None,
#      force_approximation_algorithm=False, init='spectral', learning_rate=1.0,
#      local_connectivity=1.0, low_memory=False, metric='euclidean',
#      metric_kwds=None, min_dist=0.1, n_components=2, n_epochs=None,
#      n_neighbors=15, negative_sample_rate=5, output_metric='euclidean',
#      output_metric_kwds=None, random_state=42, repulsion_strength=1.0,
#      set_op_mix_ratio=1.0, spread=1.0, target_metric='categorical',
#      target_metric_kwds=None, target_n_neighbors=-1, target_weight=0.5,
#      transform_queue_size=4.0, transform_seed=42, unique=False, verbose=False)
            # convert to arrays

latents = np.array(list(snk2vec.values()))
losses = np.array(list(snk2loss.values()))
labels = np.array(list(snk2vec.keys()))
reducer = umap.UMAP(random_state=42)
reducer.fit(latents)




embedding = reducer.transform(digits_df.to_numpy())
# Verify that the result of calling transform is
# idenitical to accessing the embedding_ attribute
assert(np.all(embedding == reducer.embedding_))
embedding.shape


plt.scatter(embedding[:, 0], embedding[:, 1], cmap='Spectral', s=5)
#plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.gca().set_aspect('equal', 'datalim')
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.title('UMAP projection of the Digits dataset', fontsize=24)



#%%
# Importing Image module from PIL package  
from PIL import Image  
import PIL  
  






    # if tf.io.is_jpeg(filename):
    #     image_string = tf.io.read_file(filename)
    #     im = tf.image.decode_jpeg(image_string, channels=3)
    #     im = tf.cast(im, tf.float32) / 255.0
    # else: 
    #     #print("not a jpeg")
    #     image_string = tf.io.read_file(filename)
    #     im = tf.image.decode_png(image_string, channels=0)  #  
    #     mask = tf.cast(tf.expand_dims(im[:,:,3],2),tf.float32) / 255.
    #     mask2 = tf.concat((mask,mask,mask),axis=2)
    #     im = tf.cast(im, tf.float32) / 255.0
    #     im = (1.-mask2) + mask2*im[:,:,0:3]

    # label = filename
    # return im, label





# %% 
# this is to fix the PNGs which convert to black backgrounds...
# imoved all the files into "im" and then copied back to the root dir...
files_n = glob.glob(os.path.join(cf.IMAGE_FILEPATH, "goat/im/*"))
out_root = '/home/ergonyc/Projects/DATABASE/SnkrScrpr/data/goat/img'
skip_list = []
its = 0
for f in files:
    #im = load_and_convert_to_dump(f)
    im = Image.open(f)
    its += 1
    outf = os.path.join(out_root,f.split('/')[-1])
    if im.format != 'JPEG':
        print(f"not JPEG>>>>  {im.format}")
        skip_list.append(f)
        #im.thumbnail(im.size, Image.LANCZOS)
        bg = Image.new("RGB", im.size, (255,255,255))
        bg.paste(im,box=(0,0),mask=im.convert('RGBA').split()[-1])
        im = bg


    im.save(outf, "JPEG")

    # PIL.Image.fromarray(im.numpy()).save(outf)

# save a image using extension 
ut.dump_pickle("goat_png_fix.pkl",skip_list)

# %% 
# this is to fix big border on the edges compared to the goat database
# imoved all the files into "im" and then copied back to the root dir...
files = glob.glob(os.path.join(cf.IMAGE_FILEPATH, "sns/im/*"))
out_root = '/home/ergonyc/Projects/DATABASE/SnkrScrpr/data/sns/img'

skip_list = []
its = 0
for f in files:
    #im = load_and_convert_to_dump(f)
    im = Image.open(f)
    its += 1
    outf = os.path.join(out_root,f.split('/')[-1])
    if im.format != 'JPEG':
        print(f"not JPEG>>>>  {im.format}")        
        skip_list.append(f)

        # #im.thumbnail(im.size, Image.LANCZOS)
        # bg = Image.new("RGB", im.size, (255,255,255))
        # bg.paste(im,box=(0,0),mask=im.convert('RGBA').split()[-1])
        # im = bg
    # lets take 10% pixels off
    dp = 100//12    
    wd, ht = im.size
    dy, dx = wd//dp, ht//dp
    crop_bx = (dx, dx, wd-dx, ht-dx)
    im_crop = im.crop(crop_bx)
    im_crop.save(outf, "JPEG")

    # PIL.Image.fromarray(im.numpy()).save(outf)

# save a image using extension 



# %%

ds = get_ds(pix_dim)
n_samples = get_n_samples(ds)

db = create_imgStats(ds,n_samples)

sn = db['sns']
gt = db['goat']
plt.imshow(sn.std(axis=0).squeeze())
plt.colorbar()

plt.imshow(gt.std(axis=0).squeeze())
plt.colorbar()

# %%
