'''
This file is used to train the shape autoencoder model.

It uses betacvae.py as the base model and many data functions from utils to make it simpler.
'''
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
import pcvae as cv

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
train_dataset = ut.load_prep_and_batch_data(train_data, cf_img_size, cf_batch_size, augment=True)
test_dataset =  ut.load_prep_and_batch_data(  val_data, cf_img_size, cf_batch_size, augment=False)
#%% ####################   load some data for easy access  ####################

for test_samples, test_labels in test_dataset.take(1): 
    pass
for train_samples, train_labels in train_dataset.take(1): 
    pass
x = test_samples

#%%
importlib.reload(cv) # works  pcvae.py

#%%
import kcvae as kcv 

#%%
importlib.reload(kcv) # works


#%%

def make_vae_baseline(model, model_name, data_dir):
    epochs = 20
    beta = 1 
    beta_str = f"{int(beta):04d}"
    make_dir(data_dir)
    
    vae = model(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=2, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    ut.dump_pickle(os.path.join(data_dir,f"history{model_name}_{beta_str}.pkl"), (history,betas,epochs))
    sv_path = os.path.join(data_dir,f"{beta_str}")
    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)

    return True
#%%

model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = "train-"+model_name
finished = make_vae_baseline( model, model_name, data_dir)

#%%


def get_vae_vs_beta(epochs, model, model_name, data_dir, betas):

    for beta in betas:

        if beta > 1:
            beta_str = f"{int(beta):04d}"
        else:
            beta_str = f"{beta:.1f}"


        vae = model(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                    learning_rate=0.0001, kl_weight=beta)
        #vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

        # always start from the "warmed up beta=1, 20 epochs weights"
        sv_path = os.path.join(data_dir,"0001")
        vae.load_model(sv_path, 20)  

        # vae = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
        #             learning_rate=0.0001, kl_weight=beta)

        vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

        train_history = vae.fit(train_dataset,epochs=epochs, 
                                verbose=0, validation_data=test_dataset)
                                #, initial_epoch = 11 )

        history = train_history.history
        ut.dump_pickle(os.path.join(data_dir,f"history{model_name}_{beta_str}.pkl"), (history,betas,epochs))
        sv_path = os.path.join(data_dir,f"{beta_str}")
        make_dir(sv_path)
        print('save model')
        vae.save_model(sv_path, epochs)


#%%

finished = True
model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = "train-"+model_name

betas = [.1, .5, 1, 2, 3, 4, 5, 8,16,32]
epochs = 220

if finished:
    finished = get_vae_vs_beta(epochs, model, model_name, data_dir, betas)
#%%

model = kcv.K_PCVAE_KL_Reg
model_name = "K_PCVAE_KL_Reg"
data_dir = "train-"+model_name
finished = make_vae_baseline(model, model_name, data_dir)
if finished:
    betas = [.25, .5, 1, 2, 3, 4, 5, 8, 16, 32, 50]
    epochs = 220
    finished = get_vae_vs_beta(epochs, model, model_name, data_dir, betas)


#%%


######################### make some plotting function s###########################



def collect_training_metrics(epochs, model, model_name, data_dir, betas, labels):
    training_metrics = []
    testing_metrics = []

    for beta in betas:
        
        if beta > 1:
            beta_str = f"{int(beta):04d}"
        else:
            beta_str = f"{beta:.1f}"

        strip_beginning = True
        
        history,_betas,epochs = ut.load_pickle(os.path.join(data_dir,f"history{model_name}_{beta_str}.pkl"))
        sv_path = os.path.join(data_dir,f"{beta_str}")


        # if strip_beginning:
        #     trn_ls = trn_ls[1:]
        #     trn_kl = trn_kl[1:]
        #     trn_nll = trn_nll[1:]
        #     trn_ep = trn_ep[1:]

        keys = history.keys()        
        metrics = [lab for lab in keys if not lab.startswith('val_')]
        test_metrics = [lab for lab in keys if lab.startswith('val_')]

        ylabels = dict(zip(metrics,labels))

        # metrics = ['elbo','nll','kl','kla']
        # test_metrics = ['val_elbo','val_nll','val_kl','val_kla']

        # #['elbo', 'kl', 'kla', 'nll', 'val_elbo', 'val_kl', 'val_kla', 'val_nll']
        # ylabels = {'elbo':'ELBO loss','nll':'-log(likelihood)','kl':'KL Divergence','kla':'KL Divergence2'}

        fig, axs = plt.subplots(nrows=4,sharex=True, sharey=False, gridspec_kw={'hspace': 0})
        fig.set_size_inches(16, 24)
        fig.suptitle(f"beta={beta} x loss")    

        trn_mets = []
        tst_mets = []
        ax_n = 0
        for tr,test in zip(metrics,test_metrics):

            train = history[tr]
            test = history[test]

            if strip_beginning:
                train = train[1:]
                test = test[1:]

            axs[ax_n].plot(train)
            axs[ax_n].set_autoscaley_on(b=True)
            axs[ax_n].plot(test)
            axs[ax_n].set(yscale='log',ylabel=ylabels[tr])
            trn_mets.append(train)
            tst_mets.append(test)
            ax_n += 1


        training_metrics.append(np.stack(trn_mets))
        testing_metrics.append(np.stack(tst_mets))

    # training = dict(zip(metrics,training_metrics))
    # testing = dict(zip(test_metrics,testing_metrics))

    ut.dump_pickle(os.path.join(data_dir,f"train_test_metricsXbeta.pkl"), (betas,training_metrics, testing_metrics) )
    return (betas,training_metrics, testing_metrics)





def collate_training_metrics(epochs, training_metrics, testing_metrics, model_name, data_dir, betas, labels):

    ### once we make sure everything works... we'll redo this with 250-300 epochs per run
    c =['tab:blue', 'tab:red', 'tab:cyan','tab:pink', 'tab:green','tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive'] 

    trn = np.stack(training_metrics,axis=2)
    tst = np.stack(testing_metrics,axis=2)


    fig, axs = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)
    fig.suptitle(f"beta x train metrics {model_name}")

    #keys = history.keys()
    # metrics = ['elbo','nll','kl','kla']
    # test_metrics = ['val_elbo','val_nll','val_kl','val_kla']
    
    #metrics = [lab for lab in keys if not lab.startswith('val_')]
    #test_metrics = [lab for lab in keys if lab.startswith('val_')]
    #['elbo', 'kl', 'kla', 'nll', 'val_elbo', 'val_kl', 'val_kla', 'val_nll']
    #labels = ['ELBO loss',]
    #
    ylabels = labels #dict(zip(metrics,labels))
    beta_str = [f"{int(b):03d}" if b>=1 else f"{b:.2f}" for b in betas]


    trn_mets = []
    tst_mets = []
    ax_n = 0
    for i in range(3):
        axs[i][0].plot(trn[i,].squeeze())
        axs[i][0].set_autoscaley_on(b=True)
        axs[i][0].set(yscale='log',ylabel=ylabels[i])
        axs[i][1].plot(tst[i,].squeeze())
        axs[i][1].set_ylim(axs[i][0].get_ylim())
        axs[i][1].set(yscale='log')
        if i==0:
            axs[i][0].set_title('Train')
            axs[i][1].set_title('Test')
        elif i==1:
            axs[i][1].legend(beta_str, title='beta_norm', loc='upper left')


    pic_name = os.path.join(data_dir,f"masterfig.png")
    _ = fig.savefig(pic_name)





#%%
model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = "train-"+model_name
betas = [.1, .5, 1, 2, 3, 4, 5, 8,16,32]
epochs = 220

labels = ['ELBO loss','-log(likelihood)','KL Divergence','KL Divergence2']
#labels = ['ELBO loss','KL Div reg','KL Div reg2']
res = collect_training_metrics(epochs, model, model_name, data_dir, betas,labels)

betas,training_metrics, testing_metrics = res
#%%
collate_training_metrics(epochs, training_metrics, testing_metrics, model_name, data_dir, betas, labels)


#%%
model = kcv.K_PCVAE_KL_Reg
model_name = "K_PCVAE_KL_Reg"
data_dir = "train-"+model_name
betas = [.25, .5, 1, 2, 3, 4, 5, 8, 16, 32, 50]
epochs = 220

#labels = ['ELBO loss','-log(likelihood)','KL Divergence','KL Divergence2']
labels = ['ELBO loss','KL Div reg','KL Div reg2']
res = collect_training_metrics(epochs, model, model_name, data_dir, betas,labels)

betas,training_metrics, testing_metrics = res
collate_training_metrics(epochs, training_metrics, testing_metrics, model_name, data_dir, betas, labels)

#%%
########################################3
#
#   VISUALIZE TOOLS
#
#################################

#%%
def preview_recons(x_batch,xhat_batch,xhat_probe,offset,title_txt):
    plt.figure(figsize=(12, 12))
    
            
    for i in range(4):
        plt.subplot(1, 4, i+1)
        plt.axis('Off')
        i_ = i+offset
        x = x_batch[i_,].numpy().squeeze()
        xhat = xhat_batch[i_,].numpy().squeeze()
        im = xhat_probe[i_,].numpy().squeeze()

        plt.title(title_txt+f" ex {i+offset}")
        plt.imshow(np.concatenate((x,xhat,im)))
    
#%%
#examine a batch
model = kcv.K_PCVAE
model_name = "K_PCVAE"
data_dir = "train-"+model_name
betas = [.1, .5, 1, 2, 3, 4, 5, 8,16,32]

for beta in betas:
    epochs = 220
    beta_str = f"{int(beta):04d}" if beta>1 else f"{beta:.1f}" 
    sv_path = os.path.join(data_dir,beta_str)


    vae = model(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                    learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
    vae.load_model(sv_path,epochs)  


    img_samples = next(iter(train_dataset))[0]

    x = img_samples
    xhat = vae(x)

    z = vae.encoder(x)
    assert isinstance(z, tfp.distributions.Distribution)

    z_samples = z.sample()

    mu = z.mean()
    sig = z.stddev()

    #%%
    sample_index = 0

    latent = z_samples.numpy()
    latent[:,0]=-3.
    latent = tf.convert_to_tensor(latent,dtype=tf.float32)
    #im = vae.decoder(tf.expand_dims(latent,axis=0)).numpy().squeeze()
    #im = vae.decoder(tf.expand_dims(latent,axis=0))
    im = tf.math.sigmoid(vae.decoder(latent))


    preview_recons(x,xhat,im,sample_index,beta_str)
#%%


#examine a batch
model = kcv.K_PCVAE_KL_Reg
model_name = "K_PCVAE_KL_Reg"
data_dir = "train-"+model_name
betas = [.25, .5, 1, 2, 3, 4, 5, 8, 16, 32, 50]

for beta in betas:
    epochs = 220
    beta_str = f"{int(beta):04d}" if beta>1 else f"{beta:.1f}" 
    sv_path = os.path.join(data_dir,beta_str)


    vae = model(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                    learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
    vae.load_model(sv_path,epochs)  


    img_samples = next(iter(train_dataset))[0]

    x = img_samples
    xhat = vae(x)

    z = vae.encoder(x)
    assert isinstance(z, tfp.distributions.Distribution)

    z_samples = z.sample()

    mu = z.mean()
    sig = z.stddev()

    #%%
    sample_index = 0

    latent = z_samples.numpy()
    latent[:,0]=-3.
    latent = tf.convert_to_tensor(latent,dtype=tf.float32)
    #im = vae.decoder(tf.expand_dims(latent,axis=0)).numpy().squeeze()
    #im = vae.decoder(tf.expand_dims(latent,axis=0))
    im = tf.math.sigmoid(vae.decoder(latent))


    preview_recons(x,xhat,im,sample_index,beta_str)

    gridimg= make_latent_im(x,xhat,z_samples,vae,sample_index)
    plt.figure(figsize=(400, 200))

    plt.imshow(gridimg)
    plt.show()

    
#%%



def make_latent_im(x,xhat,z_samples,vae,sample_index):

    n_steps = 13
    n = n_steps # figure with 20x20 samplesplt.subplot(3, 4, 1)
    digit_size = 192
    l = n_latents = 64

    grid_sig = np.linspace( -3, 3, n_steps)

    # decode for each square in the grid

    for i, yi in enumerate(grid_sig):
        gridcol = x[sample_index,:].numpy().squeeze()
        if i== 0:
            gridrow = xhat[sample_index,:].numpy().squeeze()

        for j in range(l): #latent_dims

            latent = z_samples[sample_index,:].numpy()
            latent[j]=yi
            latent = tf.convert_to_tensor(latent,dtype=tf.float32)


            #im = vae.decoder(tf.expand_dims(latent,axis=0)).numpy().squeeze()
            im = vae.decoder(tf.expand_dims(latent,axis=0))

            im = tf.sigmoid(im).numpy().squeeze()
            gridcol = np.vstack((gridcol,im))
            if i== 0:
                gridrow = np.vstack((gridrow,xhat[sample_index,:].numpy().squeeze()))


        if i == 0:
            gridimg = np.hstack((gridrow,gridcol))
        else:
            gridimg = np.hstack((gridimg,gridcol))
                             
    return gridimg                     






#%%

# LOOKS LIKE 
# beta = 2 or 3 is okay... maybe train longer at that value to get a "final"  model for further testing
# leaky relu and batch normalization vs.  dropout are an interesting option..
# 

# get a sample of images and create the latent

    

preview_recons(x,xhat,im,sample_index,beta_str)

#%%
n_steps = 13
n = n_steps # figure with 20x20 samplesplt.subplot(3, 4, 1)
digit_size = 192
l = n_latents = 64

grid_sig = np.linspace( -3, 3, n_steps)

# set sig to .0000001 (keep some noise)
# change each mu "sample" to the target 


#%%




# decode for each square in the grid

for i, yi in enumerate(grid_sig):
    gridcol = x[sample_index,:].numpy().squeeze()
    if i== 0:
        gridrow = tf.sigmoid(xhat[sample_index,:]).numpy().squeeze()

    for j in range(l): #latent_dims

        latent = z_samples[sample_index,:].numpy()
        latent[j]=yi
        latent = tf.convert_to_tensor(latent,dtype=tf.float32)


        #im = vae.decoder(tf.expand_dims(latent,axis=0)).numpy().squeeze()
        im = vae.decoder(tf.expand_dims(latent,axis=0))

        im = tf.sigmoid(im).numpy().squeeze()
        gridcol = np.vstack((gridcol,im))
        if i== 0:
            gridrow = np.vstack((gridrow,xhat[sample_index,:].numpy().squeeze()))


    if i == 0:
        gridimg = np.hstack((gridrow,gridcol))
    else:
        gridimg = np.hstack((gridimg,gridcol))
                             
                             
                             # def preview_recons(im_batch,rec_batch):
#     plt.figure(figsize=(12, 12))
#     plot_index = 0
#     for i in range(4): # dataset.take(12):
        
#         image = im_batch[i,].numpy().squeeze()
#         rec = rec_batch[i,].numpy().squeeze()
#         plot_index += 1
#         plt.subplot(3, 4, plot_index)
#         # plt.axis('Off')
#         #label = get_label_name(label.numpy())
#         #plt.title('Label: %s' % label)
#         plt.imshow(np.concatenate((image,rec)))
        
# # Explore raw training dataset images.

# preview_recons(train_batch,pred_batch)
    
#%%


plt.figure(figsize=(400, 200))

plt.imshow(gridimg)
plt.show()

#%%
half = gridimg.shape[0]//2

plt.figure(figsize=(40, 40))
plt.subplot(1,2,1)
plt.imshow(gridimg[:half,:,:])


plt.subplot(1,2,2)
plt.imshow(gridimg[half:,:,:])
plt.show()




#%%
model = kcv.K_PCVAE_KL_Reg
model_name = "K_PCVAE_KL_Reg"
data_dir = "train-"+model_name
betas = [.25, .5, 1, 2, 3, 4, 5, 8, 16, 32, 50]
epochs = 500
beta = 5

def overtrain_vae(model, model_name, data_dir,beta,epochs):

    beta_str = f"{int(beta):04d}"
    make_dir(data_dir)
    
    vae = model(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=0, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    ut.dump_pickle(os.path.join(data_dir,f"history{model_name}_{beta_str}.pkl"), (history,betas,epochs))
    sv_path = os.path.join(data_dir,f"{beta_str}")
    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)


#%%
# load the data files
def plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta_norm,epochs,test_epochs):
    """
    TODO: convert to np:  train =  np.array(train_loss)
        epochs = np.arange(0,np.shape), etc.
    """
    c =['tab:blue', 'tab:red', 'tab:cyan','tab:pink', 'tab:green','tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive'] 
    strip_beginning = True


    tst_ep=np.array(test_epochs)
    trn_ep=np.array(range(epochs))

    trn_ls=np.array(train_loss)
    tst_ls=np.array(test_loss)

    trn_kl=np.array(train_kl)
    tst_kl=np.array(test_kl)

    trn_nll=np.array(train_nll)
    tst_nll=np.array(test_nll)

    tst_ep=np.array(test_epochs)
    trn_ep=np.array(range(epochs))


    if strip_beginning:
        trn_ls = trn_ls[1:]
        trn_kl = trn_kl[1:]
        trn_nll = trn_nll[1:]
        trn_ep = trn_ep[1:]
    
    #plt.ylim((2.5*10**-1,3.5*10**4))
    fig, axs = plt.subplots(nrows=3,sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)
    fig.suptitle(f"beta={beta_norm} x loss")
    axs[0].plot(trn_ep,np.transpose(trn_ls))
    axs[0].set_autoscaley_on(b=True)
    axs[0].plot(tst_ep,np.transpose(tst_ls))
    axs[0].set(yscale='log')#, ylim=(2.6*10**-1,5*10**-1))

    #axs[0].plot([0,epochs,epochs, 0,0],[train_loss[-1],train_loss[-1],test_loss[-1],test_loss[-1],train_loss[-1]],'k:')
    
    axs[1].plot(trn_ep,np.transpose(trn_kl))
    axs[1].set_autoscaley_on(b=True)
    axs[1].plot(tst_ep,np.transpose(tst_kl))
    axs[1].legend(["train","test"], title='beta_norm', loc='upper left')
    axs[1].set(yscale='log')
    #axs[1].set( ylim=(5*10**-5,3*10**-1) )

    axs[2].plot(trn_ep,np.transpose(trn_nll))
    axs[2].set_autoscaley_on(b=True)
    axs[2].plot(tst_ep,np.transpose(tst_nll))
    axs[2].set(yscale='log')
    #axs[2].set( ylim=(2.2*10**-1,4*10**-1) )
    
 
    #plt.show()
    # for ax in axs.flat:
    #     #ax.set_autoscaley_on(b=True)
    #     #ax.set(xlim=(15,400))
    #     ax.set(yscale='log')
    pic_name = os.path.join("data",f"lossfig{beta_norm}.png")
    _ = fig.savefig(pic_name)

#%% 
def plot_runXbeta(train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,beta_norm,epochs,test_epochs):
    """
    TODO: convert to np:  train =  np.array(train_loss)
        epochs = np.arange(0,np.shape), etc.
    """
    c =['tab:blue', 'tab:red', 'tab:cyan','tab:pink', 'tab:green','tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive'] 

    tst_ep=np.array(test_epochs)
    trn_ep=np.array(range(epochs))

    trn_lss=np.array(train_losses)
    tst_lss=np.array(test_losses)

    trn_kls=np.array(train_kls)
    tst_kls=np.array(test_kls)

    trn_nlls=np.array(train_nlls)
    tst_nlls=np.array(test_nlls)

    tst_ep=np.array(test_epochs)
    trn_ep=np.array(range(epochs))


    fig, axs = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)
    fig.suptitle(f"beta={beta_norm} x loss")
    axs[0][0].plot(trn_ep,np.transpose(trn_lss))
    axs[0][1].plot(tst_ep,np.transpose(tst_lss))
    #axs[0].plot([0,epochs,epochs, 0,0],[train_loss[-1],train_loss[-1],test_loss[-1],test_loss[-1],train_loss[-1]],'k:')
    
    axs[1][0].plot(trn_ep,np.transpose(trn_kls))
    axs[1][1].plot(tst_ep,np.transpose(tst_kls))
    axs[1][1].legend([f"{b:2f}" for b in beta_norm], title='beta_norm', loc='upper left')

    axs[2][0].plot(trn_ep,np.transpose(trn_nlls))
    axs[2][1].plot(tst_ep,np.transpose(tst_nlls))
    
    #plt.show()
    for i, ax in enumerate(axs.flat):
        print(i)
        ax.set_autoscaley_on(b=True)
        #ax.set(xlim=(15,320))
        ax.set(yscale='log')
        # if i==0 or i ==1:
        #     ax.set(ylim=(2.6*10**-1,5*10**-1))
        # elif i==2 or i ==3:
        #     ax.set(ylim=(5*10**-5,3*10**-1))
        # elif i==4 or i ==5:
        #     ax.set(ylim=(2.2*10**-1,4*10**-1))

    pic_name = os.path.join("data",f"masterfig.png")
    _ = fig.savefig(pic_name)







#%%            visualize latent scraps below  
mse = tf.math.squared_difference(x,x_hat)
nl = lambda x: tf.math.log(1.+x)

vmin,vmax = np.log(np.finfo(float).eps),0-np.finfo(float).eps
mx_ = lambda x: x.numpy().squeeze().max()
mn_ = lambda x: x.numpy().squeeze().min()
mn_mx = lambda x: (x.numpy().squeeze().min(), x.numpy().squeeze().max())
mn_mx = lambda x: (mn_(x), mx_(x))

vmin,vmax = mn_mx(nl(x_hat))


# nl = lambda x: x  #do nothing
#vmin,vmax = -1,1

for i in range(2):
    fig, axs = plt.subplots(nrows=3,ncols=3, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)

    cmaps = ['Reds','Blues','Greens']
    for c in range(3):
        cmap = cmaps[c]
        ax =axs[c][0]
        vals = nl(x[i,:,:,c]).numpy().squeeze()
        pos = ax.imshow(vals,cmap=cmap, vmin=vmin, vmax=vmax,
                                interpolation='none') 
        #fig.colorbar(pos, ax=ax)


        ax =axs[c][1]
        vals = nl(x_hat[i,:,:,c]).numpy().squeeze()
        pos = ax.imshow(vals,cmap=cmap, vmin=vmin, vmax=vmax,
                                interpolation='none') 
        #fig.colorbar(pos, ax=ax)

        ax =axs[c][2]
        vals = nl(mse[i,:,:,c]).numpy().squeeze()
        pos = ax.imshow(vals,cmap=cmap, vmin=vmin, vmax=vmax,
                                interpolation='none') 
        fig.colorbar(pos, ax=ax)

    plt.show()


#%%

fig, (ax1, ax2, ax3) = plt.subplots(figsize=(13, 3), ncols=3)

# plot just the positive data and save the
# color "mappable" object returned by ax1.imshow
pos = ax1.imshow(Zpos, cmap='Blues', interpolation='none')

# add the colorbar using the figure's method,
# telling which mappable we're talking about and
# which axes object it should be near
fig.colorbar(pos, ax=ax1)

# repeat everything above for the negative data
neg = ax2.imshow(Zneg, cmap='Reds_r', interpolation='none')
fig.colorbar(neg, ax=ax2)














#%%
## check to see if i was even updating weights....


#%%

for epoch in range(epochs):
    start_time = time.time()
    loss_metric1.reset_states()
    loss_metric2.reset_states()
    kl_loss.reset_states()
    for train_x,_ in tqdm(train_dataset):
    
        
        
    end_time = time.time()
    elbo1 = loss_metric1.result()
    kl_div = kl_loss1.result()
    elbo2 = loss_metric2.result()

    #display.clear_output(wait=False)
    print('Epoch: {}, Train set ELBO1: {}:{}(), ELBO2: {}, time elapse for current epoch: {}'.format(
            epoch, elbo1,kl_div, elbo2, end_time - start_time))

    #generate_images(vae, test_sample)


#%%


x_input = x #[32, 192, 192, 3]

z = vae.encoder.conv_layer_0(x_input) #[32, 96, 96, 3]
z = vae.encoder.conv_layer_1(z) #[32, 48, 48, 32]
z = vae.encoder.conv_layer_2(z) #[32, 24, 24, 64]
z = vae.encoder.conv_layer_3(z) #[32, 12, 12, 128]
#z = vae.encoder.dropout_layer(z)
z = vae.encoder.conv_layer_4(z) #[32, 6, 6, 256])
z = vae.encoder.flatten_layer(z)# ([32, 9216]

z_ = vae.encoder.sampler(z)
z = vae.encoder.normalTFP(z_)
#%%
prior = vae.encoder.prior

lnormalTFP = tfp.layers.MultivariateNormalTriL(vae.encoder.dim_z,
                        activity_regularizer=tfp.layers.KLDivergenceRegularizer(prior))


x_output = vae.decoder.input_l(z) #[32, 64]
x_output = vae.decoder.reshape_layer(x_output) #[32, 1, 1, 64]
x_output = vae.decoder.conv_transpose_layer_start(x_output) #[32, 12, 12, 256])
x_output = vae.decoder.conv_transpose_layer_0(x_output)#[32, 24, 24, 128])
x_output = vae.decoder.conv_transpose_layer_1(x_output)#[32, 48, 48, 64])
x_output = vae.decoder.conv_transpose_layer_2(x_output) # [32, 96, 96,  32]
x_output = vae.decoder.conv_transpose_layer_3(x_output) #[32, 192, 192, 16]

#x_output = vae.decoder.dropout_layer(x_output)
x_output = vae.decoder.conv_transpose_layer_4(x_output) # [32, 1024, 1024, 3]



#%%

# model training
vae = VAE_MNIST(dim_z=latent_dim, learning_rate=lr, analytic_kl=True, kl_weight=kl_w)
loss_metric = tf.keras.metrics.Mean()
opt = tfk.optimizers.Adam(vae.learning_rate)

for epoch in range(epochs):
    start_time = time.time()
    for train_x in tqdm(train_dataset):
        train_step(train_x, vae, opt, loss_metric)
    end_time = time.time()
    elbo = -loss_metric.result()
    #display.clear_output(wait=False)
    print('Epoch: {}, Train set ELBO: {}, time elapse for current epoch: {}'.format(
            epoch, elbo, end_time - start_time))
    #generate_images(vae, test_sample)


#%%

for train_x, train_label in train_dataset.take(1):
    results = vae.train_step(train_x)
results
#%%

        z_sample, z_mu, z_logvar = vae.encode(x,reparam=True)

        # why is the negative in the reduce_mean?
        # kl_div_a =  - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) 
        #                                         - tf.math.square(mu) 
        #                                         - tf.math.square(sd),   axis=1)
        kl_div_a = - 0.5 * tf.math.reduce_sum(1 + z_logvar 
                                                - tf.math.square(z_mu) 
                                                - tf.math.exp(z_logvar), axis=1)

                                                
        x_recons = vae.decode(z_sample,apply_sigmoid=True)
        #x_logits = self.decode(z_sample)
        # z_mu, z_logvar = self.encode(x)

        # z = self.reparameterize(z_mu, z_logvar)
        # x_recons = self.decode(z,apply_sigmoid=True)
        
        # log_likelihood log normal is MSE
        # loss is [0, 255]
        # mse = 0.00392156862745098* tf.math.squared_difference(255.*x,255.*x_recons)# 0.00392156862745098 - 1/255.
        mse = tf.math.squared_difference(x,x_recons)

        # for images the neg LL is the MSE
        neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])

        # # compute reverse KL divergence, either analytically 
        # # MC KL:         # or through MC approximation with one sample
        # logpz = self.log_normal_pdf(z, 0., 0.) #standard lognormal: mu = 0. logvar=0.
        # logqz_x = self.log_normal_pdf(z, z_mu, z_logvar)
        # kl_div_mc = logqz_x - logpz
        
        # def normal_log_pdf(sample, mean, logvar, raxis=1):
        #     log2pi = tf.math.log(2. * np.pi)
        #     return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


        def analytic_kl(sample, mean, logvar, raxis=1):
            # log((qz||x)/pz = difference in the log of the gaussian PDF
            log2pi = tf.math.log(2. * np.pi)
            logpz = tf.reduce_sum( -.5 * ((sample*sample) + log2pi),axis=raxis)
            logqz_x = tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
            return logqz_x - logpz

        kl_div_mc = analytic_kl(z_sample, z_mu, z_logvar)  # shape=(batch_size,)
        

        # # analytic KL for representing SIGMA (log_sig)
        # # kl_div_a = - 0.5 * tf.math.reduce_sum(
        # #                                 1 + 0.5*z_logvar - tf.math.square(z_mu) - tf.math.exp(0.5*z_logvar), axis=1)
        # # KL for representing the VARIANCE
        # kl_div_a = - 0.5 * tf.math.reduce_sum(
        #                                 1 + z_logvar - tf.math.square(z_mu) - tf.math.exp(z_logvar), axis=1)

        elbo = tf.math.reduce_mean(-vae.beta * kl_div_a - neg_log_likelihood)  # shape=()
        kl = tf.math.reduce_mean(kl_div_mc)  # shape=()
        nll = tf.math.reduce_mean(neg_log_likelihood)  # shape=()
        kla = tf.math.reduce_mean(kl_div_a)  # shape=()




plt.imshow(x[20,].numpy().squeeze())
plt.imshow(x_recons[20,].numpy().squeeze())
plt.imshow(mse[20,].numpy().squeeze())

cost_mini_batch = -elbo

#%%

for train_x, train_label in train_dataset.take(1):
    results = vae.train_step(train_x)
results
#%%



def make_dir(dirname):
    if os.path.isdir(dirname):
        return
    else:
        os.mkdir(dirname)









#%%

betas = [.1, .5, 1, 2, 3, 4, 5, 8, 16, 25, 32, 50, 64, 100, 150, 256, 400, 750, 1000,1500, 2000, 2750]


training_metrics = []
testing_metrics = []

for beta in betas:
    
    if beta >= 1:
        beta_str = f"{int(beta):04d}"
    else:
        beta_str = f"{beta:.1f}"

    strip_beginning = True
    
    history,_betas,epochs = ut.load_pickle(os.path.join("data",f"history_{beta_str}.pkl"))
    sv_path = os.path.join("data",f"{beta_str}")

    #vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
    #             learning_rate=cf_learning_rate, beta=beta, training=True)
    #vae.load_model(sv_path, epochs)  
    # vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate), 
    #         loss=vae.custom_sigmoid_cross_entropy_loss_with_logits)

    # if strip_beginning:
    #     trn_ls = trn_ls[1:]
    #     trn_kl = trn_kl[1:]
    #     trn_nll = trn_nll[1:]
    #     trn_ep = trn_ep[1:]
    metrics = ['elbo','nll','kl']
    test_metrics = ['val_elbo','val_nll','val_kl']


    ylabels = {'elbo':'ELBO loss','nll':'-log(likelihood)','kl':'KL Divergence'}

    fig, axs = plt.subplots(nrows=3,sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)
    fig.suptitle(f"beta={beta} x loss")    

    trn_mets = []
    tst_mets = []
    ax_n = 0
    for tr,test in zip(metrics,test_metrics):

        train = history[tr]
        test = history[test]

        if strip_beginning:
            train = train[1:]
            test = test[1:]

        axs[ax_n].plot(train)
        axs[ax_n].set_autoscaley_on(b=True)
        axs[ax_n].plot(test)
        axs[ax_n].set(yscale='log',ylabel=ylabels[tr])
        trn_mets.append(train)
        tst_mets.append(test)
        ax_n += 1


    training_metrics.append(np.stack(trn_mets))
    testing_metrics.append(np.stack(tst_mets))

ut.dump_pickle(os.path.join("data",f"train_test_metricsXbeta.pkl"), (betas,training_metrics, testing_metrics) )

### once we make sure everything works... we'll redo this with 250-300 epochs per run
c =['tab:blue', 'tab:red', 'tab:cyan','tab:pink', 'tab:green','tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:gray', 'tab:olive'] 

trn = np.stack(training_metrics,axis=2)
tst = np.stack(testing_metrics,axis=2)


fig, axs = plt.subplots(nrows=3,ncols=2, sharex=True, sharey=False, gridspec_kw={'hspace': 0})
fig.set_size_inches(16, 24)
fig.suptitle(f"beta x train metrics")



ylabels = ['ELBO loss','-log(likelihood)','KL Divergence']
beta_str = [f"{int(b):03d} ({b*cf_latent_dim/cf_pixel_dim:.4f}" if b>=1 else f"{betas[0]:.2f} ({betas[0]*cf_latent_dim/cf_pixel_dim:.6f}" for b in betas]


trn_mets = []
tst_mets = []
ax_n = 0
for i in range(3):
    axs[i][0].plot(trn[i,].squeeze())
    axs[i][0].set_autoscaley_on(b=True)
    axs[i][0].set(yscale='log',ylabel=ylabels[i])
    axs[i][1].plot(tst[i,].squeeze())
    axs[i][1].set_ylim(axs[i][0].get_ylim())
    axs[i][1].set(yscale='log')
    if i==0:
        axs[i][0].set_title('Train')
        axs[i][1].set_title('Test')
    elif i==1:
        axs[i][1].legend(beta_str, title='beta_norm', loc='upper left')


pic_name = os.path.join("data",f"masterfig2.png")
_ = fig.savefig(pic_name)



### then visualize each latent space to assess ...

# then retrain and fit the text generator


# then make the stramlit tool


#%% 





#%%  Run the Training loops

#%%
# model training
kl_loss1 = tf.keras.metrics.Mean()

loss_metric1 = tf.keras.metrics.Mean()
loss_metric2 = tf.keras.metrics.Mean()

opt1 = tf.keras.optimizers.Adam(vae1.learning_rate)
opt2 = tf.keras.optimizers.Adam(vae2.learning_rate)


#%%
epochs = 10
for epoch in range(epochs):
    start_time = time.time()
    loss_metric1.reset_states()
    loss_metric2.reset_states()
    kl_loss.reset_states()
    for train_x,_ in tqdm(train_dataset):
        cv.train_step(train_x, vae1, opt1, loss_metric1,kl_loss1)
        cv.train_step_KL_Reg(train_x, vae2, opt2, loss_metric2)
        
    end_time = time.time()
    elbo1 = loss_metric1.result()
    kl_div = kl_loss1.result()
    elbo2 = loss_metric2.result()

    #display.clear_output(wait=False)
    print('Epoch: {}, Train set ELBO1: {}:{}(), ELBO2: {}, time elapse for current epoch: {}'.format(
            epoch, elbo1,kl_div, elbo2, end_time - start_time))

    #generate_images(vae, test_sample)




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
