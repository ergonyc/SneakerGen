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


#%%
importlib.reload(cv) # works

#%% ####################   check that the model is okay   ####################

# vae = cv.PCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
#                  learning_rate=0.0001, beta=cf_beta, training=False)


vae1 = cv.PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)


vae2 = cv.PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001)


for test_samples, test_labels in test_dataset.take(1): 
    pass
for train_samples, train_labels in train_dataset.take(1): 
    pass

#%%  Run the Training loops


vae1 = cv.PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)


vae2 = cv.PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001)



#%%


# epochs = 4
# betas = np.linspace(1,1.1*(cf_pixel_dim/cf_latent_dim),num=15).round(decimals=0)

# betas = [.1, .5, 1, 2, 3, 4, 5, 8, 16, 25, 32, 50, 64, 100, 150, 256, 400, 750, 1000,1500, 2000, 2750]
# #beta_norm = ((cf_latent_dim/cf_pixel_dim)*betas).round(decimals=3)

# # def kl_loss(y_true, y_pred):
# #     kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
# #     return kl_loss

# # def nll_loss(y_true, y_pred):
# #     return LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

# # vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [nll_loss, kl_loss])

# betas = [1,5]

# beta = 1.

# vae = cv.PCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
#             learning_rate=cf_learning_rate, beta=beta, training=True)
        
# vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))#, loss=[tf.keras.losses.mse])

#%%
x = test_samples
#%%
importlib.reload(cv) # works


vae1 = cv.PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)


vae2 = cv.PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001)


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


#%%
import kcvae as kcv 

#%%
importlib.reload(kcv) # works



def make_dir(dirname):
    if os.path.isdir(dirname):
        return
    else:
        os.mkdir(dirname)

#%%

vae1 = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)

vae4 = kcv.K_PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)



#%%
epochs = 10


vae3 = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)

vae4 = kcv.K_PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)


vae3.compile(optimizer=vae3.optimizer, loss = vae3.partial_vae_loss)

            
train_history = vae3.fit(train_dataset,epochs=epochs, 
                        verbose=1, validation_data=test_dataset)
                        #, initial_epoch = 11 )

history = train_history.history


# def kl_loss(y_true, y_pred):
#     kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
#     return kl_loss

# def nll_loss(y_true, y_pred):
#     return LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

# vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [nll_loss, kl_loss])

#%%
epochs = 20
betas = [1]

for beta in betas:
    
    if beta >= 1:
        beta_str = f"{int(beta):04d}"
    else:
        beta_str = f"{beta:.1f}"


    vae = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=1, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    ut.dump_pickle(os.path.join("data2",f"KCVAE_history_{beta_str}.pkl"), (history,betas,epochs))
    sv_path = os.path.join("data2",f"{beta_str}")
    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)


    # vae = kcv.K_PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
    #              learning_rate=0.0001, kl_weight=beta)
    # vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
    # train_history = vae.fit(train_dataset,epochs=epochs, 
    #                         verbose=1, validation_data=test_dataset)
    #                         #, initial_epoch = 11 )
    # history = train_history.history
    # ut.dump_pickle(os.path.join("data3",f"KCVAE_reg_history_{beta_str}.pkl"), (history,betas,epochs))
    # sv_path = os.path.join("data3",f"{beta_str}")
    # make_dir(sv_path)
    # print('save model')
    # vae.save_model(sv_path, epochs)


#%%


betas = [.1, .5, 1, 2, 3, 4, 5, 8,16,32]
epochs = 250
for beta in betas:
    
    if beta > 1:
        beta_str = f"{int(beta):04d}"
    else:
        beta_str = f"{beta:.1f}"


    vae = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=cf_beta)
    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    # always start from the "warmed up beta=1, 20 epochs weights"
    sv_path = os.path.join("data2","0001")
    vae.load_model(sv_path, 20)  
    
    
    vae = kcv.K_PCVAE(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
                 learning_rate=0.0001, kl_weight=beta)

    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)

    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=1, validation_data=test_dataset)
                            #, initial_epoch = 11 )

    history = train_history.history
    ut.dump_pickle(os.path.join("data2",f"KCVAE_history_{beta_str}.pkl"), (history,betas,epochs))
    sv_path = os.path.join("data2",f"{beta_str}")
    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)




    # vae = kcv.K_PCVAE_KL_Reg(dim_z=cf_latent_dim, dim_x=(cf_img_size,cf_img_size,3), 
    #              learning_rate=0.0001, kl_weight=beta)
    # vae.compile(optimizer=vae3.optimizer, loss = vae3.partial_vae_loss)
    # train_history = vae.fit(train_dataset,epochs=epochs, 
    #                         verbose=1, validation_data=test_dataset)
    #                         #, initial_epoch = 11 )
    # history = train_history.history
    # ut.dump_pickle(os.path.join("data3",f"KCVAE_history_{beta_str}.pkl"), (history,betas,epochs))
    # sv_path = os.path.join("data3",f"{beta_str}")
    # make_dir(sv_path)
    # print('save model')
    # vae.save_model(sv_path, epochs)









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
            
train_history = vae.fit(train_dataset,epochs=epochs, 
                        verbose=1, validation_data=test_dataset)
                        #, initial_epoch = 11 )


history = train_history.history




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


def make_dir(dirname):
    if os.path.isdir(dirname):
        return
    else:
        os.mkdir(dirname)




#%% 
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







epochs = 4
betas = np.linspace(1,1.1*(cf_pixel_dim/cf_latent_dim),num=15).round(decimals=0)

betas = [.1, .5, 1, 2, 3, 4, 5, 8]
#beta_norm = ((cf_latent_dim/cf_pixel_dim)*betas).round(decimals=3)

# def kl_loss(y_true, y_pred):
#     kl_loss =  -0.5 * K.sum(1 + log_var - K.square(mean_mu) - K.exp(log_var), axis = 1)
#     return kl_loss

# def nll_loss(y_true, y_pred):
#     return LOSS_FACTOR*r_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

# vae_model.compile(optimizer=adam_optimizer, loss = total_loss, metrics = [nll_loss, kl_loss])

for beta in betas:
    
    if beta > 1:
        beta_str = f"{int(beta):04d}"
    else:
        beta_str = f"{beta:.1f}"

    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                learning_rate=cf_learning_rate, beta=beta, training=True)
            
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))#, loss=[tf.keras.losses.mse])

                
    train_history = vae.fit(train_dataset,epochs=epochs, 
                            verbose=1, validation_data=test_dataset)
                            #, initial_epoch = 11 )


    history = train_history.history


    fig, axs = plt.subplots(nrows=3,sharex=True, sharey=False, gridspec_kw={'hspace': 0})
    fig.set_size_inches(16, 24)
    fig.suptitle(f"beta={beta} x loss")
    axs[0].plot(history['elbo'][1:])
    axs[0].set_autoscaley_on(b=True)
    axs[0].plot(history['val_elbo'][1:])
    axs[0].set(yscale='log',ylabel='ELBO loss')#, ylim=(3*10**4,1*10**5))
    axs[1].plot(history['nll'][1:])
    axs[1].set_autoscaley_on(b=True)
    axs[1].plot(history['val_nll'][1:])
    axs[1].set(yscale='log',ylabel='neg log liklihood')#, ylim=(3*10**4,1*10**5))
    axs[1].legend(["train","test"], title=f"beta={beta_str}", loc='upper left')

    axs[2].plot(history['kl'][1:])
    axs[2].set_autoscaley_on(b=True)
    axs[2].plot(history['val_kl'][1:])
    axs[2].set(ylabel='KL div')#,yscale='log', ylim=(5*10**1,5*10**2))
    fig.show()   



    ut.dump_pickle(os.path.join("data",f"history_{beta_str}.pkl"), (history,betas,epochs))
    sv_path = os.path.join("data",f"{beta_str}")
    make_dir(sv_path)
    print('save model')
    vae.save_model(sv_path, epochs)

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
epochs = 350
test_interval = 10
betas = np.linspace(1,1.1*(cf_pixel_dim/cf_latent_dim),num=15).round(decimals=0)
#beta_norm = ((cf_latent_dim/cf_pixel_dim)*betas).round(decimals=3)


train_losses = []
test_losses = []
train_kls = []
test_kls = []
train_nlls = []
test_nlls = []

for beta in betas:

    train_loss=[]
    test_loss=[]
    train_kl = []
    test_kl = []
    train_nll = []
    test_nll = []
    # instantiate the model for the current beta
    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                 learning_rate=cf_learning_rate, beta=beta, training=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))






    test_epochs = []
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        vae.reset_metrics()
        batch_index = 1
        total_train_batchs = 63
        for train_x, train_label in train_dataset:
            results = vae.train_step(train_x)
            if epoch > 0:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}] Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            else:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}]  ")
            stdout.flush()  
            batch_index = batch_index + 1

        end_time = time.time()
        dt = end_time - start_time

        elbo = results['elbo'].numpy()
        kl = results['kl'].numpy()
        nll = results['nll'].numpy()

        # stdout.write("\n\rEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
        # stdout.flush()  
#        print(f"\nEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")

        #enerate_images(vae, test_sample)
        train_loss.append(elbo)
        train_kl.append(kl)
        train_nll.append(nll)

        ## TEST LOSSin chekmakedirs
        if (epoch+1) % test_interval == 0:
            start_time = time.time()
            vae.reset_metrics()
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                results = vae.test_step(test_x)

            end_time = time.time()
            dt = end_time - start_time
            elbo = results['elbo'].numpy()
            kl = results['kl'].numpy()
            nll = results['nll'].numpy()
            test_loss.append(elbo)
            test_kl.append(kl)
            test_nll.append(nll)
            test_epochs.append(epoch)
            stdout.write(f"\rTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            #print(f"\n\nTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            stdout.flush()

    plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta,epochs,test_epochs)


    ut.dump_pickle(os.path.join("data",f"loss_beta{beta}.pkl"), (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) )
    sv_path = os.path.join("data",f"{beta}")
    make_dir(sv_path)
    vae.save_model(sv_path, epoch)
    
    train_losses.append(train_loss)
    train_kls.append(train_kl)
    train_nlls.append(train_nll)
    test_losses.append(test_loss)
    test_kls.append(test_kl)
    test_nlls.append(test_nll)

ut.dump_pickle(os.path.join("data",f"losses_beta.pkl"), (train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) )
#%% 


#%% 

epochs = 320
test_interval = 10
#beta_norm = [.25*2**x for x in range(0,13)]
beta_norm = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 64.0, 128.0, 256.0, 512.0]

train_losses = []
test_losses = []
train_kls = []
test_kls = []
train_nlls = []
test_nlls = []

for b in beta:

    train_loss=[]
    test_loss=[]
    train_kl = []
    test_kl = []
    train_nll = []
    test_nll = []

    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                 learning_rate=cf_learning_rate, beta=b, training=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))

    test_epochs = []
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        vae.reset_metrics()
        batch_index = 0
        total_train_batchs = 63
        for train_x, train_label in train_dataset:
            results = vae.train_step(train_x)
            if epoch > 0:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}] Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            else:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}]  ")
            stdout.flush()  
            batch_index += 1
            
        end_time = time.time()
        dt = end_time - start_time

        elbo = results['elbo'].numpy()
        kl = results['kl'].numpy()
        nll = results['nll'].numpy()

        # stdout.write("\n\rEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
        # stdout.flush()  
#        print(f"\nEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")

        #enerate_images(vae, test_sample)
        train_loss.append(elbo)
        train_kl.append(kl)
        train_nll.append(nll)

        ## TEST LOSSin chekmakedirs
        if (epoch+1) % test_interval == 0:
            start_time = time.time()
            vae.reset_metrics()
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                results = vae.train_step(test_x)

            end_time = time.time()
            dt = end_time - start_time
            elbo = results['elbo'].numpy()
            kl = results['kl'].numpy()
            nll = results['nll'].numpy()
            test_loss.append(elbo)
            test_kl.append(kl)
            test_nll.append(nll)
            test_epochs.append(epoch)
            stdout.write(f"\rTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            #print(f"\n\nTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            stdout.flush()

    plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta,epochs,test_epochs)


    ut.dump_pickle(os.path.join("data",f"loss_beta{beta}.pkl"), (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) )
    sv_path = os.path.join("data",f"{beta}")
    make_dir(sv_path)
    vae.save_model(sv_path, epoch)
    
    train_losses.append(train_loss)
    train_kls.append(train_kl)
    train_nlls.append(train_nll)
    test_losses.append(test_loss)
    test_kls.append(test_kl)
    test_nlls.append(test_nll)

ut.dump_pickle(os.path.join("data",f"losses_beta2.pkl"), (train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) )

#%%

epochs = 400
test_interval = 10
beta_norm = [.25*2**x for x in range(-2,10)]

#beta_norm = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 64.0, 128.0, 256.0, 512.0]

train_losses = []
test_losses = []
train_kls = []
test_kls = []
train_nlls = []
test_nlls = []

for beta in beta_norm:

    train_loss=[]
    test_loss=[]
    train_kl = []
    test_kl = []
    train_nll = []
    test_nll = []

    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                 learning_rate=cf_learning_rate, beta_norm=beta, training=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))

    test_epochs = []
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        vae.reset_metrics()
        batch_index = 0
        total_train_batchs = 63
        for train_x, train_label in train_dataset:
            results = vae.train_step(train_x)
            if batch_index > 0:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}] Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            else:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}]  ")
            stdout.flush()
            batch_index += 1

        end_time = time.time()
        dt = end_time - start_time

        elbo = results['elbo'].numpy()
        kl = results['kl'].numpy()
        nll = results['nll'].numpy()

        # stdout.write("\n\rEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
        # stdout.flush()  
#        print(f"\nEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")

        #enerate_images(vae, test_sample)
        train_loss.append(elbo)
        train_kl.append(kl)
        train_nll.append(nll)

        ## TEST LOSSin chekmakedirs
        if (epoch+1) % test_interval == 0:
            start_time = time.time()
            vae.reset_metrics()
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                results = vae.train_step(test_x)

            end_time = time.time()
            dt = end_time - start_time
            elbo = results['elbo'].numpy()
            kl = results['kl'].numpy()
            nll = results['nll'].numpy()
            test_loss.append(elbo)
            test_kl.append(kl)
            test_nll.append(nll)
            test_epochs.append(epoch)
            stdout.write(f"\rTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            #print(f"\n\nTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            stdout.flush()

    plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta,epochs,test_epochs)


    ut.dump_pickle(os.path.join("data2",f"loss_beta{beta}.pkl"), (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) )
    sv_path = os.path.join("data2",f"{beta}")
    make_dir(sv_path)
    vae.save_model(sv_path, epoch)
    
    train_losses.append(train_loss)
    train_kls.append(train_kl)
    train_nlls.append(train_nll)
    test_losses.append(test_loss)
    test_kls.append(test_kl)
    test_nlls.append(test_nll)

ut.dump_pickle(os.path.join("data2",f"losses_beta2.pkl"), (train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) )

#%%


#%%

epochs = 300
test_interval = 10
beta_norm = [.9*1.25**x for x in range(0,9)]
#beta_norm = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 64.0, 128.0, 256.0, 512.0]

train_losses = []
test_losses = []
train_kls = []
test_kls = []
train_nlls = []
test_nlls = []

for beta in beta_norm:

    train_loss=[]
    test_loss=[]
    train_kl = []
    test_kl = []
    train_nll = []
    test_nll = []

    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                 learning_rate=cf_learning_rate, beta_norm=beta, training=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))
    # start from the trained beta_norm = 1.0
    sv_path = os.path.join("data2",f"{1.0}")
    make_dir(sv_path)
    vae.load_model(sv_path, 399)
    

    test_epochs = []
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        vae.reset_metrics()
        batch_index = 0
        total_train_batchs = 63
        for train_x, train_label in train_dataset:
            results = vae.train_step(train_x)
            if epoch > 0:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}] Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            else:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}]  ")
            stdout.flush()
            batch_index += 1

        end_time = time.time()
        dt = end_time - start_time

        elbo = results['elbo'].numpy()
        kl = results['kl'].numpy()
        nll = results['nll'].numpy()

        # stdout.write("\n\rEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
        # stdout.flush()  
#        print(f"\nEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")

        #enerate_images(vae, test_sample)
        train_loss.append(elbo)
        train_kl.append(kl)
        train_nll.append(nll)

        ## TEST LOSSin chekmakedirs
        if (epoch+1) % test_interval == 0:
            start_time = time.time()
            vae.reset_metrics()
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                results = vae.train_step(test_x)

            end_time = time.time()
            dt = end_time - start_time
            elbo = results['elbo'].numpy()
            kl = results['kl'].numpy()
            nll = results['nll'].numpy()
            test_loss.append(elbo)
            test_kl.append(kl)
            test_nll.append(nll)
            test_epochs.append(epoch)
            stdout.write(f"\rTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            #print(f"\n\nTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            stdout.flush()

    plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta,epochs,test_epochs)


    ut.dump_pickle(os.path.join("data2",f"loss_beta{beta}.pkl"), (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) )
    sv_path = os.path.join("data2",f"{beta}")
    make_dir(sv_path)
    vae.save_model(sv_path, epoch)
    
    train_losses.append(train_loss)
    train_kls.append(train_kl)
    train_nlls.append(train_nll)
    test_losses.append(test_loss)
    test_kls.append(test_kl)
    test_nlls.append(test_nll)

ut.dump_pickle(os.path.join("data2",f"losses_beta4.pkl"), (train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) )

#%%



#%%

epochs = 500
test_interval = 10
beta_norm = list(.01*np.round(100*np.linspace(1.5,5.,num=6)))
#beta_norm = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 64.0, 128.0, 256.0, 512.0]

train_losses = []
test_losses = []
train_kls = []
test_kls = []
train_nlls = []
test_nlls = []

for beta in beta_norm:

    train_loss=[]
    test_loss=[]
    train_kl = []
    test_kl = []
    train_nll = []
    test_nll = []

    vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
                 learning_rate=cf_learning_rate, beta_norm=beta, training=True)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))
    # # start from the trained beta_norm = 1.0
    # sv_path = os.path.join("data2",f"{1.0}")
    # make_dir(sv_path)
    # vae.load_model(sv_path, 399)
    

    test_epochs = []
    for epoch in tqdm(range(epochs)):
        start_time = time.time()
        vae.reset_metrics()
        batch_index = 0
        total_train_batchs = 63
        for train_x, train_label in train_dataset:
            results = vae.train_step(train_x)
            if batch_index > 0:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}] Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            else:
                stdout.write(f"\r[{batch_index:3d}/{total_train_batchs:3d}]  ")
            stdout.flush()
            batch_index += 1

        end_time = time.time()
        dt = end_time - start_time

        elbo = results['elbo'].numpy()
        kl = results['kl'].numpy()
        nll = results['nll'].numpy()

        # stdout.write("\n\rEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
        # stdout.flush()  
#        print(f"\nEpoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")

        #enerate_images(vae, test_sample)
        train_loss.append(elbo)
        train_kl.append(kl)
        train_nll.append(nll)

        ## TEST LOSSin chekmakedirs
        if (epoch+1) % test_interval == 0:
            start_time = time.time()
            vae.reset_metrics()
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                results = vae.train_step(test_x)

            end_time = time.time()
            dt = end_time - start_time
            elbo = results['elbo'].numpy()
            kl = results['kl'].numpy()
            nll = results['nll'].numpy()
            test_loss.append(elbo)
            test_kl.append(kl)
            test_nll.append(nll)
            test_epochs.append(epoch)
            stdout.write(f"\rTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            #print(f"\n\nTEST->   Epoch: {epoch}, Train set ELBO: {elbo:04f} | kl-{kl:04f} | nll-{nll:04f} | time elapse for current epoch: {dt:02f}")
            stdout.flush()

    plot_imgrun_loss(train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,beta,epochs,test_epochs)


    ut.dump_pickle(os.path.join("data2",f"loss_beta{beta}.pkl"), (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) )
    sv_path = os.path.join("data2",f"{beta}")
    make_dir(sv_path)
    vae.save_model(sv_path, epoch)
    
    train_losses.append(train_loss)
    train_kls.append(train_kl)
    train_nlls.append(train_nll)
    test_losses.append(test_loss)
    test_kls.append(test_kl)
    test_nlls.append(test_nll)

ut.dump_pickle(os.path.join("data2",f"losses_beta5.pkl"), (train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) )

#%% 
epochs = 320
test_interval = 10
#beta_norm = [.25*2**x for x in range(0,13)]
beta_norm = [0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0, 25.0, 64.0, 128.0, 256.0, 512.0]

for beta in beta_norm[1:]:

    (train_loss,test_loss,train_kl,test_kl, train_nll, test_nll,test_epochs) =  ut.load_pickle(os.path.join("data",f"loss_beta{beta}.pkl"), )
    sv_path = os.path.join("data",f"{beta}")
     #vae.save_model(sv_path, epoch)
    

(train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,test_epochs,beta_norm) = \
                    ut.load_pickle(os.path.join("data",f"losses_beta2.pkl"))




plot_runXbeta(train_losses,test_losses,train_kls,test_kls, train_nlls, test_nlls,beta_norm,epochs,test_epochs)



#%% 
# custom loss function with tf.nn.sigmoid_cross_entropy_with_logits
# def custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits):
#     raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
#                                             labels=x_true, logits=x_recons_logits)
#     neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
#     return tf.math.reduce_mean(neg_log_likelihood/cf_pixel_dim)

vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate))
es=tf.keras.callbacks.EarlyStopping(patience=10)

train_history = vae.fit(train_dataset, epochs=epochs, 
                        verbose=1, validation_data=test_dataset)#, shuffle=True, callbacks=[es])



#%% Define Training methods
def step_model(epochs, display_interval=-1, save_interval=10, test_interval=10, current_losses=([],[])) :
    """
    custom training loops to enable dumping images of the progress
    """

    model.training=False
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
            neg_ll, kl_div = model.get_test_loss_parts(train_x)
            
            loss_batch = neg_ll+kl_div

            #neg_elbo = tf.math.reduce_mean(self.kl_weight *


            losses.append(loss_batch)
            stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
            stdout.flush()  

            batch_index = batch_index + 1

        ## TRAIN LOSS
        elbo = np.mean(losses)
        print(f'Epoch: {lg.total_epochs}   Train loss: {float(elbo):.1f}   Epoch Time: {float(time.time()-start_time):.2f}')
        lg.log_metric(elbo, 'train loss',test=False)
        elbo_train.append(elbo)

        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            if epoch == 1:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)    
            else:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=False, save_fig=True, limits=cf_limits)

        ## TEST LOSSin chekmakedirs
        test_losses = []
        for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
            #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
            test_cost_batch = model.compute_test_loss(test_x)  # this should turn off the dropout...
            test_losses.append(test_cost_batch)

        test_loss = np.mean(test_losses)
        print(f'   TEST LOSS  : {test_loss:.1f}    for epoch: {lg.total_epochs}')
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





def train_model(epochs, display_interval=-1, save_interval=10, test_interval=10,current_losses=([],[])) :
    """
    custom training loops to enable dumping images of the progress
    """
    print('\n\nStarting training...\n')
    model.training=True
    elbo_train,elbo_test = current_losses
    if len(elbo_test)>0:
        print(f"test: n={len(elbo_test)}, last={elbo_test[-1]}")
        print(f"train: n={len(elbo_train)}, last={elbo_train[-1]}")

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        losses = []
        batch_index = 1

        # DO THE AUGMENTATION HERE...
        for train_x, _ in train_dataset :
        #for train_x, label in train_dataset :
            #train_x = tf.cast(train_x, dtype=tf.float32)
            loss_batch = model.trainStep(train_x)
            losses.append(loss_batch)
            stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
            stdout.flush()  

            batch_index = batch_index + 1

        ## TRAIN LOSS
        elbo = np.mean(losses)
        print(f'Epoch: {lg.total_epochs}   Train loss: {float(elbo):.1f}   Epoch Time: {float(time.time()-start_time):.2f}')
        lg.log_metric(elbo, 'train loss',test=False)
        elbo_train.append(elbo)

        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            if epoch == 1:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)    
            else:
                ut.show_reconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=False, save_fig=True, limits=cf_limits)

        ## TEST LOSSin chekmakedirs
        if epoch % test_interval == 0:
            test_losses = []
            for test_x, test_label in test_dataset: # (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
                #test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
                test_cost_batch = model.compute_test_loss(test_x)  # this should turn off the dropout...
                test_losses.append(test_cost_batch)

            test_loss = np.mean(test_losses)
            print(f'   TEST LOSS  : {test_loss:.1f}    for epoch: {lg.total_epochs}')
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


#%% lets pick apart our loss/cost
# we already have our samples
# train_samples, train_labels in train_dataset.take(1) : pass
# test_samples, test_labels in test_dataset.take(1) : pass


#%%

 





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

cf_root_dir = lg.root_dir  #make sure we log this
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
