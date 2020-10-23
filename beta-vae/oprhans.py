
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

import betavae as cv
import utils as ut
import logger
import configs as cf

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



#%% helpers


def log_normal_pdf(sample, mean, logvar):
    log2pi = np.log(2.0 * np.pi)
    
    out = np.sum(-0.5 * ((sample - mean) ** 2.0 * np.exp(-logvar) + logvar + log2pi), axis=1)
    return out

def compute_loss(x,model):
    """
    TODO: change to compute_test_cost
        this is actually "cost" not loss since its across the batch...
    """

    mu, logvar = model.encode(x)
    z = model.reparameterize(mu, logvar)
    x_recons_logits = model.decode(z)

    z = model.encode(x)
    # monte-carlo aproximation of KL-Div with one sample...(or batch samples?)
    logpz = log_normal_pdf(z, 0.0, 0.0)
    logqz_x = log_normal_pdf(z, mu, logvar)
    kl_divergence = logqz_x - logpz
    return kl_divergence

#%%





betas,training_metrics, testing_metrics = ut.load_pickle(os.path.join("data",f"train_test_metricsXbeta.pkl"))
#betas = [.1, .5, 1, 2, 3, 4, 5, 8, 16, 25, 32, 50, 64, 100, 150, 256, 400, 750, 1000,1500, 2000, 2750]
beta_str = [f"{int(beta):04d}" if b>=1 else f"{beta:.1f}" for b in betas]

curr_idx = 8
b = betas[curr_idx]
b_str = beta_str[curr_idx]
history,_betas,epochs = ut.load_pickle(os.path.join("data",f"history_{b_str}.pkl"))
sv_path = os.path.join("data",f"{b_str}")
    
vae = cv.BCVAE(latent_dim=cf_latent_dim, input_dim=cf_img_size, 
            learning_rate=cf_learning_rate, beta=beta, training=False)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cf_learning_rate), 
         loss=vae.custom_sigmoid_cross_entropy_loss_with_logits)
vae.load_model(sv_path, epochs)



for test_samples, test_labels in test_dataset.take(1): 
    pass
for train_samples, train_labels in train_dataset.take(1): 
    pass

example_images = test_samples[:10,]

def plot_compare_vae(images=None):
    if images is None:
        for test_samples, test_labels in test_dataset.take(1) : pass
        images = test_samples[:10]

    n_to_show = images.shape[0]
    reconst_images = vae.reconstruct(images, training=False)
    # mu, logvar = vae.encode(images)
    # z = vae.reparameterize(mu, logvar)
    # reconst_images = vae.decode(z)

    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = images[i].numpy().squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

    for i in range(n_to_show):
        img = reconst_images[i].numpy().squeeze()
        sub = fig.add_subplot(2, n_to_show, i+n_to_show+1)
        sub.axis('off')
        sub.imshow(img)  

plot_compare_vae(images = example_images)      



def vae_generate_images(n_to_show=10):
    rsample = tf.cast(np.random.lognormal(0,1,size=(n_to_show,vae.latent_dim)), dtype=tf.float32)
    reconst_images = vae.decode(rsample,apply_sigmoid = True)
    fig = plt.figure(figsize=(15, 3))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    for i in range(n_to_show):
        img = reconst_images[i].numpy().squeeze()
        sub = fig.add_subplot(2, n_to_show, i+1)
        sub.axis('off')        
        sub.imshow(img)

vae_generate_images(n_to_show=10)        


from scipy.stats import norm

mu, logvar = vae.encode(test_samples)
z = vae.reparameterize(mu, logvar)

def lreparameterize(mu, logvar):
    eps = tf.random.normal(shape=mu.shape)
    #sig = tf.math.exp(0.5*logvar) 
    #log_sig = logvar*0.5  #sqrt
    return mu + eps * tf.exp(logvar * 0.5)
lz = lreparameterize(mu, logvar)


z_test = lz.numpy()
z_test = vae.encode(test_samples,reparam=True).numpy()



x = np.linspace(-3, 3, 300)

fig = plt.figure(figsize=(20, 20))
fig.subplots_adjust(hspace=0.6, wspace=0.4)

for i in range(32):
    ax = fig.add_subplot(5, 10, i+1)
    ax.hist(z_test[i,:], density=True, bins = 20)
    ax.axis('off')
    ax.text(0.5, -0.35, str(i), fontsize=10, ha='center', transform=ax.transAxes)
    ax.plot(x,norm.pdf(x))

plt.show()






#%% Logger class
class SimpleLogger: 
    def __init__(self, run_name="", root_dir=None, trainMode=False, txtMode=False):
        self.remote = cf.REMOTE
        self.training = trainMode or self.remote
        self.total_epochs = 1
        self.run_name = run_name

        self.new_run = root_dir is None
        if self.new_run:
            self.root_dir = cf.IMGRUN_DIR
            self.root_dir = os.path.join(self.root_dir, ut.add_time_stamp())
        else:
            self.root_dir = root_dir

        if txtMode:
            if self.root_dir.split("/")[-2] == "txtruns":
                print(f"root_dir already txtruns= {self.root_dir}")
            else:
                self.root_dir = self.root_dir.replace("imgruns", "txtruns")
                print(f"changed to text runs:root_dir= {self.root_dir}")

        self.img_in_dir = cf.IMAGE_FILEPATH
        self.plot_out_dir = os.path.join(self.root_dir, "plots/")
        self.model_saves = os.path.join(self.root_dir, "models/")
        self.tblogs = os.path.join(self.root_dir, "logs")
        self.test_writer = None
        self.train_writer = None

        # for writing test and validate pkl record of dataset
        self.saved_data = os.path.join( self.root_dir, "saved_data")  
        self.update_plot_dir()
        

    def write_config(self, variables, code):
        self.check_make_dirs()
        if not self.training:
            print("Cannot, in read only mode.")
            return

        if len(code) > 0:
            code_file = open(os.path.join(self.root_dir, "code_file.txt"), "w")
            for source in code:
                code_file.write(repr(source) + "\n\n")
                code_file.write(inspect.getsource(source) + "\n\n")
            code_file.close()

        filtered_vars = {key: value for (key, value) in variables.items() if (key.startswith("cf_"))}
        w = csv.writer(open(os.path.join(self.root_dir, "configs.csv"), "w"))
        for key, val in filtered_vars.items():
            w.writerow([key, val])



    def check_make_dirs(self):
        def make_dir(dirname):
            if os.path.isdir(dirname):
                return
            else:
                os.mkdir(dirname)

        make_dir(self.root_dir)
        make_dir(ut.plot_out_dir)
        make_dir(self.model_saves)
        make_dir(self.tblogs)
        make_dir(self.saved_data)
        #print(f"made {self.saved_data}")
        #self.setup_writer()


    def setup_writer(self):
        ##  need to make tensorflow logs
        if self.test_writer is None:
            train_log_dir = self.tblogs + '/train'
            test_log_dir = self.tblogs + '/test'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            self.test_writer = test_summary_writer
            self.train_writer = train_summary_writer


    def reset(self, total_epochs=1):
        self.total_epochs = total_epochs

    def log_metric(self, metric, name, test=False):
        self.check_make_dirs()
        if test:
            summary_writer = self.test_writer
        else:
            summary_writer = self.train_writer

        with summary_writer.as_default():
            tf.summary.scalar(name=name,data=metric, step=self.total_epochs)


