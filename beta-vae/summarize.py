##################
# part 1:  choose the best model... beta ~ 2 or 4
#        how to i choose the best beta?  use the disentangled metrics???
#          or just visualize the latent space
# part 2:   visualize the latent space
# part 3:   create umap / tsne summaries...
# part 4:  create a streamlit app for visualizing the manifold...
# part 4... how to explore the latent space

#%%

import utils as ut
import logger
import configs as cf
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import random

import tensorflow as tf

import cvae as cv
import utils as ut

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
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#
cf_kl_weight = cf.KL_WEIGHT
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

#%% # LOAD & PREPROCESS the from list of filessudo apt install gnome-tweak-tool
# could simplify this by making another "load_prep_batch_data(train_data,imagesize,augment=True,)"
train_dataset = ut.load_prep_and_batch_data(train_data, cf_img_size, cf_batch_size, augment=True)
test_dataset =  ut.load_prep_and_batch_data(  val_data, cf_img_size, cf_batch_size, augment=False)

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


#%% %%%%%
HOME = "/home/ergonyc"   
HOME = cf.HOME 

ROOT_DIR = cf.IMGRUN_DIR.rstrip("imgruns/")

# load the data files
def load_models_and_loss(p):
    """
    load the models and loss so we can find the best fit.. 

    input: params = dict (root_dir, img_run_id, txt_run_id, img_size,kl_weight, ..)
    """

    img_root = f"{p['root_dir']}/imgruns/{p['img_run_id']}/"
    txt_root = f"{p['root_dir']}/txtruns/{p['txt_run_id']}/"

    print(img_root)
    print(txt_root)

    epoch_mx = 0
    for filename in glob.glob(os.path.join(img_root, 'saved_data/losses*.pkl')):
        epochs = (filename.split(sep="_")[-1].rstrip(".pkl")) #split returns a list...
        if int(epochs) > epoch_mx:
            loss_file = filename
            epoch_mx = epochs
        
    print(loss_file)
    # loops over matching filenames in all subdirectories of `directory`.
    train,test = ut.load_pickle(os.path.join(img_root, f"saved_data/losses_{epoch_mx}.pkl"))
    #  need to also impute the steps...
    dstep_test = int(len(train)/len(test))

    snk2vec = ut.load_pickle(os.path.join(img_root,"snk2vec.pkl"))
    snk2loss = ut.load_pickle(os.path.join(img_root,"snk2loss.pkl"))
    return train,test,dstep_test,snk2vec, snk2loss



# load the data files
def plot_imgrun_loss(p):
    """
    load the models and loss so we can find the best fit.. 

    input: params = dict (root_dir, img_run_id, txt_run_id, img_size,kl_weight, ..)
    """

    img_root = f"{p['root_dir']}/imgruns/{p['img_run_id']}/"

    print(img_root)

    epoch_mx = 0
    for filename in glob.glob(os.path.join(img_root, 'saved_data/losses*.pkl')):
        epochs = (filename.split(sep="_")[-1].rstrip(".pkl")) #split returns a list...
        if int(epochs) > epoch_mx:
            loss_file = filename
            epoch_mx = int(epochs)
        
    print(loss_file)
    # loops over matching filenames in all subdirectories of `directory`.
    train,test = ut.load_pickle(os.path.join(img_root, f"saved_data/losses_{epoch_mx}.pkl"))

    print(len(train))
    print(len(test))
    e_train = range(0,len(train))
    e_test = range(0,len(train),int(len(train)/len(test)))





#%%  %%%%%%%%%

# # train_data = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'), allow_pickle=True)
# # val_data = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'), allow_pickle=True)
# # all_data = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'), allow_pickle=True)

# # text
# train_data = np.load(os.path.join(cf.DATA_DIR, 'train_txt_data.npy'), allow_pickle=True)
# val_data = np.load(os.path.join(cf.DATA_DIR, 'val_txt_data.npy'), allow_pickle=True)
# all_data = np.load(os.path.join(cf.DATA_DIR, 'all_txt_data.npy'), allow_pickle=True)

# id_label = np.load(os.path.join(cf.DATA_DIR,'mnp.npy'))  #IDs (filenames)
# descriptions = np.load(os.path.join(cf.DATA_DIR,'dnp.npy')) #description
# description_vectors = np.load(os.path.join(cf.DATA_DIR,'vnp.npy')) #vectors encoded
# padded_encoded_vector = np.load(os.path.join(cf.DATA_DIR,'pnp.npy')) #padded encoded

# ut.load_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"),curr_losses)


#%%  %%%%%%%%%

img_run_id = "1003-1046"   
img_size = 224  
kl_weight = 0.25
txt_run_id = "1003-1159" 
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)

#%% %%%%%%%%
img_run_id = "1001-2040"   
img_size = 224  
kl_weight = 0.5
txt_run_id = "1001-2155"  
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)
#%%  %%%%%%%%%


img_run_id = "1001-1510"   
img_size = 224  
kl_weight = 1.0
txt_run_id = "1001-1830"  

params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)

#train,test,ds,vecs,losss = load_models_and_loss(params)
#%%  %%%%%%%%%


img_run_id = "1002-0844"   
img_size = 224  
kl_weight = 2.0
txt_run_id = "1002-0957"
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)

#%%  %%%%%%%%%

#img_run_id = "1005-1010" # txtmodel probably run with against 1004-1938 
#img_size = 224 
#kl_weight = 2.
# 1005-2334 

img_run_id = "1007-1709"  
img_size = 224  
kl_weight = 2.0  # with more cvae epichs... (overfitting)
txt_run_id = "1007-1858"
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)

#%%  %%%%%%%%%



img_run_id = "1002-1244"   
img_size = 224  
kl_weight = 4.0
txt_run_id = "1002-1634"
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)
#%%  %%%%%%%%%

img_run_id = "1002-1833"   
img_size = 224  
kl_weight = 8.0
txt_run_id = "1002-2117" 
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)
#%%  %%%%%%%%%

img_run_id = "1002-2316"   
img_size = 224  
kl_weight = 16.0
txt_run_id = "1003-0847" 
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)

#%%  %%%%%%%%%

img_run_id = "1004-1938"  
img_size = 224 
kl_weight = 32.
txt_run_id = "1004-2334" 
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)



#%%  %%%%%%%%%

img_run_id = "1014-0001"  
img_size = 224 
kl_weight = 1176.
txt_run_id = "1014-0939" 
params = dict(zip(["root_dir","img_run_id","txt_run_id","img_size","kl_weight"],[ROOT_DIR,img_run_id,txt_run_id,img_size,kl_weight]))
plot_imgrun_loss(params)




# %%

x = train_samples 
model.load_model("imgruns/1014-0001",600)
kl_weight = 2.0


x = train_samples 
model.load_model("imgruns/1014-0001",660)
beta_norm = kl_weight #changed convention


mean, logvar = model.encode(train_samples)
z = model.reparameterize(mean, logvar)
x_logit = model.decode(z)


#x_samp = model.sample()

ut.show_reconstruct(model,train_samples)

cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x) 
logpx_z = -tf.math.reduce_sum(cross_ent, axis=[1, 2, 3]) 

logpz = model.log_normal_pdf(z, 0.0, 0.0)
logqz_x = model.log_normal_pdf(z, mean, logvar)


kl_divergence = logqz_x - logpz
neg_log_likelihood = -logpx_z

#elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)  # shape=()
elbo = tf.math.reduce_mean(-beta_norm* kl_divergence/cf_latent_dim - neg_log_likelihood/(cf_img_size*cf_img_size*3))  # shape=()

elbo


# betas = [.1*x**2 for x in range(0,100)]
# elbos = [-tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood) for kl_weight in betas]
# elbo = elbos[0]



#neg_ll, kl_div = model.get_test_loss_parts(train_samples)
# %%

# %%
