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
#import skimage.measure as sm
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

%reload_ext autoreload
%autoreload 2

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
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)

# cf_limits=[cf_img_size, cf_img_size, cf_img_size]
cf_limits=[cf_img_size, cf_img_size]
#ut.readMeta()
dfmeta = ut.readMeta()

#%% model viz function
# can i make this a model function??
def plotIO(model):
    print("\n Net Summary (input then output):")
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


#%% Make model and print info
model = cv.CVAE(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)
### instance of model used in GOAT blog
#model = cv.CVAE_EF(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)

model.setLR(cf_learning_rate)
model.printMSums()
model.printIO()
plotIO(model.enc_model)
plotIO(model.gen_model)

#%% Setup logger info
train_from_scratch = True
if train_from_scratch :
    lg = logger.logger(trainMode=cf.REMOTE, txtMode=False)
    lg = logger.logger(trainMode=True, txtMode=False)
else :
    shp_run_id = '0821-2318'  
    root_dir = os.path.join(cf.IMG_RUN_DIR, shp_run_id)
    lg = logger.logger(root_dir=root_dir, txtMode=False)

lg.setupCP(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer)
lg.restoreCP() # actuall reads in the  weights...



#%% define splitShuffleData

def splitShuffleData(files, test_split):
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
img_in_dir = lg.img_in_dir
#img_in_dir = "/Users/ergonyc/Projects/DATABASE/SnkrScrpr/data/"
files = glob.glob(os.path.join(img_in_dir, "*/img/*"))
#shuffle the dataset (GOAT+SNS)
files = np.asarray(files)


train_data, val_data, all_data = splitShuffleData(files,VAL_FRAC)

# ## Save base train data to file
np.save(os.path.join(cf.DATA_DIR, 'train_data.npy'), train_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'val_data.npy'), val_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'all_data.npy'), all_data, allow_pickle=True)


#%% Load base train data from file
## Save base train data to file
train_dat = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'))
val_dat = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'))
all_dat = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'))



#%% # LOAD & PREPROCESS the from list of files


train_dataset = tf.data.Dataset.from_tensor_slices(train_dat)
test_dataset = tf.data.Dataset.from_tensor_slices(val_dat)

train_dataset = ut.loadAndPrepData(cf_img_size, train_dataset, augment=True)
test_dataset = ut.loadAndPrepData(cf_img_size,  test_dataset, augment=False)

def batchAndPrepDataset(dataset):
    dataset = dataset.batch(cf_batch_size, drop_remainder=False)  #this might mess stuff up....
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

train_dataset = batchAndPrepDataset(train_dataset)
test_dataset = batchAndPrepDataset(test_dataset)


#%% # save etl.data to disk ... (to keep the partitioning) 

# #path = os.path.join(cf.IMG_RUN_DIR, "saved_data","train")
# # #save training data
# # tf.data.experimental.save(
# #     train_dataset, path, compression=None, shard_func=None
# # )
# ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","train.pkl"), train_dataset)


# #save testing data
# #path = os.path.join(cf.IMG_RUN_DIR, "saved_data","test")
# # tf.data.experimental.save(
# #     test_dataset, path, compression=None, shard_func=None
# # )
# ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","test.pkl"), test_dataset)


#%% Load all data
# why do we need to do this??? to get a list of samples??
for train_samples, train_labels in train_dataset.take(1) : pass
for test_samples, test_labels in test_dataset.take(1) : pass

#test_samples = tf.cast(test_samples, dtype=tf.float32)

#%% Load all data get number of batches... 

total_train_batchs = 0
for _ in train_dataset :
    total_train_batchs += 1


#%% Setup datasets
sample_index = 1
for sample_index in range(0,20):
    ut.plotImg(train_samples[sample_index], title='train', threshold=0.5, limits=cf_limits, save_fig=False)
    ut.plotImg(test_samples[sample_index], title='test', threshold=0.5, limits=cf_limits, save_fig=False)
#%% Show initial models

if (lg.total_epochs > 10) :
    ut.plotVox(model.reconstruct(test_samples[sample_index][None,...], training=False), limits=cf_limits, title='Recon')

sample_index = 17
ut.plotImg(train_samples[sample_index], title='train', threshold=0.5, limits=cf_limits, save_fig=False)
ut.plotImg(test_samples[sample_index], title='test', threshold=0.5, limits=cf_limits, save_fig=False)

#%% Training methods
def getTestSetLoss(dataset, batches=0) :
    test_losses = []
    for test_x, test_label in (dataset.take(batches).shuffle(100) if batches > 0 else dataset.shuffle(100)) :
        test_x = tf.cast(test_x, dtype=tf.float32) #might not need this
        test_loss_batch = model.compute_loss(test_x)
        test_losses.append(test_loss_batch)
    return np.mean(test_losses)


def trainModel(epochs, display_interval=-1, save_interval=10, test_interval=10) :
    print('\n\nStarting training...\n')
    model.training=True
    loss_batch2 = []
    loss_batchN = []
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

        elbo = np.mean(losses)

        if ((display_interval > 0) & (epoch % display_interval == 0)) :
            ut.showReconstruct(model, test_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=True, limits=cf_limits)

        if epoch % test_interval == 0:
            #test_loss2 = getTestSetLoss(test_dataset, 2)  # what should I do here??/ batch of 2???  shouldn't it be batch size??
            test_loss = getTestSetLoss(test_dataset, cf_batch_size)  # what should I do here??/ batch of 2???  shouldn't it be batch size??
            print('   TEST LOSS  : {:.1f}    for epoch: {}'.format(test_loss, lg.total_epochs))
            #print('   TEST LOSS 2: {:.1f}    for epoch: {}'.format(test_loss2, lg.total_epochs))
            lg.logMetric(test_loss, 'test loss')
            loss_batch2.append(test_loss2)
            loss_batchN.append(test_loss)

        if epoch % save_interval == 0:
            lg.cpSave()

        print('Epoch: {}   Train loss: {:.1f}   Epoch Time: {:.2f}'.format(lg.total_epochs, float(elbo), float(time.time() - start_time)))
        lg.logMetric(elbo, 'train loss')
        lg.incrementEpoch()
        if (ut.checkStopSignal(dir_path=cf.IMG_RUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")

    return epoch,loss_batchN #(loss_batch2,loss_batchN)


#%% Training data save
start_time = time.time()
losses = []
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

print('Epoch Time: {:.2f}'.format( float(time.time() - start_time)))

#path = os.path.join(cf.IMG_RUN_DIR, "saved_data","train")
# #save training data
# tf.data.experimental.save(
#     train_dataset, path, compression=None, shard_func=None
# )
ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","train_agumented.pkl"), (trainimgs,trainlabs) )


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

ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","test.pkl"), (testimgs,testlabs) )
#%% log Config...
lg.writeConfig(locals(), [cv.CVAE, cv.CVAE.__init__])
lg.updatePlotDir()

#%% 
n_epochs = 555
total_epochs = 0
epoch_n,loss_batches = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n

if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
model.saveMyModel(lg.root_dir, lg.total_epochs )

ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","losses.pkl"), loss_batches )



#%% 

#for test_samples, test_labels in train_dataset.take(1) : pass
for train_samples, train_labels in train_dataset.take(1) : pass

#%% 
sample_index = 1

for sample_index in range(32):
    ut.showReconstruct(model, train_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=False, limits=cf_limits)

#%% 

for test_samples, test_labels in test_dataset.take(1) : pass
#for train_samples, train_labels in train_dataset.take(1) : pass new environmentx=sample_index, show_original=True, save_fig=False, limits=cf_limits)


###########################
############################
#
#  Now make some easy access databases...
#
############################
###########################
#%% 

# ut.makeGifFromDir(gif_in_dir, name):

# model.saveMyModel(lg.root_dir, 138)
# #%% 

# model.loadMyModel(lg.root_dir,669)
# # Need to make methods to extract the pictures 

#%% Run model on all data to get latent vects and loss. Used for streamlit app and other places.
#preds,losses = ut.dumpReconstruct( model, train_dataset, test_dataset )
ds = ut.loadAndDump(cf_img_size, lg.img_in_dir)
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

ut.dumpPickle(os.path.join(lg.root_dir,"snk2vec.pkl"), snk2vec)
ut.dumpPickle(os.path.join(lg.root_dir,"snk2loss.pkl"), snk2loss)




#################
#################
#################
#################
#################
#################
#################


#%% Methods for exploring the design space and getting a feel for model performance 
def getRecons(num_to_get=10, isTest=True) :
    model.training = False
    if isTest:
        dataset = test_dataset
    else:
        dataset = train_dataset
    anchors, labels = [],[]
    for anchor, label in dataset.unbatch().shuffle(100000).take(num_to_get) :
        anchor = tf.cast(anchor, dtype=tf.float32)
        anchors.append(anchor)
        labels.append(label)

    anchor_vects = [model.encode(anchors[i][None,...], reparam=True) for i in range(len(anchors))]
    v = [model.sample(anchor_vects[i]).numpy()[...] for i in range (len(anchors))]


    for i, sample in enumerate(v) :
        print('Index: {}   Mid: {}'.format(i, labels[i].numpy().decode()))
        ut.plotImg(anchors[i].numpy(), threshold=None, title='Index {} Original'.format(i), limits=cf_limits)
        ut.plotImg(v[i],threshold=None, title='Index {} Reconstruct'.format(i), limits=cf_limits) 

    print([mid.numpy().decode() for mid in labels])
    return anchor_vects, labels

def interpolateDesigns(anchor_vects, labels, index1, index2, divs=10) :

    mids_string = ' {} , {} '.format(labels[index1].numpy().decode(), labels[index2].numpy().decode())
    print(mids_string)
    interp_vects = ut.interp(anchor_vects[index1].numpy(), anchor_vects[index2].numpy(), divs)

    v = model.sample(interp_vects)
    v = v.numpy()
    for i, sample in enumerate(v):
        ut.plotImgAndVect(sample,interp_vects[i], threshold=None, limits=(-4,4), show_axes=False, stats=True)

#%% See a random samplling of reconstructions to choose which ones to interpolate between
anchor_vects, labels = getRecons(num_to_get=10)

#%% Interpolate between 2 set reconstructions from the previous method
interpolateDesigns(anchor_vects, labels, 5, 9,divs=10)


#%% Shapetime journey code for fun. Shapetime journey methods :
def showRandIndices(num_to_show=100) :
    for i in np.random.randint(0, len(shape2vec), size=num_to_show) :
        vox = shapemodel.decode(shape2vec[mids[i]][None,...], apply_sigmoid=True)[0,...,0]
        ut.plotImg(vox, limits = cf_limits, title=i)



#%% Start by randomly searching for some object indices to start from
showRandIndices(100)

#%% Remember good starting indices for various categories
start_indices = {
    'Table'  : [7764, 6216, 3076, 2930, 715, 3165],
    'Chair'  : [9479, 13872, 12775, 9203, 9682, 9062, 8801, 8134, 12722, 7906, 10496, 11358, 13475, 9348, 13785, 11697],
    'Lamp'   : [15111, 15007, 14634, 14646, 15314, 14485],
    'Faucet' : [15540, 15684, 15535, 15738, 15412],
    'Clock'  : [16124, 16034, 16153],
    'Bottle' : [16690, 16736, 16689],
    'Vase'   : [17463, 17484, 17324, 17224, 17453],
    'Laptop' : [17780, 17707, 17722],
    'Bed'    : [18217, 18161],
    'Mug'    : [18309, 18368, 18448],
    'Bowl'   : [18501, 17287, 18545, 18479, 18498]}

#%% Start the journey based on the previously selected indices
#journey(journey_length = 20, vects_sample=8, max_dist=8, interp_points=6, plot_step=2, start_index = start_indices['Table'][2])



# import matplotlib.pyplot as plt


# def plot_latent(encoder, decoder):
#     # display a n*n 2D manifold of digits
#     n = cf_latent_dim
#     digit_size = 28
#     scale = 2.0
#     figsize = 15
#     figure = np.zeros((digit_size * n, digit_size * n))
#     # linearly spaced coordinates corresponding to the 2D plot
#     # of digit classes in the latent space
#     grid_x = np.linspace(-scale, scale, n)
#     grid_y = np.linspace(-scale, scale, n)[::-1]

#     for i, yi in enumerate(grid_y):
#         for j, xi in enumerate(grid_x):
#             z_sample = np.array([[xi, yi]])
#             x_decoded = decoder.predict(z_sample)
#             digit = x_decoded[0].reshape(digit_size, digit_size)
#             figure[
#                 i * digit_size : (i + 1) * digit_size,
#                 j * digit_size : (j + 1) * digit_size,
#             ] = digit

#     plt.figure(figsize=(figsize, figsize))
#     start_range = digit_size // 2
#     end_range = n * digit_size + start_range + 1
#     pixel_range = np.arange(start_range, end_range, digit_size)
#     sample_range_x = np.round(grid_x, 1)
#     sample_range_y = np.round(grid_y, 1)
#     plt.xticks(pixel_range, sample_range_x)
#     plt.yticks(pixel_range, sample_range_y)
#     plt.xlabel("z[0]")
#     plt.ylabel("z[1]")
#     plt.imshow(figure, cmap="Greys_r")
#     plt.show()


# plot_latent(model.enc_model, model.gen_model)

# %%
