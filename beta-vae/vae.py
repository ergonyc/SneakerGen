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
cf_img_size = 128
cf_latent_dim = 128
cf_max_loads_per_cat = 2000 
cf_batch_size = 32
cf_learning_rate = 4e-4
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)

# cf_limits=[cf_img_size, cf_img_size, cf_img_size]
cf_limits=[cf_img_size, cf_img_size]
#ut.readMeta()
dfmeta = ut.readMeta()

#%% Make model and print info
def plotIO(model):
    print("\n Net Summary (input then output):")
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


#%% Make model and print info
model = cv.CVAE(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)

#model = cv.CVAE_EF(cf_latent_dim, cf_img_size, cf_learning_rate, training=True)

model.setLR(cf_learning_rate)
model.printMSums()
model.printIO()
plotIO(model.enc_model)
plotIO(model.gen_model)
#tf.keras.utils.plot_model(model.enc_model, show_shapes=True, show_layer_names=True)
#tf.keras.utils.plot_model(model.gen_model, show_shapes=True, show_layer_names=True)

#%% Setup logger info
train_from_scratch = True
if train_from_scratch :
    lg = logger.logger(trainMode=cf.REMOTE, txtMode=False)
    lg = logger.logger(trainMode=True, txtMode=False)
else :
    shp_run_id = '0821-1652'  # convention??
    root_dir = os.path.join(cf.IMG_RUN_DIR, shp_run_id)
    lg = logger.logger(root_dir=root_dir, txtMode=False)



lg.setupCP(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer)
lg.restoreCP()

### Save base train data to file
# np.save(os.path.join(cf.DATA_DIR, 'some_voxs.npy'), all_voxs, allow_pickle=True)
# np.save(os.path.join(cf.DATA_DIR, 'some_mids.npy'), all_mids, allow_pickle=True)

# #%% Load base train data from file
# prefix = 'all' if cf.REMOTE else 'some'
# all_voxs = np.load(os.path.join(save_dir, prefix+'_voxs.npy'), allow_pickle=True)
# all_mids = np.load(os.path.join(save_dir, prefix+'_mids.npy'), allow_pickle=True)

#%% # LOAD & PREPROCESS the from list of files

train_dataset, test_dataset = ut.loadAndPrepData(cf_img_size, lg.img_in_dir, cf_batch_size)

#%% # save data to disk ... (to keep the partitioning) 

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

for test_samples, test_labels in train_dataset.take(1) : pass

#test_samples = tf.cast(test_samples, dtype=tf.float32)


total_train_batchs = 0
for _ in train_dataset :
    total_train_batchs += 1


#%% Setup datasets
sample_index = 31
ut.plotImg(test_samples[sample_index], title='Original', threshold=0.5, limits=cf_limits, save_fig=False)

#%% Show initial models

if (lg.total_epochs > 10) :
    ut.plotVox(model.reconstruct(test_samples[sample_index][None,...], training=False), limits=cf_limits, title='Recon')

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
            test_loss = getTestSetLoss(test_dataset, 2)
            print('   TEST LOSS: {:.1f}    for epoch: {}'.format(test_loss, lg.total_epochs))
            lg.logMetric(test_loss, 'test loss')

        if epoch % save_interval == 0:
            lg.cpSave()

        if (ut.checkStopSignal(dir_path=cf.IMG_RUN_DIR)) : break

        print('Epoch: {}   Train loss: {:.1f}   Epoch Time: {:.2f}'.format(lg.total_epochs, float(elbo), float(time.time() - start_time)))
        lg.logMetric(elbo, 'train loss')
        lg.incrementEpoch()
    return




#%% Training data save
start_time = time.time()
losses = []
batch_index = 1
imgs = []
labels = []

for train_x, label in train_dataset :
    #train_x = tf.cast(train_x, dtype=tf.float32)
    imgs.append(train_x.numpy())
    labs = list(label.numpy())
    labels.extend(labs)
    stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
    stdout.flush()
    batch_index = batch_index + 1

trainimgs = np.stack(imgs)
trainlabs = labels # np.stack(labels)

print('Epoch Time: {:.2f}'.format( float(time.time() - start_time)))

#path = os.path.join(cf.IMG_RUN_DIR, "saved_data","train")
# #save training data
# tf.data.experimental.save(
#     train_dataset, path, compression=None, shard_func=None
# )
ut.dumpPickle(os.path.join(lg.root_dir, "saved_data","train.pkl"), (trainimgs,trainlabs) )
#%% testing data save 

batch_index = 1
imgs = []
labels = []
for test_x, label in test_dataset :
    #train_x = tf.cast(train_x, dtype=tf.float32)
    imgs.append(train_x.numpy())
    labs = list(label.numpy())
    labels.extend(labs)
    
    stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, 16))
    stdout.flush()
    batch_index = batch_index + 1

flatten = lambda l: [item for sublist in l for item in sublist]

testlabs = labels # np.stack(labels)
testimgs = np.stack(imgs)
print('Epoch Time: {:.2f}'.format( float(time.time() - start_time)))
ut.dumpPickle(os.path.join(lg.root_dir, "saved_data","test.pkl"), (testimgs,testlabs) )
#%% Train model
lg.writeConfig(locals(), [cv.CVAE, cv.CVAE.__init__])
lg.updatePlotDir()

#%% 
n_epochs = 400
trainModel(n_epochs, display_interval=2, save_interval=5, test_interval=5)



#%% 

#for test_samples, test_labels in train_dataset.take(1) : pass
for train_samples, train_labels in train_dataset.take(1) : pass

#%% 
sample_index = 1

for sample_index in range(0,31):

    ut.showReconstruct(model, train_samples, title=lg.total_epochs, index=sample_index, show_original=True, save_fig=False, limits=cf_limits)

#%% 
def showAugmentationExamples(model, sample1, sample2, title, save_fig=False, limits=None):

    pred1 = model.reconstruct(sample1, training=False)
    #img1 = sample1[None, ...]

    pred2 = model.reconstruct(sample2, training=False)
    #img2 = sample2[None, ...]


    ut.plotImg(
            sample1, title="Original {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )
    ut.plotImg(
            pred1, title="Reconstruct {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )

    ut.plotImg(
            sample2, title="Original {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )
    ut.plotImg(
            pred2, title="Reconstruct {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )


#%% 
files = list(test_labels.numpy())
img1,img2 = ut.loadAndPrepDataForTesting(128,files,32)


for sample_index in range(0,len(files)):
tx, tl = img1.take(1)
tx = tf.cast(tx, dtype=tf.float32)

    showAugmentationExamples(model, img1, img2,title=files[sample_index], save_fig=False, limits=cf_limits)



#%% 

model.saveMyModel(lg.root_dir, 400)
#%% 

model.loadMyModel(lg.root_dir,99)
# Need to make methods to extract the pictures from test_dataset/train_dataset


#%% 
n_epochs = 50
trainModel(n_epochs, display_interval=2, save_interval=5, test_interval=2)


#%% 


model.saveMyModel(lg.root_dir, 138)
#%% 

model.loadMyModel(lg.root_dir,669)
# Need to make methods to extract the pictures 

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
#%% Methods for exploring the design space and getting a feel for model performance 
def getRecons(num_to_get=10, cat_label_index=-2) :
    model.training = False
    anchors, labels = [],[]
    for anchor, label in train_dataset.unbatch().shuffle(100000).take(num_to_get*50) :
        catid = -1
        try: catid = cf_cat_prefixes.index('0{}'.format(ut.getMidCat(label.numpy().decode())))
        except : print('not found\n ', label.numpy().decode())
        if (catid == cat_label_index or cat_label_index==-2) :
            anchor = tf.cast(anchor, dtype=tf.float32)
            anchors.append(anchor)
            labels.append(label)
        if (len(anchors) >= num_to_get) :
            break

    anchor_vects = [model.encode(anchors[i][None,:,:,:], reparam=True) for i in range(len(anchors))]
    v = [model.sample(anchor_vects[i]).numpy()[0,...,0] for i in range (len(anchors))]

    for i, sample in enumerate(v) :
        print('Index: {}   Mid: {}'.format(i, labels[i].numpy().decode()))
        ut.plotVox(anchors[i].numpy()[...,0], step=1, threshold=0.5, title='Index {} Original'.format(i), limits=cf_limits)
        ut.plotVox(v[i], step=2, threshold=0.5, title='Index {} Reconstruct'.format(i), limits=cf_limits) 

    print([mid.numpy().decode() for mid in labels])
    return anchor_vects, labels

def interpolateDesigns(anchor_vects, labels, index1, index2, divs=10) :

    mids_string = ' {} , {} '.format(labels[index1].numpy().decode(), labels[index2].numpy().decode())
    print(mids_string)
    interp_vects = ut.interp(anchor_vects[index1].numpy(), anchor_vects[index2].numpy(), divs)

    v = model.sample(interp_vects)
    v = v.numpy()[:, :, :, :, 0]
    for i, sample in enumerate(v):
        ut.plotVox(sample, step=1, threshold=0.5, limits=cf_limits, show_axes=False)

#%% See a random samplling of reconstructions to choose which ones to interpolate between


anchor_vects, labels = getRecons(num_to_get=10, cat_label_index=8)

#%% Interpolate between 2 set reconstructions from the previous method
interpolateDesigns(anchor_vects, labels, 3, 5)


#%% Shapetime journey code for fun. Shapetime journey methods :
def showRandIndices(num_to_show=100) :
    for i in np.random.randint(0, len(shape2vec), size=num_to_show) :
        vox = shapemodel.decode(shape2vec[mids[i]][None,...], apply_sigmoid=True)[0,...,0]
        ut.plotImg(vox, limits = cf_limits, title=i)

def journey(journey_length = 20, vects_sample=8, max_dist=8, interp_points=6, plot_step=2, start_index = 715)
    model.training=False
    journey_vecs = []
    visited_indices = [start_index]
    journey_mids = []

    mids = list(shape2vec.keys())
    vecs = np.array([shape2vec[m] for m in mids])
    vec_tree = spatial.KDTree(vecs)
    start_vect = shape2vec[mids[start_index]]
    journey_mids.append(mids[start_index])

    for i in range(journey_length) :
        n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist)
        if len(shape2vec) in close_ids :
            n_dists, close_ids = vec_tree.query(start_vect, k = vects_sample, distance_upper_bound=max_dist*3)
        close_ids = list(close_ids)  #[:1000]

        for index in sorted(close_ids, reverse=True):
            if index in visited_indices:
                close_ids.remove(index)

        next_index = random.choice(close_ids)
        next_vect = vecs[next_index]
        visited_indices.append(next_index)
        interp_vects = ut.interp(next_vect, start_vect, divs = interp_points)
        journey_vecs.extend(interp_vects)
        start_vect = next_vect
        journey_mids.append(mids[next_index])

    journey_voxs = np.zeros(( len(journey_vecs), cf_img_size, cf_img_size, cf_img_size))
    for i, vect in enumerate(journey_vecs) :
        journey_voxs[i,...] = model.decode(vect[None,...], apply_sigmoid=True)[0,...,0]

    for i, vox in enumerate(journey_voxs) :
        ut.plotVox(vox, step=plot_step, limits = cf_limits, title='', show_axes=False)

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
journey(journey_length = 20, vects_sample=8, max_dist=8, interp_points=6, plot_step=2, start_index = start_indices['Table'][2])



import matplotlib.pyplot as plt


def plot_latent(encoder, decoder):
    # display a n*n 2D manifold of digits
    n = 30
    digit_size = 28
    scale = 2.0
    figsize = 15
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap="Greys_r")
    plt.show()


plot_latent(encoder, decoder)