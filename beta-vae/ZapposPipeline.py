'''
This file is used to pipeline zappos data into the autoencoder

-- no augmentation because we will start by NOT training, just encoding....

1.LOAD

2. prep = square and resize

(2A. use new dataset to continue trining)
(2B. load SNS/GOAT data and re-train...)


3. dump new data files for the streamlit app

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

#%% Make model and print info
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

