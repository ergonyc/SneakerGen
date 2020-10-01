"""
This file is intended for three purposes:
    1. Training the text model
    2. Loading in and quickly testing the overall text2shape model
    3. Generating sample description datasets for use elsewhere
"""
# NOT WORKING!!!! no idea why using snkrFinder_dev... environ
# SnkrSL environment works... "SL = streamlit"
#%% Imports
import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
import random
import time
import spacy
from tqdm import tqdm
from sys import stdout
import glob
# import seaborn as sns

import cvae as cv
import utils as ut
import logger
import textspacy as ts
import configs as cf

np.set_printoptions(precision=3, suppress=True)

AUTOTUNE = tf.data.experimental.AUTOTUNE

JUPYTER_NOTEBOOK = False

# if JUPYTER_NOTEBOOK:
# %reload_ext autoreload
# %autoreload 2


#%% Filter out any data that is not present in shape2vec and stack it into np arrays

#######
cf_img_size = cf.IMG_SIZE # 128,192, 224, 256
cf_latent_dim = cf.LATENT_DIM # 128
cf_max_loads_per_cat = 2000
cf_batch_size = cf.BATCH_SIZE #32
cf_learning_rate = cf.TXTRUN_LR #1e-4
# ( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#%% Filter out any data that is not present in shape2vec and stack it into np arrays
cf_max_length = 200
cf_trunc_type = "post"
cf_padding_type = "post"
cf_oov_tok = "<OOV>"

#cf_kl_weight = cf.KL_WEIGHT
cf_num_epochs = cf.N_TXTRUN_EPOCH
#dfmeta = ut.read_meta()
cf_val_frac = cf.VALIDATION_FRAC
#%%  are we GPU-ed?
tf.config.experimental.list_physical_devices('GPU') 


img_run_id = "0929-2259"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 3

img_run_id = "0930-0846"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 10

img_run_id = "0930-0846"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 1


snk2vec = ut.load_pickle(os.path.join(cf.IMGRUN_DIR, img_run_id, "snk2vec.pkl"))

# infile = open(os.path.join(cf.SHAPE_RUN_DIR, img_run_id, "snk2vec.pkl"),'rb')
# snk2vec = pickle.load(infile)
# infile.close()
#%% check to see that our gpu-tensorflow is flowing
tf.config.experimental.list_physical_devices('GPU') 

#%% Define Training methods
loss_mean = tf.keras.metrics.Mean()

def train_model(num_epochs, 
                save_interval=10, 
                test_interval=5,
                current_losses=([],[])
                ):
    print("\nStarting training...")
    txtmodel.training = True

    loss_test,loss_train = current_losses
    for epoch in range(1, num_epochs):
        start_time = time.time()
        loss_mean.reset_states()
        for train_x, train_y in train_ds:
            train_x = tf.cast(train_x, dtype=tf.float32)
            loss_mean(txtmodel.trainStep(train_x, train_y))
        loss_epoch = loss_mean.result().numpy()

        if epoch % test_interval == 0:
            loss_mean.reset_states()
            for validation_x, validation_y in val_ds:  #val_ds is batched...
                validation_x = tf.cast(validation_x, dtype=tf.float32)
                pred_y = txtmodel.model(validation_x,training=False)
                loss_mean(txtmodel.compute_loss(pred_y, validation_y))

            val_loss_epoch = loss_mean.result().numpy()
            lg.log_metric(val_loss_epoch, "val loss")
            loss_test.append(val_loss_epoch)
            print("TEST LOSS: {:.3f}".format(val_loss_epoch))

        if epoch % save_interval == 0:
            print(f"saving... epoch_n = {epoch}")
            lg.save_checkpoint()

        print(
            "Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}".format(
                lg.total_epochs, float(loss_epoch), float(time.time() - start_time)
            )
        )

        lg.log_metric(loss_epoch, "train loss")
        loss_train.append(loss_epoch)
        lg.increment_epoch()

        if (ut.check_stop_signal(dir_path=cf.TXTRUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")
    
    out_losses = (loss_train,loss_test)
    return epoch, out_losses #(loss_batch2,loss_batchN)

#

#%% #################################################
##
##  LOAD/PREP data
##         - l if we've already been through this for the current database we'll load... otherwise process.
#####################################################

#%% LOAD raw description data data

save_template = cf.DATA_DIR + "/{}.npy"

id_label = np.load(os.path.join(cf.DATA_DIR,'mnp.npy'))  #IDs (filenames)
descriptions = np.load(os.path.join(cf.DATA_DIR,'dnp.npy')) #description
description_vectors = np.load(os.path.join(cf.DATA_DIR,'vnp.npy')) #vectors encoded
padded_encoded_vector = np.load(os.path.join(cf.DATA_DIR,'pnp.npy')) #padded encoded

data_from_scratch = not ut.check_for_datafiles(cf.DATA_DIR,['train_txt_data.npy','val_txt_data.npy','all_txt_data.npy'])

#data_from_scratch = True
random.seed(488)
tf.random.set_seed(488)

if data_from_scratch:
    #create
    files = glob.glob(os.path.join(cf.IMAGE_FILEPATH, "*/img/*"))
    files = np.asarray(files)
    train_data, val_data, all_data = ut.split_shuffle_data(padded_encoded_vector,cf_val_frac)
    # Save base train data to file  
    np.save(os.path.join(cf.DATA_DIR, 'train_txt_data.npy'), train_data, allow_pickle=True)
    np.save(os.path.join(cf.DATA_DIR, 'val_txt_data.npy'), val_data, allow_pickle=True)
    np.save(os.path.join(cf.DATA_DIR, 'all_txt_data.npy'), all_data, allow_pickle=True)
else:
    #load
    print(f"loading train/validate data from {cf.DATA_DIR}")
    train_data = np.load(os.path.join(cf.DATA_DIR, 'train_txt_data.npy'), allow_pickle=True)
    val_data = np.load(os.path.join(cf.DATA_DIR, 'val_txt_data.npy'), allow_pickle=True)
    all_data = np.load(os.path.join(cf.DATA_DIR, 'all_txt_data.npy'), allow_pickle=True)



# #%% Encoding methods
# def get_embeddings(vocab):
#     lexemes = [vocab[orth] for orth in vocab.vectors]  # changed in spacy v2.3...
#     max_rank = max(lex.rank for lex in lexemes)
#     vectors = np.zeros((max_rank + 1, vocab.vectors_length), dtype="float32")
#     for lex in lexemes:
#         if lex.has_vector:
#             vectors[lex.rank] = lex.vector
#     return vectors
# embeddings = get_embeddings(nlp.vocab)

# def padEnc(text):
#     texts = text if type(text) == list else [text]

#     # probably can do some sort of walrus here...
#     lexs = [[nlp.vocab[t] for t in sent.replace(".", " . ").split(" ") if len(t) > 0] for sent in texts]
#     ranks = [[l.rank for l in lex if not l.is_oov] for lex in lexs]

#     # ranks = [
#     #     [t if t != 18446744073709551615 else 0 for t in rank] for rank in ranks
#     # ]  # just change overflow to zero...
#     padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
#     return padded


# # #%% Get a list of all mids and descs and latent vectors
# save_template = cf.DATA_DIR + "/{}.npy"
# alldnp = np.load(save_template.format("alldnp"))
# mids = list(alldnp[:, 0])
# # TODO:  fix this hack... the alldnp file should be easier to move ROOT directories...
# # probably make relative and 
# all_mids = [m.replace("Users","home") for m in mids]  #fix for linux vs OSX
# all_descs = list(alldnp[:, 1])
# all_pdescs = list(padEnc(all_descs))

# #%% Filter out any data that is not present in shape2vec and stack it into np arrays
# not_found_count = 0
# mids, descs, vects, padenc = [], [], [], []
# for mid, desc, pdesc in tqdm(zip(all_mids, all_descs, all_pdescs), total=len(all_mids)):
#     mids.append(mid)
#     descs.append(desc)
#     vects.append(snk2vec[mid.encode()])
#     padenc.append(pdesc)

# mnp, dnp, vnp, pnp = np.stack(mids), np.stack(descs), np.stack(vects), np.stack(padenc)

# np.save(save_template.format("mnp"), mnp)
# np.save(save_template.format("dnp"), dnp)
# np.save(save_template.format("vnp"), vnp)
# np.save(save_template.format("pnp"), pnp)
# id_label = np.load(os.path.join(cf.DATA_DIR,'mnp.npy'))  #IDs (filenames)
# descriptions = np.load(os.path.join(cf.DATA_DIR,'dnp.npy')) #description
# description_vectors = np.load(os.path.join(cf.DATA_DIR,'vnp.npy')) #vectors encoded
# padded_encoded_vector = np.load(os.path.join(cf.DATA_DIR,'pnp.npy')) #padded encoded





#%% Make datasets
num_samples = len(padded_encoded_vector)
val_samples = int(cf_val_frac * num_samples)
train_samples = num_samples - val_samples
data_in = (padded_encoded_vector, description_vectors)

dataset = tf.data.Dataset.from_tensor_slices(data_in).shuffle(1000000)
train_ds = dataset.take(train_samples).batch(cf_batch_size)
val_ds = dataset.skip(train_samples).take(val_samples).batch(cf_batch_size)

for train_x, train_y in train_ds:
    pass

for val_x, val_y in val_ds:
    pass

total_train_batchs = 0
for _ in train_ds:
    total_train_batchs += 1

#%% #################################################
##
##  Set up the model 
##         - load current state or
##         - train from scratch
#####################################################

nlp = spacy.load("en_core_web_md")

for orth in nlp.vocab.vectors:
    _ = nlp.vocab[orth]

lexemes = [nlp.vocab[orth] for orth in nlp.vocab.vectors]  # changed in spacy v2.3...
max_rank = max(lex.rank for lex in lexemes)
embeddings = np.zeros((max_rank + 1, nlp.vocab.vectors_length), dtype="float32")
for lex in lexemes:
    if lex.has_vector:
        embeddings[lex.rank] = lex.vector


# nlp = spacy.load('en_vectors_web_lg')
# n_vectors = 105000  # number of vectors to keep
# removed_words = nlp.vocab.prune_vectors(n_vectors)

# #%% Make text model
txtmodel = ts.TextSpacy(
    cf_latent_dim, 
    learning_rate=cf_learning_rate, 
    max_length=cf_max_length, 
    training=True, 
    embeddings=embeddings
)
txtmodel.print_model_summary()
txtmodel.print_model_IO()

if JUPYTER_NOTEBOOK:
    tf.keras.utils.plot_model(txtmodel.model, show_shapes=True, show_layer_names=True)

# txtmodel.model.compile(optimizer="adam",run_eagerly=True,loss="mean_squared_error",metrics=['mse'])
txtmodel.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

# #%% Setup logger info
train_from_scratch = ( cf.CURR_TXTRUN_ID is None )
if train_from_scratch:
    lg = logger.logger(trainMode=True, txtMode=True)
    lg.setup_checkpoint(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer) # sets up the writer
    #lg.restore_checkpoint() 
    lg.check_make_dirs() # makes all the direcotries
    # copy to the current run train data to file
    np.save(os.path.join(lg.saved_data, 'train_txt_data.npy'), train_data, allow_pickle=True)
    np.save(os.path.join(lg.saved_data, 'val_txt_data.npy'), val_data, allow_pickle=True)
    np.save(os.path.join(lg.saved_data, 'all_txt_data.npy'), all_data, allow_pickle=True)
    total_epochs = 0
    curr_losses = ([],[])
else:
    root_dir = os.path.join(cf.TXTRUN_DIR, cf.CURR_TXTRUN_ID)
    lg = logger.logger(root_dir=root_dir, trainMode=True, txtMode=True)
    lg.setup_checkpoint(encoder=model.enc_model, generator=model.gen_model, opt=model.optimizer) # sets up the writer
    lg.restore_checkpoint() # actuall reads in the  weights...
    allfiles = os.listdir(lg.saved_data)
    print(f"allfiles: {allfiles}")
    total_epochs = [int(f.rstrip(".pkl").lstrip("losses_")) for f in allfiles if f.startswith("losses_")]
    total_epochs.sort(reverse=True)
    print(f"total_epochs = {total_epochs[0]}")
    total_epochs = total_epochs[0]
    curr_losses = ut.load_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"))



#%% Training data save
# do we want to save the image data for the training set... i.e. the augmented bytes?
dump_txt_data = False
if dump_txt_data:

    start_time = time.time()
    batch_index = 1
    imgs = []
    labels = []

    for train_x, label in train_ds:
        train_x = tf.cast(train_x, dtype=tf.float32)
        imgs.append(train_x.numpy())
        labels.append(label.numpy())
        # labs = list(label.numpy())
        # labels.extend(labs)
        stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, total_train_batchs))
        stdout.flush()
        batch_index = batch_index + 1

    trainimgs = imgs  # np.stack(imgs)
    trainlabs = labels  # np.stack(labels)
    print("dump Time: {:.2f}".format(float(time.time() - start_time)))
    ut.dump_pickle(os.path.join(lg.saved_data, "train.pkl"), (trainimgs, trainlabs))

    #%% testing data save
    start_time = time.time()

    batch_index = 1
    imgs = []
    labels = []
    for test_x, label in val_ds:
        # train_x = tf.cast(train_x, dtype=tf.float32)
        imgs.append(train_x.numpy())
        labs = list(label.numpy())
        labels.append(label.numpy())

        stdout.write("\r[{:3d}/{:3d}]  ".format(batch_index, 16))
        stdout.flush()
        batch_index = batch_index + 1

    flatten = lambda l: [item for sublist in l for item in sublist]

    testlabs = labels  # np.stack(labels)
    testimgs = imgs # np.stack(imgs)
    print("dump Time: {:.2f}".format(float(time.time() - start_time)))
    ut.dump_pickle(os.path.join(lg.saved_data, "test.pkl"), (testimgs, testlabs))


#%% 
# #################################################
##
##  log the run and TRAIN!!
##    - train from scratch OR 
##    - start where we left off
##
######################################################%% Train the model
lg.write_config(locals(),[ts])  #this atually makes the directories...


#%%
n_epochs = cf_num_epochs

epoch_n, curr_losses = train_model(n_epochs, save_interval=20, test_interval=20,current_losses=curr_losses )
#epoch_n,elbo_train,elbo_test = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n
if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
else:
    lg.reset(total_epochs=total_epochs)
txtmodel.save_model(lg.root_dir, lg.total_epochs )

ut.dump_pickle(os.path.join(lg.saved_data, f"losses_{total_epochs}.pkl"),curr_losses)

