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
import random as rn
import time
import spacy
from tqdm import tqdm
from sys import stdout
import matplotlib.pyplot as plt
import glob
# import seaborn as sns

import cvae as cv
import utils as ut
import logger
import textspacy as ts
import configs as cf

np.set_printoptions(precision=3, suppress=True)

AUTOTUNE = tf.data.experimental.AUTOTUNE
VAL_FRAC = 20.0 / 100.0

# %reload_ext autoreload
# %autoreload 2

#%% Filter out any data that is not present in shape2vec and stack it into np arrays

#######
cf_img_size = cf.IMG_SIZE # 128,192, 224, 256
cf_latent_dim = cf.LATENT_DIM # 128
cf_max_loads_per_cat = 2000
cf_batch_size = 32
cf_learning_rate = 5e-4
# ( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#%% Filter out any data that is not present in shape2vec and stack it into np arrays

cf_max_length = 200
cf_test_ratio = VAL_FRAC
cf_trunc_type = "post"
cf_padding_type = "post"
cf_oov_tok = "<OOV>"
cf_limits = [cf_img_size, cf_img_size]

dfmeta = ut.read_meta()


img_run_id = "0907-0217"   # last full image training runs cf_img_size =  192
cf_img_size = 192  #should this just read from config... or from a config loaded in run_id

img_run_id = "0908-2320"   # last full image training runs cf_img_size =  256
cf_img_size = 256  #should this just read from config... or from a config loaded in run_id

img_run_id = "0910-1539"   # last full image training runs cf_img_size =  256
cf_img_size = 256  #should this just read from config... or from a config loaded in run_id

img_run_id = "0911-1516"   # last full image training runs cf_img_size =  256
cf_img_size = 192  #should this just read from config... or from a config loaded in run_id

img_run_id = "0912-2052"   # last full image training runs cf_img_size =  256
cf_img_size = 192  #should this just read from config... or from a config loaded in run_id

img_run_id = "0923-1848"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 1

img_run_id = "0927-2332"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 3

img_run_id = "0928-1018"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 10

img_run_id = "0928-1504"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 2

img_run_id = "0929-0754"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 3

img_run_id = "0929-1720"   # last full image training runs cf_img_size =  256
cf_img_size = 224  #should this just read from config... or from a config loaded in run_id
cf_kl_weight = 1


snk2vec = ut.load_pickle(os.path.join(cf.IMG_RUN_DIR, img_run_id, "snk2vec.pkl"))

# infile = open(os.path.join(cf.SHAPE_RUN_DIR, img_run_id, "snk2vec.pkl"),'rb')
# snk2vec = pickle.load(infile)
# infile.close()
#%% check to see that our gpu-tensorflow is flowing
tf.config.experimental.list_physical_devices('GPU') 

#%% Setup spacy
def get_embeddings(vocab):
    lexemes = [vocab[orth] for orth in vocab.vectors]  # changed in spacy v2.3...
    max_rank = max(lex.rank for lex in lexemes)
    vectors = np.zeros((max_rank + 1, vocab.vectors_length), dtype="float32")
    for lex in lexemes:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors


# nlp = spacy.load("en_core_web_md", entity=False)
nlp = spacy.load("en_core_web_md")

for orth in nlp.vocab.vectors:
    _ = nlp.vocab[orth]

embeddings = get_embeddings(nlp.vocab)

# nlp = spacy.load('en_vectors_web_lg')
# n_vectors = 105000  # number of vectors to keep
# removed_words = nlp.vocab.prune_vectors(n_vectors)

# assert len(nlp.vocab.vectors) <= n_vectors  [mid]# unique vectors have been pruned
# assert nlp.vocab.vectors.n_keys > n_vectors  # but not the total entries

# elayer = tf.keras.layers.Embedding(
#     embeddings.shape[0],
#     embeddings.shape[1],
#     weights=[embeddings],
#     input_length=cf_max_length,
#     trainable=False,
# )

#%% Encoding methods
def padEnc(text):
    texts = text if type(text) == list else [text]

    # probably can do some sort of walrus here...
    lexs = [[nlp.vocab[t] for t in sent.replace(".", " . ").split(" ") if len(t) > 0] for sent in texts]
    ranks = [[l.rank for l in lex if not l.is_oov] for lex in lexs]

    # ranks = [
    #     [t if t != 18446744073709551615 else 0 for t in rank] for rank in ranks
    # ]  # just change overflow to zero...
    padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded


# #%% Get a list of all mids and descs and latent vectors
save_template = cf.DATA_DIR + "/{}.npy"
alldnp = np.load(save_template.format("alldnp"))
mids = list(alldnp[:, 0])
# TODO:  fix this hack... the alldnp file should be easier to move ROOT directories...
# probably make relative and 
all_mids = [m.replace("Users","home") for m in mids]  #fix for linux vs OSX
all_descs = list(alldnp[:, 1])
all_pdescs = list(padEnc(all_descs))

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


#%% LOAD data
mnp, dnp, vnp, pnp = (
    np.load(save_template.format("mnp")),
    np.load(save_template.format("dnp")),
    np.load(save_template.format("vnp")),
    np.load(save_template.format("pnp")),
)


#%% Make datasets
num_samples = len(pnp)
val_samples = int(cf_test_ratio * num_samples)
train_samples = num_samples - val_samples


#%% define splitShuffleData
# we can get rid of this... we are not using in currently... just setting the seed for repeatability
def splitShuffleData(files, test_split):
    """[summary]

    Args:
        files ([type]): [description]img_run_id = "0821-2318"   # last full image training runs
    Returns:
        [test_files,val_files,is_val]: [description]
    """

    for _ in range(100):  # shuffle 100 times
       np.random.shuffle(files)

    data_size = files.shape[0]
    train_size = int((1.0 - test_split) * data_size)
    
    # last full image training runs
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


#%% Set up the data.. maybe set up logger first?
#img_in_dir = lg.img_in_dir
img_in_dir = "/home/ergonyc/Projects/DATABASE/SnkrScrpr/data/"
files = glob.glob(os.path.join(img_in_dir, "*/img/*"))
#shuffle the dataset (GOAT+SNS)
files = np.asarray(files)

np.random.seed(488)
tf.random.set_seed(488)

train_data, val_data, all_data = splitShuffleData(pnp,cf_test_ratio)

#(pnp, vnp)

# ## Save base train da
## Save base train data to file
np.save(os.path.join(cf.DATA_DIR, 'train_data.npy'), train_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'val_data.npy'), val_data, allow_pickle=True)
np.save(os.path.join(cf.DATA_DIR, 'all_data.npy'), all_data, allow_pickle=True)


#%% Load base train data from file
train_dat = np.load(os.path.join(cf.DATA_DIR, 'train_data.npy'))
val_dat = np.load(os.path.join(cf.DATA_DIR, 'val_data.npy'))
all_dat = np.load(os.path.join(cf.DATA_DIR, 'all_data.npy'),allow_pickle=True)


dataset = tf.data.Dataset.from_tensor_slices((pnp, vnp)).shuffle(1000000)
train_ds = dataset.take(train_samples).batch(cf_batch_size)
val_ds = dataset.skip(train_samples).take(val_samples).batch(cf_batch_size)

for train_x, train_y in train_ds:
    pass

for val_x, val_y in val_ds:
    pass

total_train_batchs = 0
for _ in train_ds:
    total_train_batchs += 1

#%% Make model and print info
def plot_IO(model):
    print("\n Net Summary (input then output):")
    return tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=True)


#%% Make text model
txtmodel = ts.TextSpacy(
    cf_latent_dim, 
    learning_rate=cf_learning_rate, 
    max_length=cf_max_length, 
    training=True, 
    embeddings=embeddings
)
#txtmodel = ts.TextSpacy(128, learning_rate=6e-4, max_length=cf_max_length, training=True)

txtmodel.print_model_summary()
txtmodel.print_model_IO()

# txtmodel.model.compile(optimizer="adam",run_eagerly=True,loss="mean_squared_error",metrics=['mse'])
txtmodel.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

loss_mean = tf.keras.metrics.Mean()

plot_IO(txtmodel.model)

train_from_scratch = True

if train_from_scratch:
    lg = logger.logger(trainMode=True, txtMode=True)

else:  # TODO:  making the model actually load is nescessary
    txt_run_id = "0909-0413"
    total_epochs =  1200
    root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
    lg = logger.logger(trainMode=True,root_dir=root_dir, txtMode=True)
    lg.reset(total_epochs=total_epochs)
    txtmodel.restore_latest_model(lg.model_saves)


lg.setup_checkpoint(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restore_checkpoint()
lg.check_make_dirs()

# #%% write config
# lg.writeConfig(locals(), [cv.CVAE, cv.CVAE.__init__])
# lg.updatePlotDir()


#%%
# train_x = tf.cast(train_x, dtype=tf.float32)


#%% Training data save
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

#%% Method for training the model manually
# TODO change names of internal loss collectors... they are not "ELBO"
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

        if (ut.check_stop_signal(dir_path=cf.TXT_RUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")
    
    out_losses = (loss_train,loss_test)
    return epoch, out_losses #(loss_batch2,loss_batchN)



#%% Train the model
lg.write_config(locals(),[ts])  #this atually makes the directories...


#%%
# n_epochs = 400
# # trainModel(n_epochs, display_interval=2, save_interval=5, test_interval=5)
# total_epochs = 0
# n_epochs = 10
# epoch_n = trainModel(n_epochs, save_interval=10, validate_interval=5)

# total_epochs += epoch_n
# if ( lg.total_epochs == total_epochs ):
#     print(f"sanity epoch={total_epochs}")
# else:
#     print(f"sanity epoch={total_epochs}/l.epoch={lg.total_epochs}")

# txtmodel.save_model(lg.root_dir, lg.total_epochs )

#%%
n_epochs = 2500
total_epochs = 0
epoch_n, curr_losses = train_model(n_epochs, save_interval=20, test_interval=20,current_losses=([],[]))
#epoch_n,elbo_train,elbo_test = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n
if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
else:
    lg.reset(total_epochs=total_epochs)
txtmodel.save_model(lg.root_dir, lg.total_epochs )

ut.dump_pickle(os.path.join(lg.saved_data,"losses.pkl"),curr_losses)

