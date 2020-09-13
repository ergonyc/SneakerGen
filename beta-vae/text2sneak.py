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

dfmeta = ut.readMeta()


img_run_id = "0907-0217"   # last full image training runs cf_img_size =  192
cf_img_size = 192  #should this just read from config... or from a config loaded in run_id

img_run_id = "0908-2320"   # last full image training runs cf_img_size =  256
cf_img_size = 256  #should this just read from config... or from a config loaded in run_id

img_run_id = "0910-1539"   # last full image training runs cf_img_size =  256
cf_img_size = 256  #should this just read from config... or from a config loaded in run_id

img_run_id = "0911-1516"   # last full image training runs cf_img_size =  256
cf_img_size = 192  #should this just read from config... or from a config loaded in run_id

snk2vec = ut.loadPickle(os.path.join(cf.IMG_RUN_DIR, img_run_id, "snk2vec.pkl"))

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

# train_dataset, test_dataset = ut.loadAndPrepData(cf_vox_size, lg.vox_in_dir, cf_batch_size)

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
def plotIO(model):
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

txtmodel.printMSums()
txtmodel.printIO()

# txtmodel.model.compile(optimizer="adam", run_eagerly=True,
#     loss="mean_squared_error",metrics=['mse'])
txtmodel.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

loss_mean = tf.keras.metrics.Mean()

plotIO(txtmodel.model)

train_from_scratch = True

if train_from_scratch:
    lg = logger.logger(trainMode=True, txtMode=True)

else:  # TODO:  making the model actually load is nescessary
    txt_run_id = "0909-0413"
    total_epochs =  1200
    root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
    lg = logger.logger(trainMode=True,root_dir=root_dir, txtMode=True)
    lg.reset(total_epochs=total_epochs)
    #txtmodel.loadMyModel(lg.root_dir,total_epochs-1)
    txtmodel.restoreLatestMyModel(lg.model_saves)


lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restoreCP()
lg.checkMakeDirs()

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

# path = os.path.join(cf.IMG_RUN_DIR, "saved_data","train")
# #save training data
# tf.data.experimental.save(
#     train_dataset, path, compression=None, shard_func=None
# )
# need to force saved_data to get maid... probably in logger class..

#ut.dumpPickle(os.path.join(cf.TXT_RUN_DIR, "saved_data", "train.pkl"), (trainimgs, trainlabs))
ut.dumpPickle(os.path.join(lg.saved_data, "train.pkl"), (trainimgs, trainlabs))


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
#ut.dumpPickle(os.path.join(cf.TXT_RUN_DIR, "saved_data", "test.pkl"), (testimgs, testlabs))
ut.dumpPickle(os.path.join(lg.saved_data, "test.pkl"), (testimgs, testlabs))

#%% Method for training the model manually
# TODO change names of internal loss collectors... they are not "ELBO"
def trainModel(num_epochs, 
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
            lg.logMetric(val_loss_epoch, "val loss")
            loss_test.append(val_loss_epoch)
            print("TEST LOSS: {:.3f}".format(val_loss_epoch))

        if epoch % save_interval == 0:
            print(f"saving... epoch_n = {epoch}")
            lg.cpSave()

        print(
            "Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}".format(
                lg.total_epochs, float(loss_epoch), float(time.time() - start_time)
            )
        )

        lg.logMetric(loss_epoch, "train loss")
        loss_train.append(loss_epoch)
        lg.incrementEpoch()

        if (ut.checkStopSignal(dir_path=cf.TXT_RUN_DIR)) :
            print(f"stoping at epoch = {epoch}")
            break
        else:
            print(f"executed {epoch} epochs")
    
    out_losses = (loss_train,loss_test)
    return epoch, out_losses #(loss_batch2,loss_batchN)



#%% Train the model
lg.writeConfig(locals(),[ts])  #this atually makes the directories...


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

# txtmodel.saveMyModel(lg.root_dir, lg.total_epochs )

#%%
n_epochs = 12000
total_epochs = 0
epoch_n, curr_losses = trainModel(n_epochs, save_interval=20, test_interval=20,current_losses=([],[]))
#epoch_n,elbo_train,elbo_test = trainModel(n_epochs, display_interval=5, save_interval=5, test_interval=5)
total_epochs += epoch_n
if lg.total_epochs == total_epochs:
    print(f"sanity epoch={total_epochs}")
else:
    lg.reset(total_epochs=total_epochs)
txtmodel.saveMyModel(lg.root_dir, lg.total_epochs )

#ut.dumpPickle(os.path.join(cf.TXT_RUN_DIR, "saved_data","losses.pkl"),curr_losses)
ut.dumpPickle(os.path.join(lg.saved_data,"losses.pkl"),curr_losses)

#%%


# #model.loadMyModel(lg.root_dir,total_epochs)
# loss_batches_old = ut.loadPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","losses.pkl"))
# ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","losses_old.pkl"), loss_batches_old )

# n_epochs = 500/home/ergonyc/Projects/Project2.0/SnkrGen/beta-vae/txtruns/0908-0240
# epoch_n, curr_losses = trainModel(n_epochs, save_interval=10, test_interval=10,current_losses=curr_losses)
# total_epochs += epoch_n
# if lg.total_epochs == total_epochs:
#     print(f"sanity epoch={total_epochs}")
# else:
#     lg.reset(total_epochs=total_epochs)
# txtmodel.saveMyModel(lg.root_dir, lg.total_epochs )

# ut.dumpPickle(os.path.join(cf.IMG_RUN_DIR, "saved_data","losses.pkl"),curr_losses )

#%%

### TODO: I THINK WE HAVE THIS FILE AND ITS NOT CHANGING

# #%% Generate a large set of sample descriptions to inform nearby descriptions on app
# mid2desc = {}
# for mid in tqdm(snk2vec.keys()):
#     mid = mid.decode()
#     indices = np.where(mnp == mid)[0]
#     if len(indices) > 0:
#         desc = dnp[rn.sample(list(indices), 1)][0]
#         mid2desc[mid] = desc

# file = open(os.path.join(cf.DATA_DIR, "mid2desc.pkl"), "wb")
# pickle.dump(mid2desc, file)
# file.close()

#%%
# total_epochs =  250
# lg.reset(total_epochs=total_epochs)

# if lg.total_epochs == total_epochs:
#     print(f"sanity epoch={total_epochs}")
# txtmodel.saveMyModel(lg.root_dir, lg.total_epochs )
# #%% addd more iterations...
# n_epochs = 2
# #txtmodel.restoreLatestMyModel(lg.model_saves)
# epoch_n = trainModel(n_epochs, save_interval=10, validate_interval=5)
# total_epochs += epoch_n
# if lg.total_epochs == total_epochs:
#     print(f"sanity epoch={total_epochs}")
# else:
#     print(f"sanity epoch={total_epochs}/l.epoch={lg.total_epochs}")
# txtmodel.saveMyModel(lg.root_dir, lg.total_epochs )

# #%% addd more iterations...
# n_epochs = 51
# #%% Compare predicted and labeled vectors, useful sanity check on trained model
# index = 1
# for tx, tl in train_ds.take(1):restoreLatestMyModel
#     tx = tf.cast(tx, dtype=tf.float32)

# txtmodel.model.training = False
# pred = txtmodel.sample(tx)
# print("\nPredicted vector: \n", pred[index], "\n")
# print("Label vector: \n", tl[index])
# l = txtmodel.compute_loss(tl[index], pred[index]).numpy()
# signs_eq = sum(np.sign(pred[index]) == np.sign(tl[index])) / pred[index].shape[0]
# print(
#     "\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}  Same Sign: {:.1f}%".format(
#         index, l, np.sum(pred[index]), np.sum(tl[index]), 100 * signs_eq
#     )
# )

# #%% Get test set loss
# for tx, tl in train_ds.shuffle(100000).take(1000):
#     pass
# txtmodel.model.training = False
# pred = txtmodel.sample(tx)
# losses = np.mean(txtmodel.compute_loss(pred, tl))
# print(losses)

# #%% Load shape model
# snkmodel = cv.CVAE(cf_latent_dim, cf_img_size)
# snkmodel.printMSums()
# snkmodel.printIO()

# # img_run_id = "0828-2326"
# # img_run_id = "0831-1253"   # last full image training runs cf_img_size =  224

# #snkmodel.loadMyModel(lgImg.root_dir, 282)


# root_dir = os.path.join(cf.IMG_RUN_DIR, img_run_id)
# lgImg = logger.logger(root_dir=root_dir, trainMode=False)
# lgImg.setupCP(encoder=snkmodel.enc_model, generator=snkmodel.gen_model, opt=snkmodel.optimizer)
# lgImg.restoreCP()

# #%% Method for going from text to image
# def getImg(text):
#     ptv = padEnc(text)
#     preds = txtmodel.sample(ptv)
#     img = snkmodel.sample(preds).numpy()[0, ..., 0]
#     return img


# #%% Test text2shape model
# textlist = [
#     "The jordan 1 is a crazy thing.  Nike gave it clean lines and i hi top. 1989 never looked so fresh.",
#     "Yeezy boost is here.  For this drop the upper is dope, and chunky.",
#     "another great puma clyde.  ",
#     "dunk.  nike sb",
#     "chuck taylor. express yourself.  classic canvas and rubber",
#     "air max '95 ",
# ]

# # textlist = dnp[1:10]
# for text in textlist:
#     img = getImg(text)
#     ut.plotImg(img, limits=cf_limits, title=text[0:4restoreLatestMyModel0])

# #%% Test text2shape model
# textlist = dnp[0:3]
# for text in textlist:
#     img = getImg(text)
#     ut.plotImg(img, limits=cf_limits, title=text[0:60])


# #%% Test text2shape model
# textlist = dnp[1504:1509]
# for text in textlist:
#     img = getImg(text)
#     ut.plotImg(img, limits=cf_limits, title=text[0:60])


# #%%  restore model from runs
# train_from_scratch = False

# txt_run_id = "0829-2134"
# root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
# # root_dir = lg.root_dir #
# lg = logger.logger(root_dir=root_dir, txtMode=True,trainMode=False)

# lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)

# lg.restoreCP()

# txtmodel.restoreLatestMyModel(lg.model_saves)
# txtmodel.saveMyModel(lg.root_dir, 999)

# # Load the previously saved weights
# # txtmodel.load_weights(latest)

# # Re-evaluate the model

# #%% addd more iterations...

# trainModel(10, save_interval=10, validate_interval=5)

# #%%


# txtmodel.loadMyModel(lg.root_dir, 483)
# # Need to make methods to extract the pictures from test_dataset/train_dataset

# #%% Train the model


# #%% Compare predicted and labeled vectors, useful sanity check on trained model
# index = 1
# for tx, tl in train_ds.take(1):
#     tx = tf.cast(tx, dtype=tf.float32)

# txtmodel.model.training = False
# pred = txtmodel.sample(tx)
# print("\nPredicted vector: \n", pred[index], "\n")
# print("Label vector: \n", tl[index])
# l = txtmodel.compute_loss(tl[index], pred[index]).numpy()
# signs_eq = sum(np.sign(pred[index]) == np.sign(tl[index])) / pred[index].shape[0]
# print(
#     "\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}  Same Sign: {:.1f}%".format(
#         index, l, np.sum(pred[index]), np.sum(tl[index]), 100 * signs_eq
#     )
# )

# #%% Get test set loss
# for tx, tl in train_ds.shuffle(100000).take(1000):
#     pass
# txtmodel.model.training = False
# pred = txtmodel.sample(tx)
# losses = np.mean(txtmodel.compute_loss(pred, tl))
# print(losses)

# #%% Load shape model
# snkmodel = cv.CVAE(cf_latent_dim, cf_img_size)
# snkmodel.printMSums()
# snkmodel.printIO()

# img_run_id = "0828-2326"
# img_run_id = "0907-0217"   # last full image training runs cf_img_size =  224

# #snkmodel.loadMyModel(lgImg.root_dir, 282)


# root_dir = os.path.join(cf.IMG_RUN_DIR, img_run_id)
# lgImg = logger.logger(root_dir=root_dir, trainMode=False)
# lgImg.setupCP(encoder=snkmodel.enc_model, generator=snkmodel.gen_model, opt=snkmodel.optimizer)
# lgImg.restoreCP()

# #%% Method for going from text to image
# def getImg(text):
#     ptv = padEnc(text)
#     preds = txtmodel.sample(ptv)
#     img = snkmodel.sample(preds).numpy()[0, ..., 0]
#     return img


# #%% Test text2shape model
# textlist = [
#     "The jordan 1 is a crazy thing.  Nike gave it clean lines and i hi top. 1989 never looked so fresh.",
#     "Yeezy boost is here.  For this drop the upper is dope, and chunky.",
#     "another great puma clyde.  ",
#     "dunk.  nike sb",
#     "chuck taylor. express yourself.  classic canvas and rubber",
#     "air max '95 ",
# ]

# # textlist = dnp[1:10]
# for text in textlist:
#     img = getImg(text)
#     ut.plotImg(img, limits=cf_limits, title=text[0:40])

# #%% Test text2shape model
# textlist = dnp[55:58]
# for text in textlist:
#     img = getImg(text)
#     ut.plotImg(img, limits=cf_limits, title=text[0:60])


# # #%% Test text2shape model
# # textlist = dnp[1504:1509]
# # for text in textlist:
# #     img = getImg(text)
# #     ut.plotImg(img, limits=cf_limits, title=text[0:60])


# # #%% Run on single line of text
# # text = "ceiling lamp that is very skinny and very tall. it has one head. it has a base. it has one chain."
# # tensor = tf.constant(text)
# # tbatch = tensor[None, ...]
# # preds = txtmodel.model(tbatch)

# # #%% Generate a large set of sample descriptions to inform nearby descriptions on app
# # mid2desc = {}
# # for mid in tqdm(snk2vec.keys()):
# #     mid = mid.decode()
# #     indices = np.where(mnp == mid)[0]
# #     if len(indices) > 0:
# #         desc = dnp[rn.sample(list(indices), 1)][0]
# #         mid2desc[mid] = desc

# # file = open(os.path.join(cf.DATA_DIR, "mid2desc.pkl"), "wb")
# # pickle.dump(mid2desc, file)
# # file.close()

# # # %%of sample descriptions to show on streamlit app
# # ex_descs = []
# # for keyword in [
# #     "Table",
# #     "Chair",
# #     "Lamp",
# #     "Faucet",
# #     "Clock",
# #     "Bottle",
# #     "Vase",
# #     "Laptop",
# #     "Bed",
# #     "Mug",
# #     "Bowl",
# # ]:
# #     for i in range(50):
# #         desc = dnp[np.random.randint(0, len(dnp))]
# #         while not keyword.lower() in desc:
# #             desc = dnp[np.random.randint(0, len(dnp))]
# #         ex_descs.append(desc)
# #         print(desc)
# # np.save(os.path.join(cf.DATA_DIR, "exdnp.npy"), np.array(ex_descs))


# # %%
