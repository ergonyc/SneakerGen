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

import matplotlib.pyplot as plt

# import seaborn as sns

import cvae as cv
import utils as ut
import logger
import textspacy as ts
import configs as cf

np.set_printoptions(precision=3, suppress=True)

#%% Load text data
#######
cf_img_size = 128
cf_latent_dim = 128
cf_max_loads_per_cat = 2000
cf_batch_size = 128
cf_learning_rate = 4e-4
# ( *-*) ( *-*)>⌐■-■ ( ⌐■-■)

cf_max_length = 200
cf_test_ratio = 0.15
cf_trunc_type = "post"
cf_padding_type = "post"
cf_oov_tok = "<OOV>"
cf_limits = [cf_img_size, cf_img_size]

dfmeta = ut.readMeta()

shp_run_id = "0821-2318"
snk2vec = ut.loadPickle(os.path.join(cf.IMG_RUN_DIR, shp_run_id, "snk2vec.pkl"))

# infile = open(os.path.join(cf.SHAPE_RUN_DIR, shp_run_id, "snk2vec.pkl"),'rb')
# snk2vec = pickle.load(infile)
# infile.close()

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

# assert len(nlp.vocab.vectors) <= n_vectors  # unique vectors have been pruned
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


#%% Get a list of all mids and descs and latent vectors
save_template = cf.DATA_DIR + "/{}.npy"
alldnp = np.load(save_template.format("alldnp"))
all_mids = list(alldnp[:, 0])
all_descs = list(alldnp[:, 1])
all_pdescs = list(padEnc(all_descs))

#%% Filter out any data that is not present in shape2vec and stack it into np arrays
not_found_count = 0
mids, descs, vects, padenc = [], [], [], []
for mid, desc, pdesc in tqdm(zip(all_mids, all_descs, all_pdescs), total=len(all_mids)):
    mids.append(mid)
    descs.append(desc)
    vects.append(snk2vec[mid.astype("bytes")])
    padenc.append(pdesc)

mnp, dnp, vnp, pnp = np.stack(mids), np.stack(descs), np.stack(vects), np.stack(padenc)

np.save(save_template.format("mnp"), mnp)
np.save(save_template.format("dnp"), dnp)
np.save(save_template.format("vnp"), vnp)
np.save(save_template.format("pnp"), pnp)
#%% Save / load the generated arrays to avoid having to regenerate them
# np.save(save_template.format('mnp'), mnp) , np.save(save_template.format('dnp'), dnp) , np.save(save_template.format('pnp'), pnp)
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

dataset = tf.data.Dataset.from_tensor_slices((pnp, vnp)).shuffle(1000000)
train_ds = dataset.take(train_samples).batch(128)
val_ds = dataset.skip(train_samples).take(val_samples).batch(128)

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
    128, learning_rate=6e-4, max_length=cf_max_length, training=True, embeddings=embeddings
)
txtmodel.printMSums()
txtmodel.printIO()

# txtmodel.model.compile(optimizer="adam", run_eagerly=True,
#     loss="mean_squared_error",metrics=['mse'])

txtmodel.model.compile(optimizer="adam", loss=tf.keras.losses.MeanSquaredError())

loss_mean = tf.keras.metrics.Mean()

plotIO(txtmodel.model)


#%%
train_x = tf.cast(train_x, dtype=tf.float32)


#%% Setup logger info
train_from_scratch = True

if train_from_scratch:
    lg = logger.logger(trainMode=cf.REMOTE, txtMode=True)
    lg = logger.logger(trainMode=True, txtMode=True)
else:
    txt_run_id = "0820-1449"
    root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
    lg = logger.logger(root_dir=root_dir, txtMode=True)

lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restoreCP()

#%% Method for training the model manually
def trainModel(num_epochs, save_interval=10, validate_interval=5):
    print("\nStarting training...")
    txtmodel.training = True
    for epoch in range(1, num_epochs):
        start_time = time.time()
        loss_mean.reset_states()
        for train_x, train_y in train_ds:
            train_x = tf.cast(train_x, dtype=tf.float32)
            loss_mean(txtmodel.trainStep(train_x, train_y))
        loss_epoch = loss_mean.result().numpy()

        if epoch % validate_interval == 0:
            loss_mean.reset_states()
            for validation_x, validation_y in val_ds:
                validation_x = tf.cast(validation_x, dtype=tf.float32)
                pred_y = txtmodel.model(validation_x)
                loss_mean(txtmodel.compute_loss(pred_y, validation_y))
            val_loss_epoch = loss_mean.result().numpy()
            lg.logMetric(val_loss_epoch, "val loss")
            print("TEST LOSS: {:.3f}".format(val_loss_epoch))

        if epoch % save_interval == 0:
            lg.cpSave()

        if ut.checkStopSignal(dir_path=cf.TXT_RUN_DIR):
            print("Stop signal recieved...")
            break

        print(
            "Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}".format(
                lg.total_epochs, float(loss_epoch), float(time.time() - start_time)
            )
        )
        lg.logMetric(loss_epoch, "train loss")
        lg.incrementEpoch()


#%% write config
lg.writeConfig(locals(), [ts])

#%% Train the model


trainModel(1000, save_interval=10, validate_interval=5)

#%%
train_from_scratch = False

txt_run_id = "0820-1449"
root_dir = os.path.join(cf.TXT_RUN_DIR, txt_run_id)
lg = logger.logger(root_dir=root_dir, txtMode=True)

lg.setupCP(encoder=None, generator=txtmodel.model, opt=txtmodel.optimizer)
lg.restoreCP()
#%%

txtmodel.saveMyModel(lg.root_dir, 200)
#%% addd more iterations...

trainModel(1000, save_interval=10, validate_interval=5)

#%%


txtmodel.loadMyModel(lg.root_dir, 200)
# Need to make methods to extract the pictures from test_dataset/train_dataset

#%% Train the model

print("\nStarting training...")
txtmodel.training = True
for epoch in range(1, num_epochs):
    start_time = time.time()
    loss_mean.reset_states()
    i = 0
    for train_x, train_y in train_ds:
        train_x = tf.cast(train_x, dtype=tf.float32)
        loc_loss = txtmodel.trainStep(train_x, train_y)
        loss_mean(loc_loss)
        print(f"itter{i}")
        i += 1
    loss_epoch = loss_mean.result().numpy()

    if epoch % validate_interval == 0:
        loss_mean.reset_states()
        for validation_x, validation_y in val_ds:
            validation_x = tf.cast(validation_x, dtype=tf.float32)
            pred_y = txtmodel.model(validation_x)
            loss_mean(txtmodel.compute_loss(pred_y, validation_y))

        val_loss_epoch = loss_mean.result().numpy()
        lg.logMetric(val_loss_epoch, "val loss")
        print("TEST LOSS: {:.3f}".format(val_loss_epoch))

    if epoch % save_interval == 0:
        lg.cpSave()

    if ut.checkStopSignal():
        print("Stop signal recieved...")
        break

    print(
        "Epoch: {:4d}  Loss: {:.3f}  Time: {:.2f}".format(
            lg.total_epochs, float(loss_epoch), float(time.time() - start_time)
        )
    )
    lg.logMetric(loss_epoch, "train loss")
    lg.incrementEpoch()


#%% Compare predicted and labeled vectors, useful sanity check on trained model
index = 1
for tx, tl in train_ds.take(1):
    tx = tf.cast(tx, dtype=tf.float32)

txtmodel.model.training = False
pred = txtmodel.sample(tx)
print("\nPredicted vector: \n", pred[index], "\n")
print("Label vector: \n", tl[index])
l = txtmodel.compute_loss(tl[index], pred[index]).numpy()
signs_eq = sum(np.sign(pred[index]) == np.sign(tl[index])) / pred[index].shape[0]
print(
    "\nStats for this comparison\n{:3d}  Loss: {:.3f}  Sum pred: {:.3f}  Sum lab: {:.3f}  Same Sign: {:.1f}%".format(
        index, l, np.sum(pred[index]), np.sum(tl[index]), 100 * signs_eq
    )
)

#%% Get test set loss
for tx, tl in train_ds.shuffle(100000).take(1000):
    pass
txtmodel.model.training = False
pred = txtmodel.sample(tx)
losses = np.mean(txtmodel.compute_loss(pred, tl))
print(losses)

#%% Load shape model
snkmodel = cv.CVAE(cf_latent_dim, cf_img_size)
snkmodel.printMSums()
snkmodel.printIO()

shp_run_id = "0821-0906"
root_dir = os.path.join(cf.IMG_RUN_DIR, shp_run_id)
lg = logger.logger(root_dir=root_dir, trainMode=False)
lg.setupCP(encoder=snkmodel.enc_model, generator=snkmodel.gen_model, opt=snkmodel.optimizer)
lg.restoreCP()

#%% Method for going from text to image
def getImg(text):
    ptv = padEnc(text)
    preds = txtmodel.sample(ptv)
    img = snkmodel.sample(preds).numpy()[0, ..., 0]
    return img


#%% Test text2shape model
textlist = [
    "The jordan 1 is a crazy thing.  Nike gave it clean lines and i hi top. 1989 never looked so fresh.",
    "Yeezy boost is here.  For this drop the upper is dope, and chunky.",
    "another great puma clyde.  ",
]

textlist = [
    "dunk.  nike sb",
    "chuck taylor. express yourself.  classic canvas and rubber",
    "air max '95 ",
]

# textlist = dnp[1:10]
for text in textlist:
    img = getImg(text)
    ut.plotImg(img, limits=cf_limits, title=text[0:40])

#%% Test text2shape model
textlist = dnp[0:3]
for text in textlist:
    img = getImg(text)
    ut.plotImg(img, limits=cf_limits, title=text[0:60])


#%% Test text2shape model
textlist = dnp[1504:1525]
for text in textlist:
    img = getImg(text)
    ut.plotImg(img, limits=cf_limits, title=text[0:60])


#%% Run on single line of text
text = "ceiling lamp that is very skinny and very tall. it has one head. it has a base. it has one chain."
tensor = tf.constant(text)
tbatch = tensor[None, ...]
preds = txtmodel.model(tbatch)

#%% Generate a balanced set of sample descriptions to show on streamlit app
ex_descs = []
for keyword in [
    "Table",
    "Chair",
    "Lamp",
    "Faucet",
    "Clock",
    "Bottle",
    "Vase",
    "Laptop",
    "Bed",
    "Mug",
    "Bowl",
]:
    for i in range(50):
        desc = dnp[np.random.randint(0, len(dnp))]
        while not keyword.lower() in desc:
            desc = dnp[np.random.randint(0, len(dnp))]
        ex_descs.append(desc)
        print(desc)
np.save(os.path.join(cf.DATA_DIR, "exdnp.npy"), np.array(ex_descs))

#%% Generate a large set of sample descriptions to inform nearby descriptions on app
mid2desc = {}
for mid in tqdm(shape2vec.keys()):
    indices = np.where(mnp == mid)[0]
    if len(indices) > 0:
        desc = dnp[rn.sample(list(indices), 1)][0]
        mid2desc[mid] = desc

file = open(os.path.join(cf.DATA_DIR, "mid2desc.pkl"), "wb")
pickle.dump(mid2desc, file)
file.close()
