"""
This file is used to generate the UMAP maps from the shape2vec file generated in vae.py

It visualizes the UMAP maps in a variety of ways including 2D and 3D maps with plotly and seaborn.
It also outputs the vectors in a pandas dataframe file that can be read in by the streamlit app for interactive visualization.
"""
#%% Imports

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from plotly.offline import plot
import plotly.express as px
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import seaborn as sns
import tqdm
import utils as ut
import configs as cf
from PIL import Image


OUTPUT_2D_UMAP = "df_sl_2dUMAP.csv"
OUTPUT_3D_UMAP = "df_sl_3dUMAP.csv"


#%% Read text2vec pickle
shape_run_id = "0824-2245"
run_root_dir = os.path.join(cf.IMG_RUN_DIR, shape_run_id)

snk2vec = ut.loadPickle(os.path.join(run_root_dir, "snk2vec.pkl"))
snk2loss = ut.loadPickle(os.path.join(run_root_dir, "snk2loss.pkl"))




#%% Encoding methods
def get_embeddings(vocab):
    lexemes = [vocab[orth] for orth in vocab.vectors]  # changed in spacy v2.3...
    max_rank = max(lex.rank for lex in lexemes)
    vectors = np.zeros((max_rank + 1, vocab.vectors_length), dtype="float32")
    for lex in lexemes:
        if lex.has_vector:
            vectors[lex.rank] = lex.vector
    return vectors
embeddings = get_embeddings(nlp.vocab)

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

#%% Filter out any data that is not present in shape2vec and stack it into np arrays
not_found_count = 0
mids, descs, vects, padenc = [], [], [], []
for mid, desc, pdesc in tqdm(zip(all_mids, all_descs, all_pdescs), total=len(all_mids)):
    mids.append(mid)
    descs.append(desc)
    vects.append(snk2vec[mid.encode()])
    padenc.append(pdesc)

mnp, dnp, vnp, pnp = np.stack(mids), np.stack(descs), np.stack(vects), np.stack(padenc)

np.save(save_template.format("mnp"), mnp)
np.save(save_template.format("dnp"), dnp)
np.save(save_template.format("vnp"), vnp)
np.save(save_template.format("pnp"), pnp)
id_label = np.load(os.path.join(cf.DATA_DIR,'mnp.npy'))  #IDs (filenames)
descriptions = np.load(os.path.join(cf.DATA_DIR,'dnp.npy')) #description
description_vectors = np.load(os.path.join(cf.DATA_DIR,'vnp.npy')) #vectors encoded
padded_encoded_vector = np.load(os.path.join(cf.DATA_DIR,'pnp.npy')) #padded encoded




bright = [
    "#023EFF",
    "#FF7C00",
    "#1AC938",
    "#E8000B",
    "#8B2BE2",
    "#9F4800",
    "#F14CC1",
    "#A3A3A3",
    "#000099",
    "#00D7FF",
    "#222A2A",
]

#%% Run TSNE on the latent vectors
latent_dim = snk2vec.get(b"/Users/ergonyc/Projects/DATABASE/SnkrScrpr/data/goat/img/75d9be084d.jpg").shape[0]
latent_vects = np.zeros((len(snk2vec), latent_dim))
files = []
for i, key in enumerate(snk2vec.keys()):
    latent_vects[i, :] = snk2vec[key]
    files.append(key)

perp, lr = 40, 200
tsne = TSNE(n_components=2, n_iter=1100, verbose=3, perplexity=perp, learning_rate=lr)
lvects = tsne.fit_transform(latent_vects)
plt.scatter(lvects[:, 0], lvects[:, 1], s=0.99, marker=".")


#%% Run TSNE on the latent vectors


import plotly.express as px

# from sklearn.datasets import load_digits
from umap import UMAP


umap_2d = UMAP(random_state=0)
umap_2d.fit(latent_vects)

projections = umap_2d.transform(latent_vects)

fig = px.scatter(projections, x=0, y=1)
fig.show()

#%%
### REDUCE FEATURE DIMENSIONS ###
pca = PCA(n_components=50, random_state=33)
pca_score = pca.fit_transform(latent_vects)

tsne = TSNE(n_components=2, random_state=33, n_iter=300, perplexity=5)
T = tsne.fit_transform(pca_score)
fig, ax = plt.subplots()

ax.scatter(T.T[0], T.T[1])
plt.grid(False)

shown_images = np.array([[1.0, 1.0]])
choose_200 = np.random.randint(1, T.shape[0], 200)
for i in choose_200:

    img = Image.open(files[i])
    img = img.resize((16, 16))

    shown_images = np.r_[shown_images, [T[i]]]
    imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(img, cmap=plt.cm.gray_r), T[i])
    ax.add_artist(imagebox)

plt.show()

#%% Put TSNE data into frames and format
dfmeta = ut.readMeta()
df_subset = pd.DataFrame(list(shape2vec.keys()), columns=["mid"])
df_subset["tsne1"] = lvects[:, 0]
df_subset["tsne2"] = lvects[:, 1]
if (lvects.shape[1]) == 3:
    df_subset["tsne3"] = lvects[:, 2]

df_subset = pd.merge(df_subset, dfmeta, how="left", on=["mid", "mid"])
dfloss = pd.DataFrame.from_dict(shape2loss, orient="index", columns=["loss"])
dfloss["logloss"] = np.log(dfloss.loss)
dfloss.index.name = "mid"
df_subset = pd.merge(df_subset, dfloss, how="left", on=["mid", "mid"])

# Just to rescale them for easy viewing
df_cols_to_increase = ["dx", "dy", "dz", "dsq"]
for col in df_cols_to_increase:
    df_subset[col] = (df_subset[col] - df_subset[col].min() + 0.1) * 0.05

#%% Plot 2D tsne with seaborn
plt.rcParams["figure.figsize"] = (12, 8)
sns.scatterplot(data=df_subset, x="tsne1", y="tsne2", hue="cattext", s=10, linewidth=0, palette="bright")
plt.axis("off")

#%% Plotly 2D tsne
fig = px.scatter(
    data_frame=df_subset.dropna(),
    hover_data=["subcats", "mid"],
    size="dz",
    x="tsne1",
    y="tsne2",
    color="cattext",
)
fig.update_traces(marker=dict(size=6, opacity=1.0, line=dict(width=0.0)))  # size=2.3
plot(
    fig,
    filename="tsne_plot.html",
    auto_open=False,
    config={"scrollZoom": True, "modeBarButtonsToRemove": ["lasso2d", "zoom2d"]},
)

#%% Plotly 3D tsne
fig = px.scatter_3d(
    data_frame=df_subset.dropna(),
    hover_data=["subcats", "mid"],
    size="dy",
    x="tsne1",
    y="tsne2",
    z="tsne3",
    color_discrete_sequence=bright,
    color="cattext",
)
fig.update_traces(marker=dict(size=2.3, opacity=1.0, line=dict(width=0)))  # size=2.3
plot(fig, filename="tsne_plot.html", auto_open=False)

#%% Save tsne vectors
df_sl_cols_keep = [
    "mid",
    "tsne1",
    "tsne2",
    "cat",
    "dx",
    "dy",
    "dz",
    "cattext",
    "dsq",
    "cx",
    "cy",
    "cz",
    "csq",
    "subcats",
]
df_subset[df_sl_cols_keep].to_csv(os.path.join(cf.DATA_DIR, "df_sl.csv"))

