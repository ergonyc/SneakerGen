"""
This is the streamlit app for my Insight AI 2020A.SV project.
All file references are relative to the this file in the github project so that it works with streamlit for teams (SL4T).

The app can be launched locally by using a terminal to navigate to the directory of this .py file and running this command:
    streamlit run streamlit_app.py
"""
#%%
import sys
import streamlit as st

header = st.title("")
header.header("Importing libraries...")

import numpy as np
import pandas as pd

server_up = True
try:
    import re
    import time
    # from tensorflow.keras.preprocessing.sequence import pad_sequences
    # import spacy
    # import textspacy as ts   #from scipy import spatial
    # import skimage.measure as sm
    # import matplotlib.pyplot as pltmodel
    import matplotlib.image as mpimg
    import matplotlib.pyplot as plt
    # from mpl_toolkits.mplot3d import Axes3D
    # import plotly
    # import plotly.express as px
    # import plotly.figure_factory as FF
    import bokeh.plotting as bplt #import figure, show, output_notebook
    #from bokeh.models import HoverTool, ColumnDataSource, CategoricalColorMapper
    import bokeh
    # from bokeh.palettes import Spectral10
    
    import umap
    
    import os
    from sys import stdout

    import pandas as pd
    import random
    import pickle
    # from tqdm import tqdm
    # import glob
    # import seaborn as sns
    # import umap
    import kcvae as kcv 

    import utils as ut
    import logger
    import configs as cf
    import tensorflow as tf
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    #from scipy import spatial  #for now just brute force to find neighbors
    import scipy 
    #from scipy.spatial import distance

    from io import BytesIO
    from PIL import Image
    import base64
    
except:
    header.header("Server is currently overloaded, please try again later!")
    server_up = False



#%% Setup generic values...
#######
cf_img_size = cf.IMG_SIZE
cf_latent_dim = cf.LATENT_DIM
cf_batch_size = cf.BATCH_SIZE #32
cf_learning_rate = cf.IMGRUN_LR #4e-4
cf_limits = [cf_img_size, cf_img_size]
cf_pixel_dim = (cf_img_size,cf_img_size,3)
#( *-*) ( *-*)>⌐■-■ ( ⌐■-■)
#
cf_kl_weight = cf.KL_WEIGHT
cf_beta = cf_kl_weight
cf_num_epochs = cf.N_IMGRUN_EPOCH
#dfmeta = ut.read_meta()
cf_val_frac = cf.VALIDATION_FRAC

### currently these are dummies until i can re-write the interpolation functions
# TODO remove "categories" and fix the interpolatin tools
cf_cat_prefixes = ut.cf_cat_prefixes = ["goat", "sns"]
cf_num_classes = len(cf_cat_prefixes)

#######  Hold our variables...
models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]
klws = [1,2,3,5,10]
latent_dims = [32,40,64]
pix_dims = [128,160,192]
epochs = 400


# TODO: pack this into a simple class or add model/modelname to dictionary
# set default model/params names


model = models[0]
model_name = model_names[0]

p_vals = [cf_latent_dim, cf_pixel_dim, cf_kl_weight, cf_batch_size]
p_names = ['z_dim','x_dim','kl_weight','batch_size']
params = dict(zip(p_names,p_vals))


#%%  are we GPU-ed?
tf.config.experimental.list_physical_devices('GPU') 

########################################3
#  BOKEH
#
##########################################3
def init_bokeh_plot(umap_df):

    bplt.output_notebook()

    datasource = bokeh.models.ColumnDataSource(umap_df)
    color_mapping = bokeh.models.CategoricalColorMapper(factors=["sns","goat"],
                                        palette=bokeh.palettes.Spectral10)

    plot_figure = bplt.figure(
        title='UMAP projection VAE latent',
        plot_width=1000,
        plot_height=1000,
        tools=('pan, wheel_zoom, reset')
    )

    plot_figure.add_tools(bokeh.models.HoverTool(tooltips="""
    <div>
        <div>
            <img src='@image' style='float: left; margin: 5px 5px 5px 5px'/>
        </div>
        <div>
            <span style='font-size: 14px'>@fname</span>
            <span style='font-size: 14px'>@loss</span>
        </div>
    </div>
    """))

    plot_figure.circle(
        'x',
        'y',
        source=datasource,
        color=dict(field='db', transform=color_mapping),
        line_alpha=0.6,
        fill_alpha=0.6,
        size=4
    )

    return plot_figure


##  HELPERS
def embeddable_image(label):
    return image_formatter(label)


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((64, 64), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f"data:image/png;base64,{image_base64(im)}"




#%% Setup sub methods

def set_widemode_hack():
    max_width_str = f"max-width: 2000px;"
    st.markdown(
        f""" <style> .reportview-container .main .block-container{{ {max_width_str} }} </style> """,
        unsafe_allow_html=True,
    )

##

def embeddable_image(label):
    return image_formatter(label)

def get_image(path):
    i = Image.open(path)
    i.thumbnail((256, 256), Image.LANCZOS)
    return i


def get_thumbnail(path):
    i = Image.open(path)
    i.thumbnail((64, 64), Image.LANCZOS)
    return i

def image_base64(im):
    if isinstance(im, str):
        im = get_thumbnail(im)
    with BytesIO() as buffer:
        im.save(buffer, 'png')
        return base64.b64encode(buffer.getvalue()).decode()

def image_formatter(im):
    return f"data:image/png;base64,{image_base64(im)}"




# do we need it loaded... it might be fast enough??
#@st.cache
def load_UMAP_data():
    # params = ['z_dim','x_dim','kl_weight','batch_size']
    # params and "model_name" should be globally available
    # TODO: kill "params" and just load the kl, latents, xdim globals
    data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
    #vae,hist = load_and_prime_model(model,model_name,params,data_dir,epochs)

    #snk2umap = ut.load_snk2umap(data_dir,params['kl_weight'])
    load_dir = os.path.join(data_dir,f"kl_weight{int(params['kl_weight']):03d}")
    snk2umap = ut.load_pickle(os.path.join(load_dir,"snk2umap.pkl"))
    

    return snk2umap

# do we need it loaded... it might be fast enough??
#@st.cache
def load_latent_data():
    # params = ['z_dim','x_dim','kl_weight','batch_size']
    # params and "model_name" should be globally available
    # TODO: kill "params" and just load the kl, latents, xdim globals
    data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"

    snk2umap = load_UMAP_data()
    #vae,hist = load_and_prime_model(model,model_name,params,data_dir,epochs)
    snk2loss, snk2vec = ut.load_snk2pickles(data_dir,params['kl_weight'])
    # desc_df = pd.read_pickle(f"{cf.DESCRIPTIONS}.pkl")
    # meta_df = pd.read_pickle(f"{cf.META_DATA}.pkl")

    mids = list(snk2vec.keys())
    vecs = np.array([snk2vec[m] for m in mids])
    vec_tree = scipy.spatial.KDTree(vecs)


    latents = np.array(list(snk2vec.values()))
    losses = np.array(list(snk2loss.values()))
    labels = np.array(mids)

    labels2 = np.array(list(snk2umap.keys()))
    embedding = np.array(list(snk2umap.values()))

    assert(np.all(labels == labels2))    
    umap_df = pd.DataFrame(embedding, columns=('x', 'y'))

    umap_df['digit'] = [str(x.decode()) for x in labels]
    umap_df['image'] = umap_df.digit.map(lambda f: embeddable_image(f))
    umap_df['fname'] = umap_df.digit.map(lambda x: f"{x.split('/')[-3]} {x.split('/')[-1]}")
    umap_df['db'] = umap_df.digit.map(lambda x: f"{x.split('/')[-3]}")
    umap_df['loss'] = [f"{x:.1f}" for x in losses]

    return umap_df,snk2vec,latents, labels, vecs,vec_tree,mids


#%%



# @st.cache
def loadExampleDescriptions():
    #example_descriptions = np.load(os.path.join(os.getcwd(), "data/exdnp.npy"))  #line 544 text2snk.py
    desc_df = pd.read_pickle(f"{cf.DESCRIPTIONS}.pkl")
    example_descriptions = desc_df.description.to_numpy()
    return list(example_descriptions)


# @st.cache(allow_output_mutation=True)
def make_text_model():
    # model_in_dir = os.path.join(os.getcwd(), "models/textencoder")
    # textmodel = ts.TextSpacy(cf_latent_dim, max_length=cf_max_length, training=False)
    # textmodel.loadMyModel(model_in_dir, 10569) #10k epochs!!!
    # return textmodel
    return None

#@st.cache(allow_output_mutation=True)
def get_spacy():
    nlp = spacy.load("en_core_web_md", parser=False, tagger=False, entity=False)
    nlp = []
    return nlp #nlp.vocab


def interp(vec1, vec2, divs=5, include_ends=True):
    """[This seems to work for any vectors on our manifold...]
    """
    out = []
    amounts = np.array(range(divs + 1)) / divs if include_ends else np.array(range(1, divs)) / divs
    for amt in amounts:
        interpolated_vect = vec1 * amt + vec2 * (1 - amt)
        out.append(interpolated_vect)
    return np.stack(out)




# def get_im_from_text(text, snkmodel, textmodel, nlp):
#     def pad_enc(text, vocab):
#         texts = text if type(text) == list else [text]
#         lexs = [[vocab[t].rank for t in sent.replace(".", " . ").split(" ") if len(t) > 0] for sent in texts]
#         ranks = [[l.rank for l in lex if not l.is_oov] for lex in lexs]
#         padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
#         return padded

#     ptv = pad_enc(text, nlp)
#     preds = textmodel.sample(ptv)
#     img = snk.sample(preds).numpy()[0, ..., 0]
#     return img, preds

# def load_text_model():
#     return -1




#%%


def show_im(img,plot_it=True, title=""):

    fig = plt.imshow(img)
    # fig = FF. 
    # .create_trisurf(
    #     x=verts[:, 0], y=verts[:, 1], z=verts[:, 2], simplices=faces, title=title, aspectratio=aspect
    # )
    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(nticks=1, range=[0, cf_vox_size + 1], backgroundcolor="white", gridcolor="white"),
    #         yaxis=dict(nticks=1, range=[0, cf_vox_size + 1], backgroundcolor="white", gridcolor="white"),
    #         zaxis=dict(nticks=1, range=[0, cf_vox_size + 1], backgroundcolor="white", gridcolor="white"),
    #     ),
    #     width=900,
    #     height=700,
    #     margin=dict(r=20, l=10, b=10, t=10),
    # )
    return fig


#@st.cache(allow_output_mutation=True)
def load_snkr_model() :
    # model_in_dir = os.path.join(os.getcwd(), 'models/autoencoder')
    # shapemodel = cv.CVAE(128, 64, training=False)
    # shapemodel.loadMyModel(model_in_dir, 195)
    # params['kl_weight'] = kl
    # params['z_dim'] = l
    data_dir = f"data/{model_name}-X{params['x_dim'][0]}-Z{params['z_dim']}"
    print(data_dir)
    epochs = 400
    #vae, hist = load_and_prime_model(model,model_name,params,data_dir,epochs)

    # do i need to do this part?
    # filename = f"overtrain-{model_name}-kl_weight{params['kl_weight']:03d}.pkl"    
    # hist,p = ut.load_pickle(os.path.join(data_dir,filename))

    # vae = model(dim_z=p['z_dim'], dim_x=p['x_dim'], 
    #             learning_rate=0.0001, kl_weight=p['kl_weight'])
    vae = model(dim_z=params['z_dim'], dim_x=params['x_dim'],learning_rate=0.0001, kl_weight=params['kl_weight'])
    vae.compile(optimizer=vae.optimizer, loss = vae.partial_vae_loss)
    sv_path = os.path.join(data_dir,f"kl_weight{params['kl_weight']:03d}")
    vae.load_model(sv_path, epochs)

    return vae

# # plotVox
# def plotImg(imgIn, title="", tsnedata=None):

#     fig = plt.figure(figsize=(14, 8))
#     ax = fig.add_subplot(111, projection="3d")
#     ax.axis("off")
#     ax.set_xlim(0, cf_vox_size)
#     ax.set_ylim(0, cf_vox_size)
#     ax.set_zlim(0, cf_vox_size)
#     ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], linewidth=0.2, antialiased=True)

#     fig.canvas.draw()
#     data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep="")
#     data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
#     return data

#########################################################################
#%% Setup main methods
def vect_to_sneak():
    set_widemode_hack()

    header.title("Text 2 shape")
    loading_text = st.empty()
    loading_text.text("Making text encoder model..[not really].")

    vae = load_snkr_model()
    loading_text.text("Making shape generator model...")

    # loading_text.text("Getting Spacy Embeddings...")
    # vocab = getSpacy()
    # textmodel = makeTextModel()
    # loading_text.text("Models done being made!")

    description = st.text_input("Enter Sneaker Description:", value="Modern class, with a progressive vibe.  Urban story of high tech speed and lightness.  Yeezy and Jordan can eat their hearts out.  Originally released in 2003.")

    description = condition_text_input(description) #TODO: impliment conditioning

    # vox, encoding = getVox(description, shapemodel, textmodel, vocab)

    # verts, faces = createMesh(vox, step=1)
    # fig = showImg(verts, faces)
    # st.write(fig)

    # st.header("Similar descriptions:")
    # mid2desc = loadMid2Desc()
    # shape2vec, mids, vecs, vec_tree = loadShape2Vec()
    # _, close_ids = vec_tree.query(encoding, k=5)
    # close_ids = list(close_ids[0])
    # for i, index in enumerate(close_ids):
    #     try:
    #         mid = mids[int(index)]
    #         st.write("{}. {}".format(i + 1, mid2desc[mid]))
    #     except:
    #         continue

#%%

def vect_explore():
    set_widemode_hack()
    header.title("Vector exploration")



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
 
    model_options = model_names
    X_options = [str(x) for x in pix_dims]
    Z_options = [str(x) for x in latent_dims]
    kl_options = [str(x) for x in klws]
    
    model_option = st.sidebar.selectbox(
        "model",
        model_options,
    )

    klweight_option = st.sidebar.selectbox(
        "kl weight",
        kl_options,
    )

    X_option = st.sidebar.selectbox(
        "pix dim",
        X_options,
    )

    Z_option = st.sidebar.selectbox(
        "latent dim",
        Z_options,
    )

    color_option = st.sidebar.selectbox(
    "Color Data",
    [
        "Category",
        "Length",
        "Height",
        "Width",
        "Squareness",
        "Class Length",
        "Class Height",
        "Class Width",
        "Class Square",
        "Log Loss",
    ],)

    size = st.sidebar.number_input("Plot Dot Size", value=6.0, min_value=0.1, max_value=30.0, step=1.0)
    
    
    print(f"model:{model_option}, z-{Z_option}, x-{X_option}, kl-{klweight_option}")


    model_name = model_option
    model = model_names[0]
    
    params['x_dim'] = (X_option,X_option,3)
    params['z_dim'] = Z_option
    params['kl_weight'] = klweight_option

    data_df,snk2vec,latents, labels, vecs,vec_tree,mids = load_latent_data()

    bokeh_fig = init_bokeh_plot(data_df)

    st.write(bokeh_fig)

    
    #     padding = 1.05
    #     xmin, xmax = df_tsne.tsne1.min(), df_tsne.tsne1.max()
    #     ymin, ymax = df_tsne.tsne2.min(), df_tsne.tsne2.max()
    #     xlims, ylims = [xmin * padding, xmax * padding], [ymin * padding, ymax * padding]

    #     fig = px.scatter(
    #         data_frame=df_tsne.dropna(),
    #         range_x=xlims,
    #         range_y=ylims,
    #         hover_name="Category",
    #         hover_data=["Sub Categories", "Anno ID"],
    #         size="Width",
    #         x="tsne1",
    #         y="tsne2",
    #         color=color_option,
    #         color_discrete_sequence=bright,
    #         width=1400,
    #         height=900,
    #     )

    #     fig.update_traces(marker=dict(size=size, opacity=0.7, line=dict(width=0.1)))  # size=2.3

    #     # addMIDLines(df_tsne, fig)
    #     st.write(fig)
    #     addThumbnailSelections(df_tsne)


#%%

def sneaker_gen():
    set_widemode_hack()
    header.title("Snk Interpolations")
    subheader = st.subheader("Loading data...")

    # model_name = "K_PCVAE"
    klws = [1,2,3,5,10]
    latent_dims = [32,40,64]
    pix_dims = [128,160,192]

    p_names = ['z_dim','x_dim','kl_weight','batch_size']
    #models = [kcv.K_PCVAE, kcv.K_PCVAE_KL_Reg]
    model_names = ["K_PCVAE", "K_PCVAE_KL_Reg"]

    model_options = model_names
    X_options = [str(x) for x in pix_dims]
    Z_options = [str(x) for x in latent_dims]
    kl_options = [str(x) for x in klws]

    starting_model = st.sidebar.selectbox("Choose starting model:", model_options)
    starting_X = st.sidebar.selectbox("Choose starting img size:", X_options)
    starting_Z = st.sidebar.selectbox("Choose starting latent:", Z_options)
    starting_kl = st.sidebar.selectbox("Choose starting kl weight:", kl_options)
    
    data_df,snk2vec,latents, labels, vecs,vec_tree,mids =load_latent_data()
    
    header.title('Text 2 shape')
    loading_text = st.empty()
    
    loading_text.text('Getting Spacy Embeddings...[not]')
    #vocab = getSpacy()
    loading_text.text('Making text encoder model...[not]')
    #textmodel = makeTextModel()
    loading_text.text('Making shape generator model...[yes!]')
    vae = load_snkr_model()
    loading_text.text('Models done being made![not]')


    start_index = random.choice(mids)
    end_index = random.choice(mids)

    ## WHAT DOES THE JOURNEY DO?
 
    start_vect = snk2vec[start_index]
    end_vect = snk2vec[end_index]

    visited_indices = [start_index]

    # n_journeys = 1
    # for _ in range(n_journeys):
    journey_vecs = []
    journey_mids = []


    journey_mids.append(start_index)
    #subheader.subheader("Generating models... please wait...")
    journey_vecs.append(start_vect)

    n_steps = 5
    interp_vects = interp(end_vect, start_vect,divs=n_steps)

    closest = []
    for i,vect in enumerate(interp_vects):

        max_dist = 8
        n_dists, close_ids = vec_tree.query(vect, k=10, distance_upper_bound=max_dist)
        # if we got infinities do it again...
        if len(snk2vec) == close_ids[0]:
            n_dists, close_ids = vec_tree.query(
                start_vect, k=vects_sample, distance_upper_bound=max_dist * 3
            )
        #vector looks most like this:
        closest.append(close_ids[0])
        journey_mids.append(mids[close_ids[0]])

    journey_mids
    journey_vecs = interp_vects



    
  
  

    journey_imgs = np.zeros( (len(journey_vecs), params['x_dim'][0], params['x_dim'][1], params['x_dim'][2]) )
    for i, vect in enumerate(journey_vecs):
        journey_imgs[i,] = vae.decode(vect[None, ], apply_sigmoid=True)

    #subheader.subheader("Showing models... (may have to scroll down)")
    
    fig, ax = plt.subplots(1,n_steps,figsize=(16, 4))
    ax = ax.flatten()
    
    for i in range(n_steps):
        im = journey_imgs[i,:,:,:].squeeze()
        # ut.plot_img(im, 
        #         title="", 
        #         stats=False, 
        #         limits=None, 
        #         show_axes=True, save_fig=False, show_fig=True, threshold=None)

        # empty.image(data)
        ax[i].imshow(im.squeeze())
        # plt.colorbar()


    subheader.subheader("All done!")

  
    # # this is the flow for making image from text...
    # # get description
    # description = st.text_input('Enter Shape Description:', value='a regular chair with four legs.')
    # description = conditionTextInput(description)
    # # encode into latent space
    # #    vox, encoding = getVox(description, shapemodel, textmodel, vocab)
    # # create the image
    # verts, faces = createMesh(vox, step=1)
    # # show the image
    # fig = show_image(verts, faces)
    st.write(fig)


def show_example_img():
    img = get_image("media/Jordan.jpg")
    return show_im(img)
    


def manual():
    
    example_descriptions = loadExampleDescriptions()
    header.title("Streamlit App Manual")
    st.write(
        """
            This is my streamlit app for version 2.0 of my Insight DS.SV.2020A project: SnkrFinder  (formerly http:/ergodatainsights.tech)
            
            See slides related to the development of this app [here](http://bit.ly/SneakerFinder-slides) 
            and the github repo with code [here](https://github.com/ergonyc/SneakerFinder.
            
            **Below is an example of what can be generated. Input was xxx**
            """
    )
    # just a random number
    example_sneaker = 555

    fig = show_example_img()
        
    st.write(fig)

    st.write(
        """
            ## Available Tabs:            
            - ### Text to sneaker generator
            - ### Latent vector exploration
            - ### Sneaker interpolation
            
            ## Text to Sneaker Generator
            This tab allows you to input a description and the generator will make a model based on that description.
            The 3D plotly viewer generally works much faster in Firefox compared to chrome so use that if chrome is being slow.
            
            The bottom of this tab shows similar descriptions to the input description. Use these samples to see new designs and 
            learn how the model interprets the text.
            
            #### Models were trained on data scraped from:
            - goat.com
            - sneakersnstuff.com
            
            """
    )

    if st.button("-->Click here to get some random example descriptions<--"):
        descs = random.sample(example_descriptions, 5)
        for d in descs:
            st.write(d)

    st.write(
        """    
            ## Latent Vector Exploration
            This tab shows the plot of the shape embedding vectors reduced from the full model dimensionality of 128 dimensions
            down to 2 so they can be viewed easily. The method for dimensionality reduction was UMAP.  (TSNE also available)
            
            #### In the exploration tab, there are several sidebar options:
            - **Color data**
                - This selector box sets what determines the color of the dots. (the class selections are particularly interesting!)
            - Plot dot size
                - This sets the dot size. Helpful when zooming in on a region.
            - Model IDs 
                - This allows for putting in multiple model IDs to see how they're connected on the graph.
            - **Anno IDs to view**
                - From the hover text on the TSNE plot points you can see the 'Anno ID' (annotation ID) and enter it into this box to see a render of the object and 1 of its generated descriptions.
                - Multiple IDs can be entered and separated by commas.
                - The renders can be viewed in the sidebar or in the main area below the TSNE graph.
            
            Hover for sneaker image & reconstructed version
            
            The embeddings are very well clustered according to differt footwear classes -- slippers, sneakers, and boots 
            (other shoes excluded)  By playing with the color data, it can be seen that the clusters are also organized very strongly
            by specific attributes about the sneakers such as i...

            ### UMAP map showing different colors for the different shape classes:            
                """
    )

    tsne_pic = os.path.join(os.getcwd(), "media/umap_small.png")
    img = mpimg.imread(tsne_pic)
    st.image(img, use_column_width=True)

    st.write(
        """    
            ## Sneaker Interpolation
            This tab is just for fun and is intended to show how well the model can interpolate between example sneakers. 
            Note that this runs the model many times and as such can be quite slow online. You may need to hit 'stop' 
            and then 'rerun' from th menu in the upper right corner to make it behave properly.
            
            To generate these plots, the algorithm finds the nearest K shape embedding vectors
            (K set by the variety parameter in the sidebar) and randomly picks one of them.
            Then it interpolates between the current vector and the random new vector
            and at every interpolated point it generates a new model from the interpolated latent space vector.
            Then it repeats indefinitely finding new vectors as it goes.
            
            #### In this tab there are 2 sidebar options:
            - Starting sneaker
                - This sets the starting category for the algorithm but it will likely wander off into other categories
                after a bit
            - Variety parameter
                - This determines the diversity of the models by setting how many local vectors to choose from.
        
            ### Example pre-rendered gif below:
                """
    )

    # this just grabs givs adn renders them...
    cat_options = [
        "Jordan",
        "Yeezy",
    ]
    # gif_urls = [
    #     "https://github.com/starstorms9/shape/blob/master/media/{}.jpg?raw=true".format(cat.lower())
    #     for cat in cat_options
    # ]
    gif_urls = [
         "media/{}.jpg".format(cat)
         for cat in cat_options
     ]
    selected_cat = st.selectbox("Select a category to see shape interpolations", cat_options, index=0)
    gif_url = gif_urls[cat_options.index(selected_cat)]
    
    st.image(gif_url, use_column_width=True)




#%% Main selector system
# 
#  
# 
# 
#################################

if server_up:
    modeOptions = ["Manual", "Text to Shape", "Latent Vect Exploration", "Shape Interpolation"]
    st.sidebar.header("Select Mode:")
    mode = st.sidebar.radio("", modeOptions, index=0)

    tabMethods = [manual, vect_to_sneak, vect_explore, sneaker_gen]
    tabMethods[modeOptions.index(mode)]()



#%%

###############################################
###
####################################################



def condition_text_input(text):
    # this should be a text scrubber... 
    print("condition_text_input(text) not implimented")    
    return text

def addThumbnailSelections(df_):
    annoid_input = st.sidebar.text_input("Anno IDs to view (comma separated, from plot):")
    sidebar_renders = st.sidebar.checkbox("Show renders in sidebar?")
    if len(annoid_input) > 1:
        annosinspect = [
            annoid.strip().replace("'", "").replace("[", "").replace("]", "")
            for annoid in re.split(",", annoid_input)
            if len(annoid) > 1
        ]

        #pic_in_dir = "https://starstorms-shape.s3-us-west-2.amazonaws.com/renders/"
        mid2desc = loadMid2Desc()

        for i, aid in enumerate(annosinspect):
            try:
                mid = df_tsne[df_tsne["Anno ID"] == int(aid)]["Model ID"].values[0]
                fullpath = os.path.join(pic_in_dir, mid + ".jpg")
                img = mpimg.imread(fullpath)

                if sidebar_renders:
                    st.sidebar.text("Annod ID: {}".format(annosinspect[i]))
                    desc_empty = st.sidebar.empty()
                    st.sidebar.image(img, use_column_width=True)
                    desc_empty.text(mid2desc[mid])
                else:
                    st.text("Annod ID: {}".format(annosinspect[i]))
                    desc_empty = st.empty()
                    st.image(img, use_column_width=False)
                    desc_empty.text(mid2desc[mid])
            except:
                if sidebar_renders:
                    st.sidebar.text("Could not find {}".format(annosinspect[i]))
                else:
                    st.text("Could not find {}".format(annosinspect[i]))


def addMIDLines(df_tsne, fig):
    midslist = list(df_tsne["Model ID"])
    mids_input = st.sidebar.text_area("Model IDs (comma separated)")
    midsinspect = [
        mid.strip().replace("'", "").replace("[", "").replace("]", "")
        for mid in re.split(",", mids_input)
        if len(mid) > 20
    ]
    some_found = False
    for mid in midsinspect:
        found = mid in midslist
        some_found = some_found or found
        if not found:
            st.sidebar.text("{} \n Not Found".format(mid))

    if some_found:
        midpoints = [
            [df_tsne.tsne1[midslist.index(mid)], df_tsne.tsne2[midslist.index(mid)]]
            for mid in midsinspect
            if (mid in midslist)
        ]
        dfline = pd.DataFrame(midpoints, columns=["x", "y"])
        fig.add_scatter(
            name="Between Models",
            text=dfline.index.values.tolist(),
            mode="lines+markers",
            x=dfline.x,
            y=dfline.y,
            line=dict(width=5),
            marker=dict(size=10, opacity=1.0, line=dict(width=5)),
        )
    return



# %%
