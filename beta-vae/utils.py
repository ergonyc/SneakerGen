"""
This file is a repository of commonly used methods for a variety of different parts of the program.

The basic categories are:
    1. Basic math functions
    2. Data loading and modifying methods
    3. 3D model loading and modification methods
    4. Various 2D and 3D plotting methods
"""

#%% Imports
import numpy as np
import os
import subprocess
import re

# import skimage.measure as sm
import pickle
import time
import json
import pandas as pd
import random
from tqdm import tqdm
from scipy import signal
from shutil import copyfile
import configs as cf
import glob as glob

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# import plotly.graph_objects as go
# from plotly.offline import plot
# import plotly.figure_factory as FF
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

#%% Global variables
plot_out_dir = "."
tax = []
meta = []
kernel = "none"

#%% Basic Functions
def minmax(arr):
    """
    Prints the min and max of array
    Parameters
    ----------
    arr : np array
        Numpy array to find min and max of
    Returns
    -------
    None, only prints out the min and max
    Examples
    --------
    > minmax(np.array([1,2,3,4,5]))
    Min: 1
    Max: 5
    """
    print("Min: {} \nMax: {}".format(np.min(arr), np.max(arr)))


def interp(vec1, vec2, divs=5, include_ends=True):
    """
    Interpolates between 2 arbitrary dimension vectors and returns the result.
    Parameters
    ----------
    vec1 : np array
        Starting vector to interpolate from.
    vec2 : np array
        Ending vector to interpolate to.
    divs : integer
        Number of divisions between the vectors.
    include_ends: bool
        Wether to include the starting and ending vectors in the output array.
    Returns
    -------
    Numpy array with 1 extra dimension of size divs (+ 2 if include_ends=True).
    """
    out = []
    amounts = np.array(range(divs + 1)) / divs if include_ends else np.array(range(1, divs)) / divs
    for amt in amounts:
        interpolated_vect = vec1 * amt + vec2 * (1 - amt)
        out.append(interpolated_vect)
    return np.stack(out)


def super_sample(list_to_sample, samples):
    """
    Samples from a list. If requested number of samples exceeds the length of the list, it repeats.

    Parameters
    ----------
    list_to_sample : list
        List to sample from.
    samples : integer
        Number of samples to return. May be larger or smaller than the length of list_to_sample.

    Returns
    -------
    List of length samples that contains randomly shuffled samples from the list_to_sample.
    """
    pop = len(list_to_sample)
    if samples > pop:
        result = random.sample(list_to_sample, pop)
        for i in range(samples - pop):
            result.append(random.choice(list_to_sample))
        return result
    else:
        return random.sample(list_to_sample, samples)


#%% Data methods
def dump_pickle(filepath, item_to_save):
    f = open(filepath, "wb")
    pickle.dump(item_to_save, f)
    f.close()


def load_pickle(filepath):
    infile = open(filepath, "rb")
    item = pickle.load(infile)
    infile.close()
    return item


def add_time_stamp(path=""):
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    return path + time.strftime("%m%d-%H%M")


def get_sub_dirs(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]



##############################################
##############################################

def parse_function(filename, label):
    image_string = tf.read_file(filename)

    # Don't use tf.image.decode_image, or the output shape will be undefined
    image = tf.image.decode_jpeg(image_string, channels=3)

    # This will convert to float values in [0, 1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    image = tf.image.resize_images(image, [64, 64])
    return resized_image, label


def zoom_in(image, zoomf, pad_value=1.0):
    # returns a square image the size of the inital width
    shape_f = tf.cast(tf.shape(image), tf.float32)
    initial_height, initial_width = shape_f[0], shape_f[1]
    if initial_height != initial_width:
        print("WARNING:  the imput image should have been square!!!")
        print(shape_f)

    x1 = 0 - (1 - zoomf) / 2.0  # * (initial_width - 1)
    x2 = 1 + (1 - zoomf) / 2.0  # * (initial_width - 1)
    y1 = 0 - (1 - zoomf) / 2.0  # * (initial_width - 1)
    y2 = 1 + (1 - zoomf) / 2.0  # * (initial_width - 1)
    box = [y1, x1, y2, x2]
    image = tf.image.crop_and_resize(
        tf.expand_dims(image, 0),
        boxes=[box],
        box_indices=[0],
        crop_size=[initial_width, initial_width],
        extrapolation_value=pad_value,
    )
    return image


def load_and_convert(filename):
    image_string = tf.io.read_file(filename)
    im = tf.image.decode_jpeg(image_string, channels=3)

    label = filename
    return tf.cast(im, tf.float32) / 255.0, label


def load_square_and_augment(filename, img_size=64):
    pad_value = 1.0
    # ZOOMS between 100 and 110
    # random flip..
    zoomf = tf.random.uniform(shape=[1], minval=100, maxval=110)
    pad_value = 1.0
    image, label = load_and_convert(filename)

    shape_f = tf.cast(tf.shape(image), tf.float32)
    shape_f = tf.cast(tf.shape(image), tf.float32)
    if len(shape_f) > 3:
        initial_height, initial_width = shape_f[1], shape_f[2]
    else:
        initial_height, initial_width = shape_f[0], shape_f[1]
        image = tf.expand_dims(image, 0)

    delta_y = (initial_width - initial_height) / initial_width
    delta_z = zoomf - 1.0

    x1 = 0 + 0.5 * (-delta_z)  # * (initial_width - 1)
    x2 = 1 + 0.5 * delta_z  # * (initial_width - 1)

    y1 = 0.5 * (-delta_y - delta_z)
    y2 = 1 + 0.5 * (delta_y + delta_z)

    delta_y = 0.5 * (initial_width - initial_height) / initial_width

    x1 = 0.0  # - 0.025 * (initial_width - 1)
    x2 = 1.0  # + 0.025 * (initial_width - 1)
    y1 = 0.0 - delta_y
    y2 = 1.0 + delta_y

    box = [y1, x1, y2, x2]

    image = tf.image.crop_and_resize(
        image,
        boxes=[box],
        box_indices=[0],
        crop_size=[img_size, img_size],
        method="bilinear",
        extrapolation_value=pad_value,
        name=None,
    )

    if tf.random.uniform(()) > 0.5:
        # random mirroring
        print("flipping  ")
        image = tf.image.flip_left_right(image)

    # label = tf.constant(-1, tf.int32)
    image = tf.squeeze(image)
    return image, label


# load and resize with make the image square
def load_and_square(filename, img_size=64):
    pad_value = 1.0
    image, label = load_and_convert(filename)

    shape_f = tf.cast(tf.shape(image), tf.float32)
    print(f"shape = {shape_f}, len = {len(shape_f)}")
    shape_f = tf.cast(tf.shape(image), tf.float32)
    if len(shape_f) > 3:
        initial_height, initial_width = shape_f[1], shape_f[2]
    else:
        initial_height, initial_width = shape_f[0], shape_f[1]
        image = tf.expand_dims(image, 0)
        print("expand now... squeeze later")
    print(f"height = {initial_height},w = {initial_width}")

    delta_y = (initial_width - initial_height) / initial_width

    x1 = 0.0  # - 0.025 * (initial_width - 1)
    x2 = 1.0  # + 0.025 * (initial_width - 1)
    y1 = 0.0 - 0.5 * delta_y
    y2 = 1.0 + 0.5 * delta_y

    # box = [0.0 - delta_y, 0.0, 1.0 + delta_y, 1.0]
    box = [y1, x1, y2, x2]

    image = tf.image.crop_and_resize(
        image,
        boxes=[box],
        box_indices=[0],
        crop_size=[img_size, img_size],
        method="bilinear",
        extrapolation_value=pad_value,
        name=None,
    )

    # # image = tf.image.crop_and_resize(
    #     image, offset_height, offset_width, target_size, target_size, constant_values=pad_value
    # )
    # label = tf.constant(filename, tf.string)
    image = tf.squeeze(image)
    return image, label


# all_voxs, all_mids = ut.loadData(cf_img_size, cf_max_loads_per_cat, lg.vox_in_dir, cf_cat_prefixes)
def load_data(target_size, files):
    """[summary]

    Args:
        target_size ([int]): [number of pixels]
        files ([list of stirngs]): [list of filepaths ]

    Returns:
        [data]: [data from slices]
    """
    ds = tf.data.Dataset.from_tensor_slices(files)

    return ds


def load_and_prep_data(target_size, ds, augment=False):
    """[summary]

    Args:
        target_size ([int]): [description]
        ds ([dataset]): [directory or list of files]
        cf_batch_size ([int]): [description]

    Returns:
        [type]: [description]
    """
    # LOAD

    # SPLIT
    # TEST_FRAC = 20.0 / 100.0
    # train_dataset, ds = splitShuffleData(ds, TEST_FRAC)
    # train_dataset, test_dataset = splitShuffleData(ds, TEST_FRAC)
    # take
    # val_dataset, test_dataset = splitData(ds, 0.5)

    # PREP
    if augment:
        prep = lambda x: load_square_and_augment(x, target_size)
    else:
        prep = lambda x: load_and_square(x, target_size)

    dataset = ds.map(prep, num_parallel_calls=AUTOTUNE)

    # train_dataset = train_dataset.batch(cf_batch_size, drop_remainder=True)
    # train_dataset = train_dataset.prefetch(AUTOTUNE)

    return dataset  # , validate


def load_and_prep_for_testing(target_size, input, cf_batch_size):
    """[summary]

    Args:
        target_size ([int]): [description]
        inputs ([list or str]): [directory or list of files]
        cf_batch_size ([int]): [description]

    Returns:
        [type]: [description]
    """
    files = input

    ds = loadData(target_size, files)
    # just returns the files
    prep_and_augment = lambda x: load_square_and_augment(x, target_size)

    batch1 = ds.map(prep_and_augment, num_parallel_calls=AUTOTUNE)
    batch2 = ds.map(prep_and_augment, num_parallel_calls=AUTOTUNE)
    return batch1, batch2


##############################################
##############################################
##############################################


def read_header(fp):
    line = fp.readline().strip()
    if not line.startswith(b"#binvox"):
        raise IOError("Not a binvox file")
    dims = list(map(int, fp.readline().strip().split(b" ")[1:]))
    translate = list(map(float, fp.readline().strip().split(b" ")[1:]))
    scale = list(map(float, fp.readline().strip().split(b" ")[1:]))[0]
    line = fp.readline()
    return dims, translate, scale


def get_JSON(json_fp, df=False):
    if df:
        json_file = pd.read_json(json_fp)
    else:
        with open(json_fp, "r") as json_file:
            json_file = json.load(json_file)
    return json_file


def read_meta():
    global meta
    meta = pd.read_csv(cf.META_DATA_CSV)
    return meta


#%% 3D Model Functions


def show_pic(modelid, title="", pic_in_dir=""):
    if pic_in_dir == "":
        pic_in_dir = cf.RENDERS_DIR
    fullpath = os.path.join(pic_in_dir, modelid + ".png")
    img = mpimg.imread(fullpath)
    plt.suptitle(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def anno_to_mID(annoid):
    return meta[meta.annoid == annoid].mid.values[0]


def get_metric(original_voxs, generated_voxs):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(original_voxs, generated_voxs)).numpy() * 1000


def check_stop_signal(dir_path="/data/sn/all/"):
    stop_path = os.path.join(dir_path, "stop")
    go_path = os.path.join(dir_path, "go")
    if os.path.isdir(stop_path):
        os.rename(stop_path, go_path)
        return True
    else:
        return False


#%% Display Functions
# Create gif from images in folder with bash and ImageMagick (replace XX with max number of images or just set high and error)
# !convert -delay 15 -loop 0 *{000..XXX}.png car2truck.gif
def make_gif_from_dir(gif_in_dir, name):
    """
    This handy method takes a directory as input that has the saved files from from matplotlib plots and renames them properly so they can be turned into a bouncing cyclic gif.
    Primarily used for creating the interpolation animations.
    """
    natsort = lambda l: sorted(l, key=lambda x: int(re.sub("\D", "", x) or 0))
    pngs = natsort(list(os.listdir(gif_in_dir)))

    total_len = len(pngs) * 2
    for i, fp in enumerate(pngs):
        fullpath = os.path.join(gif_in_dir, fp)
        fixpath = os.path.join(gif_in_dir, "Figure_{:03d}.png".format(i))
        newpath = os.path.join(gif_in_dir, "Figure_{:03d}.png".format(total_len - i))
        copyfile(fullpath, newpath)
        os.rename(fullpath, fixpath)
    subprocess.call(
        "convert -delay 15 -loop 0 " + gif_in_dir + "*{000..XXX}.png " + str(name) + ".gif", shell=True
    )


def _padEnc(text, vocab):
    texts = text if type(text) == list else [text]
    lexs = [[vocab[t].rank for t in sent.replace(".", " . ").split(" ") if len(t) > 0] for sent in texts]
    ranks = [[l.rank for l in lex if not l.is_oov] for lex in lexs]
    padded = pad_sequences(ranks, maxlen=cf_max_length, padding=cf_padding_type, truncating=cf_trunc_type)
    return padded


def _getImg(text, snkmodel, textmodel, nlp):
    ptv = padEnc(text, nlp)
    preds = textmodel.sample(ptv)
    img = snkmodel.sample(preds).numpy()[0, ..., 0]
    return img, preds


def plot_img(
    imgin, title="", stats=False, limits=None, show_axes=True, save_fig=False, show_fig=True, threshold=None
):

    img = np.squeeze(imgin)
    fig = plt.figure()
    ax = fig.add_subplot(121)

    if stats:
        plt.hist(imgin.flatten(), bins=10)

    vflat = img.flatten()
    if threshold is None:
        threshold = (np.min(vflat) + np.max(vflat)) / 2

    if np.any(img) is False:
        print("No image for: {}".format(title))
        vflat = imgin.flatten()
        plt.hist(vflat, bins=10)
        plt.suptitle(title)
        return

    if stats:
        ax = fig.add_subplot(122)
        # plt.subplot(1, 2, 2)
        # pos = plt.imshow(img)
        # plt.suptitle(title)
        # plt.show()
        # return

    if not show_axes:
        ax.axis("off")

    pos = plt.imshow(img)
    plt.suptitle(title)
    # plt.colorbar(pos, ax=ax)

    global plot_out_dir
    if save_fig:
        _ = fig.savefig(os.path.join(plot_out_dir, title))
    if show_fig:
        _ = plt.show()

    # DO WE NEED TO CLOSE THE PLOT??
    return


def plot_img_and_vect(
    imgin,
    latent_vect,
    title="",
    stats=False,
    limits=None,
    show_axes=True,
    save_fig=False,
    show_fig=True,
    threshold=None,
):

    img = np.squeeze(imgin)
    fig = plt.figure()
    ax = fig.add_subplot(121)

    if stats:
        data = latent_vect.flatten()
        plt.barh([x for x in range(len(data))], data, height=1.2)

        ax.set_xlim(limits)

    else:
        pos = plt.imshow(latent_vect.reshape(latent_vect.flatten().shape[0] // 8, 8))
        plt.colorbar(pos, ax=ax)

    vflat = img.flatten()
    if threshold is None:
        threshold = (np.min(vflat) + np.max(vflat)) / 2

    if np.any(img) is False:
        print("No image for: {}".format(title))
        vflat = imgin.flatten()
        plt.hist(vflat, bins=10)
        plt.suptitle(title)
        return

    ax = fig.add_subplot(122)

    if not show_axes:
        ax.axis("off")

    pos = plt.imshow(img)
    plt.suptitle(title)
    # plt.colorbar(pos, ax=ax)

    global plot_out_dir
    if save_fig:
        _ = fig.savefig(os.path.join(plot_out_dir, title))
    if show_fig:
        _ = plt.show()
    return


def show_reconstruct(
    model, samples, index=0, title="", show_original=True, show_reconstruct=True, save_fig=False, limits=None
):
    predictions = model.reconstruct(samples[index][None, ...], training=False)
    xvox = samples[index][None, ...]

    if np.max(predictions) < 0.5:
        print("No voxels")
        return

    if show_original:
        plot_img(
            xvox, title="Original {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )
    if show_reconstruct:
        plot_img(
            predictions, title="Reconstruct {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )


def load_and_dump(target_size, input):
    """[summary]

    Args:
        target_size ([int]): [description]
        inputs ([list or str]): [directory or list of files]

    Returns:
        [EagerTensor/dataset?]: [dataset. cropped, squared]
    """
    # LOAD
    if isinstance(input, str):
        files = glob.glob(os.path.join(input, "*/img/*"))
    else:  # type(input) == 'list'
        files = input

    ds = loadData(target_size, files)
    # just returns the files

    # PREP
    prep = lambda x: load_and_square(x, target_size)

    # prep_and_augment = lambda x: load_square_and_augment(x, target_size)

    ds = ds.map(prep, num_parallel_calls=AUTOTUNE)

    # validate = val_dataset.batch(cf_batch_size, drop_remainder=False)
    return ds  # , validate


def dump_reconstruct(model, samples, test_samples):
    """[dumps the model encoding and lossess]

    Args:
        model ([CVAE]): [the model]
        samples ([EagerTensor]): [train_dataset]
        test_samples ([EagerTensor]): [test+dataset]

    Returns:
        [tuple]: [list of encodings and list of losses]
    """
    preds = []
    losses = []
    # for index in range(len())
    for train_x, label in samples:
        # sample = tf.cast(index, dtype=tf.float32)
        # predictions = model.reconstruct(samples[index][None, ...], training=False)
        # predictions = model.reconstruct(sample, training=False)
        preds.append(model.encode(train_x, reparam=True).numpy()[0])
        losses.append(model.compute_loss(train_x).numpy())

    for train_x, label in test_samples:
        # sample = tf.cast(index, dtype=tf.float32)
        # predictions = model.reconstruct(samples[index][None, ...], training=False)
        # predictions = model.reconstruct(sample, training=False)
        preds.append(model.encode(train_x, reparam=True).numpy()[0])
        losses.append(model.compute_loss(train_x).numpy())

    return preds, losses


def start_streamlit(filepath):
    subprocess.call("streamlit run {}".format(filepath), shell=True)
    subprocess.call("firefox new-tab http://localhost:8501/")


#%% For plotting meshes side by side
def map_z2color(zval, colormap, vmin, vmax):
    # map the normalized value zval to a corresponding color in the colormap

    if vmin > vmax:
        raise ValueError("incorrect relation between vmin and vmax")
    t = (zval - vmin) / float((vmax - vmin))  # normalize val
    R, G, B, alpha = colormap(t)
    return (
        "rgb("
        + "{:d}".format(int(R * 255 + 0.5))
        + ","
        + "{:d}".format(int(G * 255 + 0.5))
        + ","
        + "{:d}".format(int(B * 255 + 0.5))
        + ")"
    )


def tri_indices(simplices):
    # simplices is a numpy array defining the simplices of the triangularization
    # returns the lists of indices i, j, k

    return ([triplet[c] for triplet in simplices] for c in range(3))
