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


def superSample(list_to_sample, samples):
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
def dumpPickle(filepath, item_to_save):
    f = open(filepath, "wb")
    pickle.dump(item_to_save, f)
    f.close()


def loadPickle(filepath):
    infile = open(filepath, "rb")
    item = pickle.load(infile)
    infile.close()
    return item


def addTimeStamp(path=""):
    os.environ["TZ"] = "US/Pacific"
    time.tzset()
    return path + time.strftime("%m%d-%H%M")


def getSubDirs(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]


# def getMixedFPs(vox_in_dir, num_models_load, cat_prefixes):
#     vox_fps = os.listdir(vox_in_dir)
#     cat_vox_fps = [
#         [
#             os.path.join(cat, directory, "models/model_normalized.solid.binvox")
#             for directory in os.listdir(os.path.join(vox_in_dir, cat))
#         ]
#         for cat in cat_prefixes
#     ]
#     cat_vox_fps = [superSample(cat, int(num_models_load / len(cat_prefixes))) for cat in cat_vox_fps]

#     vox_fps = []
#     for cat_list in cat_vox_fps:
#         vox_fps.extend(cat_list)
#     return vox_fps


##############################################
##############################################


##############################################
##############################################
# all_voxs, all_mids = ut.loadData(cf_vox_size, cf_max_loads_per_cat, lg.vox_in_dir, cf_cat_prefixes)
CLASS_NAMES = ["goat", "sns"]


# def get_label(file_path):
#     # convert the path to a list of path components
#     parts = tf.strings.split(file_path, os.path.sep)
#     # The second to last is the class-directory
#     # out=parts.one_hot(CLASS_NAMES, 3)
#     return parts[-2] == CLASS_NAMES


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
    # image = tf.image.decode_jpeg(image_string, channels=3)
    # label = tf.constant(-1, tf.int32)
    # return image, label
    # return tf.cast(im, tf.float32) / 255.0, tf.constant(-1, tf.int32)
    # label = tf.constant(filename, tf.string)
    label = filename
    return tf.cast(im, tf.float32) / 255.0, label

    # image = tf.image.convert_image_dtype(image, tf.float32)  # Cast and normalize the image to [0,1]


def load_square_and_augment(filename, img_size=64):
    pad_value = 1.0
    # ZOOMS between 100 and 110
    # random flip..
    zoomf = tf.random.uniform(shape=[1], minval=100, maxval=110)
    print(f"zoomfactor = {zoomf[0]}")
    print(f"zoomfactor(cast) = {tf.cast(zoomf[0], tf.float32)}")

    pad_value = 1.0
    image, label = load_and_convert(filename)
    # image_string=tf.io.read_file(filename)
    # image=tf.image.decode_jpeg(image_string,channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)

    shape_f = tf.cast(tf.shape(image), tf.float32)
    print(f"shape = {shape_f}, len = {len(shape_f)}")
    shape_f = tf.cast(tf.shape(image), tf.float32)
    if len(shape_f) > 3:
        initial_height, initial_width = shape_f[1], shape_f[2]
        print("already expanded")
    else:
        initial_height, initial_width = shape_f[0], shape_f[1]
        image = tf.expand_dims(image, 0)
        print("expanding")
    print(f"height = {initial_height},w = {initial_width}")

    # x1 = 0.0  # - 0.025 * (initial_width - 1)
    # x2 = 1.0  # + 0.025 * (initial_width - 1)
    # y1 = 0.0 - 0.5 * (initial_width - initial_height) / initial_width
    # y2 = 1.0 + 0.5 * (initial_width - initial_height) / initial_width

    delta_y = (initial_width - initial_height) / initial_width
    delta_z = zoomf - 1.0

    x1 = 0 + 0.5 * (-delta_z)  # * (initial_width - 1)
    x2 = 1 + 0.5 * delta_z  # * (initial_width - 1)

    y1 = 0.5 * (-delta_y - delta_z)
    y2 = 1 + 0.5 * (delta_y + delta_z)

    # x1 = 0.0  # - 0.025 * (initial_width - 1)
    # x2 = 1.0  # + 0.025 * (initial_width - 1)
    # y1 = 0.0 - 0.5 * (initial_width - initial_height) / initial_width
    # y2 = 1.0 + 0.5 * (initial_width - initial_height) / initial_width
    delta_y = 0.5 * (initial_width - initial_height) / initial_width

    x1 = 0.0  # - 0.025 * (initial_width - 1)
    x2 = 1.0  # + 0.025 * (initial_width - 1)
    y1 = 0.0 - delta_y
    y2 = 1.0 + delta_y

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
    # image_string = tf.io.read_file(filename)
    # image = tf.image.decode_jpeg(image_string, channels=3)
    # image = tf.image.convert_image_dtype(image, tf.float32)

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

    # x1 = 0.0  # - 0.025 * (initial_width - 1)
    # x2 = 1.0  # + 0.025 * (initial_width - 1)
    # y1 = 0.0 - 0.5 * (initial_width - initial_height) / initial_width
    # y2 = 1.0 + 0.5 * (initial_width - initial_height) / initial_width
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
def loadData(target_size, files):
    """[summary]

    Args:
        target_size ([int]): [number of pixels]
        files ([list of stirngs]): [list of filepaths ]

    Returns:
        [data]: [data from slices]
    """
    ds = tf.data.Dataset.from_tensor_slices(files)

    return ds


def loadAndPrepData(target_size, ds, augment=False):
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


def loadAndPrepDataForTesting(target_size, input, cf_batch_size):
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


def getJSON(json_fp, df=False):
    if df:
        json_file = pd.read_json(json_fp)
    else:
        with open(json_fp, "r") as json_file:
            json_file = json.load(json_file)
    return json_file


def readTax(tax_fn):
    global tax
    tax = pd.read_json(tax_fn)
    tax["numc"] = tax.apply(lambda row: len(row.children), axis=1)
    return tax


def readMeta():
    global meta
    meta = pd.read_csv(cf.META_DATA_CSV)
    return meta


def getMidCat(modelid):
    global meta
    if len(meta) < 2:
        meta = readMeta()
    return meta.cat[meta.mid == modelid].to_numpy()[0]


def getCats(labels_tensor, cf_cat_prefixes):
    output = ["0{}".format(getMidCat(item.numpy().decode())) for item in labels_tensor]
    outcats = [
        cf_cat_prefixes.index(item) if item in cf_cat_prefixes else len(cf_cat_prefixes) for item in output
    ]
    return tf.convert_to_tensor(outcats, dtype=tf.int32)


def getCatName(catid):
    global tax
    if len(tax) == 0:
        tax = readTax()
    return tax.name[tax.synsetId == int(catid)].to_numpy()[0]


def renameVoxs(vox_in_dir, prefix):
    for i, file in enumerate(os.listdir(vox_in_dir)):
        fullpath = os.path.join(vox_in_dir, file)
        newpath = os.path.join(vox_in_dir, "{}_{:05d}.binvox".format(prefix, i))
        print(fullpath, "\n", newpath, "\n")
        os.rename(fullpath, newpath)


#%% 3D Model Functions


def showPic(modelid, title="", pic_in_dir=""):
    if pic_in_dir == "":
        pic_in_dir = cf.RENDERS_DIR
    fullpath = os.path.join(pic_in_dir, modelid + ".png")
    img = mpimg.imread(fullpath)
    plt.suptitle(title)
    plt.imshow(img)
    plt.axis("off")
    plt.show()


def annoToMid(annoid):
    return meta[meta.annoid == annoid].mid.values[0]


def getMetric(original_voxs, generated_voxs):
    return tf.reduce_mean(tf.keras.losses.mean_squared_error(original_voxs, generated_voxs)).numpy() * 1000


def checkStopSignal(dir_path="/data/sn/all/"):
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
def makeGifFromDir(gif_in_dir, name):
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


# def plotMesh(verts, faces):
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection="3d")
#     ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], linewidth=0.2, antialiased=True)
#     plt.show()


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


def plotImg(
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
    return


def plotImgAndVect(
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


def showReconstruct(
    model, samples, index=0, title="", show_original=True, show_reconstruct=True, save_fig=False, limits=None
):
    predictions = model.reconstruct(samples[index][None, ...], training=False)
    xvox = samples[index][None, ...]

    if np.max(predictions) < 0.5:
        print("No voxels")
        return

    if show_original:
        plotImg(
            xvox, title="Original {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )
    if show_reconstruct:
        plotImg(
            predictions, title="Reconstruct {}".format(title), stats=False, save_fig=save_fig, limits=limits,
        )


def loadAndDump(target_size, input):
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


def dumpReconstruct(model, samples, test_samples):
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


def startStreamlit(filepath):
    subprocess.call("streamlit run {}".format(filepath), shell=True)
    subprocess.call("firefox new-tab http://localhost:8501/")


# def exportBinvoxes(in_dir, out_dir, obj_prefix, vox_size):
#     # Remove any .binvox files in directory
#     subprocess.call("rm {}/*.binvox".format(in_dir), shell=True)
#     # Create binvox files     In bash it's this:    for f in objs/{obj_prefix}*.obj; do file=${f%%.*}; ./binvox ${file}.obj -pb -d {vox_size};  done;
#     subprocess.call(
#         "for f in {}/{}*.obj; do file=${{f%%.*}}; ./binvox ${{file}}.obj -pb -d {};  done;".format(
#             in_dir, obj_prefix, vox_size
#         ),
#         shell=True,
#     )
#     # Move binvox files to output dir
#     subprocess.call("mv {}/*.binvox {}".format(in_dir, out_dir), shell=True)


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


# def plotly_trisurf(x, y, z, simplices, colormap=cm.RdBu, plot_edges=None):
#     """
#     This function plots 3D meshes with plotly. It was copied from plotly documentation.
#     """
#     # x, y, z are lists of coordinates of the triangle vertices
#     # simplices are the simplices that define the triangularization;
#     # simplices is a numpy array of shape (no_triangles, 3)

#     points3D = np.vstack((x, y, z)).T
#     tri_vertices = list(map(lambda index: points3D[index], simplices))  # vertices of the surface triangles
#     zmean = [np.mean(tri[:, 2]) for tri in tri_vertices]  # mean values of z-coordinates of triangle vertices
#     min_zmean = np.min(zmean)
#     max_zmean = np.max(zmean)
#     facecolor = [map_z2color(zz, colormap, min_zmean, max_zmean) for zz in zmean]
#     I, J, K = tri_indices(simplices)

#     triangles = go.Mesh3d(x=x, y=y, z=z, facecolor=facecolor, i=I, j=J, k=K, name="")

#     if plot_edges is None:  # the triangle sides are not plotted
#         return [triangles]
#     else:
#         # define the lists Xe, Ye, Ze, of x, y, resp z coordinates of edge end points for each triangle
#         # None separates data corresponding to two consecutive triangles
#         Xe = []
#         Ye = []
#         Ze = []
#         for T in tri_vertices:
#             Xe.extend([T[k % 3][0] for k in range(4)] + [None])
#             Ye.extend([T[k % 3][1] for k in range(4)] + [None])
#             Ze.extend([T[k % 3][2] for k in range(4)] + [None])

#         # define the lines to be plotted
#         lines = go.Scatter3d(x=Xe, y=Ye, z=Ze, mode="lines", line=dict(color="rgb(70,70,70)", width=0.5))
#         return [triangles, lines]


# def plotlySurf(verts, faces):
#     x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
#     triangles, lines = plotly_trisurf(x, y, z, faces, colormap=cm.RdBu, plot_edges=True)
#     return triangles, lines
