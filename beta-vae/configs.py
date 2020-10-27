"""
This configuration file stores common variables for use across the program.
It uses a simple file exists check to determine whether or not it is running remotely or locally
and changes the filepaths accordingly for convenience.
"""

#%% Imports
import os


LATENT_DIM = 64
IMG_SIZE = 192  # 224, 256 (128)
KL_WEIGHT = 20.0  # rough guess for "optimally disentangled"

# CURR_IMGRUN_ID = '1014-0001'
CURR_IMGRUN_ID = None  # train from scratch
# CURR_TXTRUN_ID = '0922-1614'
CURR_TXTRUN_ID = None  # train from scratch
N_IMGRUN_EPOCH = 400
N_TXTRUN_EPOCH = 3000

BATCH_SIZE = 32
IMGRUN_LR = 4e-4
TXTRUN_LR = 1e-4

VALIDATION_FRAC = 20.0 / 100.0  # standard 80/20 test/validate split


#%% Properties
# This folder path should only exist on the remote machine
REMOTE = os.path.isdir("/data/sn/all")

# HOME = "/Users/ergonyc"  #osx
# HOME = "/home/ergonyc"    #linux
HOME = os.environ.get("HOME")


if REMOTE:  # Running on EC2 instance or similar
    META_DATA_CSV = "/data/sn/all/meta/dfmeta.csv"
    VOXEL_FILEPATH = "/data/sn/all/all/"
    IMGRUN_DIR = "/data/sn/all/runs/"
    TXTRUN_DIR = "/data/sn/all/txtruns/"
    DATA_DIR = "/data/sn/all/data/"

else:  # Running locally
    META_DATA_CSV = HOME + "/Projects/DATABASE/SnkrScrpr/data/basic_data_raw.csv"

    """
    This is the directory file that stores all of the metadata information gathered and analyzed by the program to generate descriptions.
    It is made by the program and shouldn't require additional action but is a useful resource to inspect manually    
    """

    # ROOT_FILEPATH = HOME + "/Projects/Project2.0/SnkrScrpr/data/"
    # FILEPATH_GOAT = HOME + "/Projects/Project2.0/SnkrScrpr/data/goat/img/"
    # FILEPATH_SNS = HOME + "/Projects/Project2.0/SnkrScrpr/data/sns/img/"

    ROOT_FILEPATH = HOME + "/Projects/DATABASE/SnkrScrpr/data/"
    FILEPATH_GOAT = HOME + "/Projects/DATABASE/SnkrScrpr/data/goat/img/"
    FILEPATH_SNS = HOME + "/Projects/DATABASE/SnkrScrpr/data/sns/img/"

    IMAGE_FILEPATH = ROOT_FILEPATH
    """
    This is the location of the image data scraped from GOAT and SNS. 
    """

    IMGRUN_DIR = HOME + "/Projects/Project2.0/SnkrGen/beta-vae/imgruns/"
    TXTRUN_DIR = HOME + "/Projects/Project2.0/SnkrGen/beta-vae/txtruns/"
    """
    These are the run log and model checkpoint folders. This folder structure is generated and managed by the logger.py class.

    Example run directory tree structure:
        TODO
    RUN_DIR
    ├── 0217-0434
    │   ├── code_file.txt
    │   ├── configs.csv
    │   ├── logs
    │   │   └── events.out.tfevents.1581942983.ip-172-31-21-198
    │   ├── models
    │   │   ├── checkpoint
    │   │   ├── ckpt-161.data-00000-of-00002
    │   │   ├── ckpt-161.data-00001-of-00002
    │   │   ├── ckpt-161.index
    │   │   └── epoch_161.h5
    │   └── plots
    ├── 0217-0437
    │   ├── code_file.txt
    │   ├── configs.csv
    │   ├── logs
    │   │   └── events.out.tfevents.1581943124.ip-172-31-24-21
    │   ├── models
    │   │   ├── checkpoint
    │   │   ├── ckpt-258.data-00000-of-00002
    │   │   ├── ckpt-258.data-00001-of-00002
    │   │   ├── ckpt-258.index
    │   │   └── epoch_258.h5
    │   ├── saved_data

    │   └── plots
        ...
    """

    DATA_DIR = HOME + "/Projects/Project2.0/SnkrGen/beta-vae/data/"

    """
    This folder is used to cache various computation and memory intensive generated files like the randomized descriptions of objects.
    """

    RENDERS_DIR = HOME + "/Projects/Project2.0/SnkrGen/beta-vae/renders/"
    """
    This folder is used to store rendered images of the models for quick and easy viewing and for use in the streamlit app.
    Primarily used when inspecting the quality of generated descriptions.
    """

    PARTNET_META_STATS_DIR = HOME + "/Projects/Project2.0/SnkrGen/beta-vae/stats/"
    """
    This folder contains all of the metadata that is used to generate the descriptions.
    TODO
    Specifically, only the meta.json and result_after_merging.json files are necessary.
    """


# %%
