"""
This class is used to automatically create and organize training info.

For each run, it creates a new folder structure like this example:
    
0131-0411/
├── code_file.txt
├── configs.csv
├── logs
│   └── events.out.tfevents.1580472882.ip-172-31-26-36
├── models
│   ├── checkpoint
│   ├── ckpt-1.data-00000-of-00002
│   ├── ckpt-1.data-00001-of-00002
│   ├── ckpt-1.index
│   ├── ckpt-2.data-00000-of-00002
│   ├── ckpt-2.data-00001-of-00002
│   └── ckpt-2.index
└── plots
    ├── Original.png
    ├── Reconstruct 10.png
    └── Reconstruct 15.png
    
The top level folder is named DDDD-TTTT where DDDD is the date with month followed by date and TTT is the military time the folder was created.
The code_file.txt contains copies of the actual code from any input functions / classes.
The configs.csv takes in any local variables with the prefix cf_ and puts them into a csv.
The logs file is a tensorboard log file created with easy_tf2_log.py
The models folder contains all of the saved models from training.
The plots folder contains any output plots during training such as shape reconstructions.    
"""

#HACK... remove easy_tf2_log... its not working.... putting tf2 loggin into the logger.

#%% Imports
import os
import inspect
import csv

import utils as ut
import tensorflow as tf
import configs as cf

#%% Logger class
class logger: 
    def __init__(self, run_name="", root_dir=None, trainMode=False, txtMode=False):
        self.remote = cf.REMOTE
        self.training = trainMode or self.remote
        self.total_epochs = 1
        self.run_name = run_name

        self.new_run = root_dir is None
        if self.new_run:
            self.root_dir = cf.IMG_RUN_DIR
            self.root_dir = os.path.join(self.root_dir, ut.add_time_stamp())
        else:
            self.root_dir = root_dir

        if txtMode:
            if self.root_dir.split("/")[-2] == "txtruns":
                print(f"root_dir already txtruns= {self.root_dir}")
            else:
                self.root_dir = self.root_dir.replace("imgruns", "txtruns")
                print(f"changed to text runs:root_dir= {self.root_dir}")

        self.img_in_dir = cf.IMAGE_FILEPATH
        self.plot_out_dir = os.path.join(self.root_dir, "plots/")
        self.model_saves = os.path.join(self.root_dir, "models/")
        self.tblogs = os.path.join(self.root_dir, "logs")
        self.test_writer = None
        self.train_writer = None

        # for writing test and validate pkl record of dataset
        self.saved_data = os.path.join( self.root_dir, "saved_data")  
        self.updatePlotDir()
        

    def check_make_dirs(self):
        def make_dir(dirname):
            if os.path.isdir(dirname):
                return
            else:
                os.mkdir(dirname)

        make_dir(self.root_dir)
        make_dir(ut.plot_out_dir)
        make_dir(self.model_saves)
        make_dir(self.tblogs)
        make_dir(self.saved_data)
        #print(f"made {self.saved_data}")
        self.setupWriter()


    def setup_writer(self):
        ##  need to make tensorflow logs
        if self.test_writer is None:
            train_log_dir = self.tblogs + '/train'
            test_log_dir = self.tblogs + '/test'
            train_summary_writer = tf.summary.create_file_writer(train_log_dir)
            test_summary_writer = tf.summary.create_file_writer(test_log_dir)
            #etl.set_dir(self.tblogs)
            self.test_writer = test_summary_writer
            self.train_writer = test_summary_writer


    def reset(self, total_epochs=1):
        self.total_epochs = total_epochs

    def log_metric(self, metric, name, test=False):
        self.check_make_dirs()
        if test:
            summary_writer = self.test_writer
        else:
            summary_writer = self.train_writer

        with summary_writer.as_default():
            tf.summary.scalar(name=name,data=metric, step=self.total_epochs)
    
            # tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
        # tb_callback.set_model(model)
    def increment_epoch(self):
        self.total_epochs = self.total_epochs + 1

    def setup_checkpoint(self, generator, encoder, opt):
        if encoder is None:
            self.checkpoint = tf.train.Checkpoint(optimizer=opt, generator=generator)
        else:
            self.checkpoint = tf.train.Checkpoint(optimizer=opt, generator=generator, encoder=encoder)

        self.cpmanager = tf.train.CheckpointManager(
            self.checkpoint, directory=self.model_saves, max_to_keep=(3 if (self.remote) else 20)
        )
        # AH add for better tensorboard??
        # tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
        # tb_callback.set_model(model)        # tb_callback = tf.keras.callbacks.TensorBoard(LOG_DIR)
        # tb_callback.set_model(model)

    def save_checkpoint(self):
        self.cpmanager.save()

    def restore_checkpoint(self, path=None):
        if not os.path.isdir(self.root_dir):
            print(f"folder ({self.root_dir}) not found, not restored.")
            return
        if path == None:
            status = self.checkpoint.restore(self.cpmanager.latest_checkpoint)
            print("(NONE)Latest model chkp path is : {} setupWriter...".format(self.cpmanager.latest_checkpoint))
            #etl.set_dir(self.tblogs)
            self.setupWriter()
            return status
        else:
            status = self.checkpoint.restore(path)
            print("Latest model chkp path is : {}  setupWriter".format(status))
            #etl.set_dir(self.tblogs)
            self.setupWriter()
            return status

    def write_config(self, variables, code):
        self.check_make_dirs()
        if not self.training:
            print("Cannot, in read only mode.")
            return

        if len(code) > 0:
            code_file = open(os.path.join(self.root_dir, "code_file.txt"), "w")
            for source in code:
                code_file.write(repr(source) + "\n\n")
                code_file.write(inspect.getsource(source) + "\n\n")
            code_file.close()

        filtered_vars = {key: value for (key, value) in variables.items() if (key.startswith("cf_"))}
        w = csv.writer(open(os.path.join(self.root_dir, "configs.csv"), "w"))
        for key, val in filtered_vars.items():
            w.writerow([key, val])

    def updatePlotDir(self):
        ut.plot_out_dir = self.plot_out_dir
