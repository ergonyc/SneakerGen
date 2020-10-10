"""
This file was originally from here: https://github.com/mrahtz/easy-tf-log
But I updated it to work with TF 2.0 so I could use it in my project.
The only changes are adding two 'tf.compat.v1.' prefixes to the Summary method.
"""

import os
import os.path as osp
import time
from typing import Union

import numpy as np
import tensorflow as tf
# from tensorflow.core.util import event_pb2
# from tensorflow.python import pywrap_tensorflow
# from tensorflow.python.util import compat

class EventsFileWriterWrapper:
    """
    Rename EventsFileWriter's flush() and add_event() methods to be consistent
    with EventsWriter's methods.
    """
    def __init__(self, events_file_writer):
        self.writer = events_file_writer

    def WriteEvent(self, event):
        self.writer.flush()

class Logger(object):
    DEFAULT = None

    def __init__(self, log_dir=None, writer=None, tb_callback = None):
        self.key_steps = {}
        self.rate_values = {}
        #self.writer = None  # type: Union[None, pywrap_tensorflow.EventsWriter, tf.summary.FileWriter]

        # self.tb_callback = None  # hacking in a callback...
        if log_dir is None and writer is None:
            log_dir = "logs"
            self.set_log_dir(log_dir)
        elif log_dir is not None and writer is None:
            self.set_log_dir(log_dir)
        elif log_dir is None and writer is not None:
            self.set_writer(writer)
        else:
            raise ValueError("Only one of log_dir or writer must be specified")
        self.log_dir = log_dir
        
        # hacking in a callback...
        if tb_callbak is None:
            tb_callback = tf.keras.callbacks.TensorBoard(log_dir)
        else:
            self.tb_callback = tb_callback  
        

    def set_log_dir(self, log_dir):
        os.makedirs(log_dir, exist_ok=True)
        path = osp.join(log_dir, "events")
        # Why don't we just use an EventsFileWriter?
        # By default, we want to be fork-safe - we want to wopiprk even if we
        # create the writer in one process and try to use it in a forked
        # process. And because EventsFileWriter uses a subthread to do the
        # actual writing, EventsFileWriter /isn't/ fork-safe.
        #self.writer = pywrap_tensorflow.EventsWriter(compat.as_bytes(path))
        self.writer = tf.summary.create_file_writer(path)  # do i need "as_default()"
        #tf.summary.trace_export(name="all", step=step, profiler_outdir=self.log_dir)



    def set_writer(self, writer):
        """
        Set the log writer to an existing tf.summary.FileWriter instance
        (so that you can write both TensorFlow summaries and easy_tf_log events to the same log file)
        """
        # self.writer = EventsFileWriterWrapper(writer)
        self.writer = tf.summary.create_file_writer(writer) # do i need "as_default()"
        #tf.summary.trace_export(name="all", step=step, profiler_outdir=self.log_dir)


    def logkv(self, k, v, step=None):
        self.log_key_value(k, v, step)

    def log_key_value(self, key, value, step=None):
        # def summary_val(k, v):
        #     kwargs = {"tag": k, "simple_value": float(v)}
        #     return tf.compat.v1.Summary.Value(**kwargs)

        # summary = tf.compat.v1.Summary(value=[summary_val(key, value)])

        # event = event_pb2.Event(wall_time=time.time(), summary=summary)
        # # Use a separate step counter for each key
        # if key not in self.key_steps:e
        #     self.key_steps[key] = 0
        # if step is not None:
        #     self.key_steps[key] = step
        # event.step = self.key_steps[key]
        # self.writer.WriteEvent(event)
        # self.writer.Flush()
        tf.summary.scalar(key, value, step=step)
        self.writer.flush()
        
    def log_list_stats(self, key, values_list):
        for suffix, f in [("min", np.min), ("max", np.max), ("avg", np.mean), ("std", np.std)]:
            self.logkv(key + "_" + suffix, f(values_list))

    def measure_rate(self, key, value):
        if key in self.rate_values:
            last_val, last_time = self.rate_values[key]
            interval = time.time() - last_time
            self.logkv(key + "_rate", (value - last_val) / interval)
        self.rate_values[key] = (value, time.time())

    def close(self):
        if self.writer:
            #self.writer.Close()
            self.writer.close()  #prot to tf2
            self.writer = None


def set_dir(log_dir):
    Logger.DEFAULT = Logger(log_dir=log_dir)


def set_writer(writer):
    Logger.DEFAULT = Logger(writer=writer)


def tflog(key, value, step=None):
    if not Logger.DEFAULT:
        set_dir("logs")
    Logger.DEFAULT.log_key_value(key, value, step)
