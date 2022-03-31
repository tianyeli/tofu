'''
tensorboard logger

view in tensorboard
1. call the following in command line
$ tensorboard --logdir=<your_log_dir> --port=<your_port_number, e.g. 6006>

2. open link in browsers: localhost:6006

Please see LICENSE for the licensing information
'''
import time
import os
from collections import OrderedDict
from utils.utils import AverageMeter, AdvancedMeter, safe_mkdir
try:
    from StringIO import StringIO  # Python 2.7
except ImportError:
    from io import BytesIO         # Python 3.x

# -----------------------------------------------------------------------------

class Logger(object):

    def __init__(self, log_dir):
        import tensorflow as tf
        self.tf = tf
        self.log_dir = log_dir
        safe_mkdir(self.log_dir)

        self.tf_version_major = float(tf.__version__.split('.')[0])
        if self.tf_version_major < 2:
            self.writer = tf.summary.FileWriter(self.log_dir)
        else:
            # new syntax in tensorflow 2.0
            # https://github.com/eriklindernoren/PyTorch-YOLOv3/issues/327
            self.writer = tf.summary.create_file_writer(self.log_dir)

    def add_numerical(self, data, step, name):
        if self.tf_version_major < 2:
            summary = self.tf.Summary(value=[self.tf.Summary.Value(tag=name, simple_value=data)])
            self.writer.add_summary(summary, step)
        else:
            with self.writer.as_default():
                self.tf.summary.scalar(name, data, step=step)
                # self.writer.flush()

    def add_visual(self, img_np, step, name):
        raise NotImplementedError

    def update_numericals(self, numericals, step, mode='train'):
        # note: tesnorboard shows losses vs step (iteration) number
        for tag, value in numericals.items():
            self.add_numerical(data=value, step=step, name=mode+"_"+tag)

    def update_visuals(self, visuals, step, mode='train'):

        if self.tf_version_major < 2:
            img_summaries = []
            for label, image_numpy in visuals.items():

                # write the image to a string
                try:
                    s = StringIO()
                except:
                    s = BytesIO()

                if isinstance(image_numpy, list):
                    for i in range(len(image_numpy)):
                        scipy.misc.toimage(image_numpy[i]).save(s, format="jpeg")
                        img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy[i].shape[0], width=image_numpy[i].shape[1])
                        img_summaries.append(self.tf.Summary.Value(tag=mode+"_"+label, image=img_sum))
                else:
                    scipy.misc.toimage(image_numpy).save(s, format="jpeg")
                    img_sum = self.tf.Summary.Image(encoded_image_string=s.getvalue(), height=image_numpy.shape[0], width=image_numpy.shape[1])
                    img_summaries.append(self.tf.Summary.Value(tag=mode+"_"+label, image=img_sum))

            # create and write Summary
            summary = self.tf.Summary(value=img_summaries)
            self.writer.add_summary(summary, step)

        else: # tf ver >= 2
            # images are ranged in [0, 1] in float32, in shape (h, w, c)
            import numpy as np
            for label, image_numpy in visuals.items():
                with self.writer.as_default():
                    if isinstance(image_numpy, list):
                        # assuming all images in the list are in same shape
                        imgs = np.concatenate([ im[np.newaxis, :] for im in image_numpy ], axis=0)
                        self.tf.summary.image(mode+"_"+label, imgs, step=step)
                    else:
                        self.tf.summary.image(mode+"_"+label, image_numpy[np.newaxis, :, :, :], step=step)
