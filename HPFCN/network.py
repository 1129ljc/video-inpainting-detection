import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

g = tf.Graph()
with g.as_default() as g:
    tf.train.import_meta_graph('/ssd4/ljc/Deep_inpainting_localization/DVI_model/model.ckpt-0.642142-3.meta')

with tf.Session(graph=g) as sess:
    file_writer = tf.summary.FileWriter(logdir='./network', graph=g)
