# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 14:04:02 2021

@author: deep
"""
from __future__ import print_function
import tensorflow.compat.v1 as tf
import numpy as np
import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim




def vggish_embedding(filepath):
    """function to get 128 dimension vggish embedding directly from raw audio input"""
    checkpoint_path = 'vggish_model.ckpt'
    pca_params_path = 'vggish_pca_params.npz'
    input_batch = vggish_input.wavfile_to_examples(filepath)
    with tf.Graph().as_default(), tf.Session() as sess:
        vggish_slim.define_vggish_slim()
        vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
        features_tensor = sess.graph.get_tensor_by_name(
      vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
      vggish_params.OUTPUT_TENSOR_NAME)
        [embedding_batch] = sess.run([embedding_tensor],
                               feed_dict={features_tensor: input_batch})
    return embedding_batch


def get_vggish_feature(filepaths):
    """function that will return embeddings from all the input in one list"""
    temp = []
    for ret in filepaths:
        temp.append(vggish_embedding(ret))
    return temp
        
    

class vggish_stat():
    """taking standard deviation and mean as final features from the entire segment"""
    def __init__(self,data):
        self.data = data
    def mean(self):
        MEAN = []
        for embedding in self.data:
            mean = []
            samples = 128
            for idx in range(samples):
                temp = np.mean(embedding[:,idx])
                mean.append(temp)
            MEAN.append(mean)
        return MEAN
    def std(self):
        STD = []
        for embedding in self.data:
            std = []
            samples = 128
            for idx in range(samples):
                temp = np.std(embedding[:,idx])
                std.append(temp)
            STD.append(std)
        return STD

