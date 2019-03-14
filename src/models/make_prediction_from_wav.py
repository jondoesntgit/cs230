#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:33:28 2019

@author: jwitmer
"""
"

import tensorflow as tf
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

import sys
sys.path.append('..')
from features.vggish_input import waveform_to_examples as w2e
from features import vggish_slim, vggish_postprocess, vggish_params
from pathlib import Path
from dotenv import load_dotenv
import os
import librosa as lib


if __name__ == '__main__':

    #%% get VGG model paths
    load_dotenv()
    CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
    CHECKPOINT = str(Path(CHECKPOINT).expanduser())
    PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
    PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())
    CLASSIFIER_PATH = 'audioset_multilabel_M18.h5'
    
    #%% load classifier model
    classifier = tf.keras.model.load(CLASSIFIER_PATH)
    
    #%% get data paths and load .wav data
    # This appears to be a fix for a common problem on macs...
    os.environ['KMP_DUPLICATE_LIB_OK']='True' # Hacky way to suppress a warning
    print(sys.argv)
    p = Path('~/Downloads/example.wav').expanduser()
    if len(sys.argv) >= 2:
        p = Path(sys.argv[1])
    
    print(str(p))
    
    y, sr = lib.load(str(p))
    
    #%% feed .wav to VGGish and classifier
    examples_batch = w2e(y, sr)

    postprocessed_batch = generate_VGGish_features(examples_batch, CHECKPOINT, PCA_PARAMS)

    prediction = classifier.predict(postprocessed_batch)
        
    #%% print predicted classes
    class_IDs = ['Male speech', 'Female speech', 'Bird', 'Water', 'Engine', 'Siren']

    predicted_class_inds = np.nonzero(prediction > 0.5)[0]
    predicted_class_IDs = class_IDs[predicted_class_inds]
    
    print('==============================================')
    print('PREDICTED CLASSES:')
    print('----------------------------------------------')
    print(predicted_class_IDs)
    print('==============================================')
    


#%%
def generate_VGGish_features(examples_batch, CHECKPOINT, PCA_PARAMS):
    
    with tf.Graph().as_default(), tf.Session() as sess:
    
        # Prepare a postprocessor to munge the model embeddings.
        pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)
        print(examples_batch)
        print(examples_batch.shape)
    
        # Define the model in inference mode, load the checkpoint, and
        # locate input and output tensors.
    
        vggish_slim.define_vggish_slim(training=False)
        vggish_slim.load_vggish_slim_checkpoint(sess, CHECKPOINT)
        features_tensor = sess.graph.get_tensor_by_name(
            vggish_params.INPUT_TENSOR_NAME)
        embedding_tensor = sess.graph.get_tensor_by_name(
            vggish_params.OUTPUT_TENSOR_NAME)
    
        # Run inference and postprocessing.
        [embedding_batch] = sess.run([embedding_tensor],
                                     feed_dict={features_tensor: examples_batch})
        postprocessed_batch = pproc.postprocess(embedding_batch)
        
        return postprocessed_batch


    
    
    
    
    






