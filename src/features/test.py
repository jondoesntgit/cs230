#!/usr/bin/env python
"""
This is a short little executable script that takes a .wav file stored in
~/Downloads/example.wav, and dumps out the numpy representation of the vggish
features.

The script is not yet complete.
"""

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

load_dotenv()
CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
CHECKPOINT = str(Path(CHECKPOINT).expanduser())
PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())

# This appears to be a fix for a common problem on macs...
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Hacky way to suppress a warning

print(sys.argv)
p = Path('~/Downloads/example.wav').expanduser()
if len(sys.argv) >= 2:
    p = Path(sys.argv[1])

print(str(p))
y, sr = lib.load(str(p))
examples_batch = w2e(y, sr)

#print(examples_batch)

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

    # Numpy array of shape (`t`, 128)
    print(postprocessed_batch)
    print(postprocessed_batch.shape)
    plt.imshow(postprocessed_batch)
    plt.show()
