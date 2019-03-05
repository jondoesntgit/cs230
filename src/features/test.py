#!/usr/bin/env python

import tensorflow as tf
import numpy as np
from scipy.io import wavfile

import sys
sys.path.append('..')
from features.vggish_input import wavfile_to_examples as w2e
from features import vggish_slim, vggish_postprocess, vggish_params
from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()
CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
CHECKPOINT = str(Path(CHECKPOINT).expanduser())
PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())


p = Path('~/Downloads/example.wav').expanduser()

print(str(p))
examples_batch = w2e(str(p))

print(examples_batch)

with tf.Graph().as_default(), tf.Session() as sess:

    # Prepare a postprocessor to munge the model embeddings.
    pproc = vggish_postprocess.Postprocessor(PCA_PARAMS)

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
    print(embedding_batch)
    postprocessed_batch = pproc.postprocess(embedding_batch)
    print(postprocessed_batch)
