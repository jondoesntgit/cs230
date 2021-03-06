import tornado.ioloop
import tornado.web
import tornado.autoreload
import tornado.websocket
import os
import json
from pathlib import Path
import matplotlib.pyplot as plt
import librosa
import tensorflow as tf
from dotenv import load_dotenv
import os
import librosa as lib
import numpy as np
import sys
sys.path.append('../src/')
from features.vggish_input import waveform_to_examples as w2e
from features import vggish_slim, vggish_postprocess, vggish_params
from optparse import OptionParser
import logging

logging.basicConfig(level=logging.DEBUG)

load_dotenv()
CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
CHECKPOINT = str(Path(CHECKPOINT).expanduser())
PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Hacky way to suppress a warning
CLASSIFIER_PATH = 'audioset_multilabel_M23.h5'

def get_mel_spectrogram(y, sr, name):
    arr = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.imshow(np.log(arr))
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    sub = Path(name).stem
    fname = 'processed/%s_mel.png' % sub
    plt.savefig(fname, dpi=300, bbox_inches='tight')
    return fname

def get_vggish_features(y, sr, name):
    with tf.Graph().as_default(), tf.Session() as sess:
    # Prepare a postprocessor to munge the model embeddings.
        examples_batch = w2e(y, sr)
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
        postprocessed_batch = pproc.postprocess(embedding_batch)

        # Numpy array of shape (`t`, 128)
        plt.imshow(postprocessed_batch)
        plt.xlabel('Feature')
        plt.ylabel('Time')
        fname = 'processed/%s_vggish.png' % Path(name).stem
        plt.savefig(fname, dpi=300, bbox_inches='tight')
        return fname, postprocessed_batch

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('static/template.html', title='Deep Audio Classifier')

class AudioUploadHandler(tornado.web.RequestHandler):
    def post(self):
        print('Handling post')
        #print('self', self)
        #print('request', self.request)
        #print('body', self.request.body)
        #print('request', self.request)
        #print('files', self.request.files)
        #print(self.request.files.keys())
        #print(type(self.request.files['audio'][0])))
        file = self.request.files['audio'][0]
        p = Path('uploads/%s' % file['filename'])
        with p.open('wb') as f:
            f.write(file['body'])
        self.write(file['filename'])

class EchoWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        logging.debug('WebSocket opened')

    def on_message(self, message):
        logging.debug(f'Processing {message}')
        p = Path('uploads') / message
        #self.write_message(u'You said: ' + message)

        y, sr = librosa.load(str(p))
        mel_path = get_mel_spectrogram(y, sr, message)
        logging.debug('Writing mel_path')
        self.write_message(json.dumps({'mel_path':  mel_path}))

        vggish_path, vggish_features = get_vggish_features(y, sr, message)
        logging.debug('Writing vggish_path')
        self.write_message(json.dumps({'vggish_path': vggish_path}))

        vggish_features = vggish_features[np.newaxis,:,:,np.newaxis]

        #%% load classifier model
        classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
        prediction = classifier.predict(vggish_features)[0]
        logging.debug('Writing labels')
        self.write_message(json.dumps({'labels': prediction.tolist()}))
        logging.debug(f'Finished processing {message}')

    def on_close(self):
        print('Webscoetk closed')

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
        (r'/upload_audio', AudioUploadHandler),
        (r'/websocket', EchoWebSocket),
        (r'/static/(.*)', tornado.web.StaticFileHandler, {'path': 'static/'}),
        (r'/processed/(.*)', tornado.web.StaticFileHandler, {'path': 'processed/'})
    ], gzpi=True)

if __name__ == "__main__":

    parser = OptionParser()
    parser.add_option('-p', '--port', dest='port', default=80, type=int)
    parser.add_option('--host', dest='hostname', default='localhost')
    options, args = parser.parse_args()

    port = options.port
    hostname = options.hostname

    app = make_app()
    app.listen(port)
    tornado.autoreload.start()
    for dir, _, files in os.walk('static'):
        [tornado.autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    tornado.autoreload.watch('app.py')
    tornado.ioloop.IOLoop.current().start()
