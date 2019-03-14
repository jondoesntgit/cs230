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

load_dotenv()
CHECKPOINT = os.getenv('VGGISH_MODEL_CHECKPOINT')
CHECKPOINT = str(Path(CHECKPOINT).expanduser())
PCA_PARAMS = os.getenv('EMBEDDING_PCA_PARAMETERS')
PCA_PARAMS = str(Path(PCA_PARAMS).expanduser())
os.environ['KMP_DUPLICATE_LIB_OK']='True' # Hacky way to suppress a warning
CLASSIFIER_PATH = 'audioset_multilabel_M18.h5'

def get_mel_spectrogram(y, sr):
    arr = librosa.feature.melspectrogram(y=y, sr=sr)
    plt.imshow(arr)
    plt.ylabel('Frequency')
    plt.xlabel('Time')
    plt.savefig('processed/mel.png', dpi=300, bbox_inches='tight')
    return 'processed/mel.png'

def get_vggish_features(y, sr):
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
        plt.savefig('processed/vggish.png', dpi=300, bbox_inches='tight')
        return 'processed/vggish.png', postprocessed_batch

class MainHandler(tornado.web.RequestHandler):
    def get(self):
        #self.write("Hello, world2")
        items = ['Item1', 'item2', 'item3']
        self.render('static/template.html', title='My Title', items=items)

class AudioUploadHandler(tornado.web.RequestHandler):
    def post(self):
        print('Handling post')
        #print('self', self)
        #print('request', self.request)
        #print('body', self.request.body)
        #print('request', self.request)
        #print('files', self.request.files)
        #print(type(self.request.files['audio'][0]))
        file = self.request.files['audio'][0]
        p = Path('uploads/tmp.wav')
        with p.open('wb') as f:
            f.write(file['body'])
        self.write(json.dumps({'localFile': 'tmp.wav'}))

        #r = json.dumps({'status': 'okay'})
        #self.write(r)

class EchoWebSocket(tornado.websocket.WebSocketHandler):
    def open(self):
        print('WebSocket opened')

    def on_message(self, message):
        print(f'Processing {message}')
        p = Path('uploads') / message
        #self.write_message(u'You said: ' + message)

        y, sr = librosa.load(str(p))
        mel_path = get_mel_spectrogram(y, sr)
        self.write_message(json.dumps({'mel_path':  mel_path}))

        vggish_path, vggish_features = get_vggish_features(y, sr)
        self.write_message(json.dumps({'vggish_path': vggish_path}))

        vggish_features = vggish_features[np.newaxis,:,:,np.newaxis]
        #print(vggish_features.shape)
                #%% load classifier model
        classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
        prediction = classifier.predict(vggish_features)[0]
        self.write_message(json.dumps({'labels': prediction.tolist()}))

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
    app = make_app()
    app.listen(5000)
    tornado.autoreload.start()
    for dir, _, files in os.walk('static'):
        [tornado.autoreload.watch(dir + '/' + f) for f in files if not f.startswith('.')]
    tornado.autoreload.watch('app.py')
    tornado.ioloop.IOLoop.current().start()
