# Extracting Data from ESC-50 Dataset
# Behrad Habib Afshar
# 5-Feb-2019

"""
Usage

python esc50.py --log=DEBUG

Extracts features from each of the datasets and puts them into a folder defined by ESC50_SPLITS in the .env file

From here, the files can be loaded using

.. code:: python

	import numpy as np
	train_data = np.load('train_data.npy')
	train_labels = np.load('train_labels.npy')
   
"""

import numpy as np
import librosa as lib
import os
import pickle
from pathlib import Path
from dotenv import load_dotenv
import logging
import getopt
import sys
import tqdm

loglevel = 'WARNING'
options, remainder = getopt.gnu_getopt(sys.argv[1:], 'l', ['log='])
for opt, arg in options:
	if opt in ('-l', '--log'):
		loglevel = arg

numeric_level = getattr(logging, loglevel.upper(), None)
if not isinstance(numeric_level, int):
	raise ValueError('Invalid log level: %s' % loglevel)
logging.basicConfig(level=numeric_level)

load_dotenv()
ESC50_RAW_DATA = Path(os.getenv('ESC50_REPO')).expanduser() / 'audio'
ESC50_SPLITS_PATH = Path(os.getenv('ESC50_SPLITS')).expanduser()

logging.debug('Creating directories.')
train_path = ESC50_SPLITS_PATH / 'train'
dev_path = ESC50_SPLITS_PATH / 'dev'
test_path = ESC50_SPLITS_PATH / 'test'

for p in (train_path, dev_path, test_path):
	p.mkdir(parents=True, exist_ok=True)
	assert p.exists()

# Spectrogram settings
n_fft = 1024
hop_length = 512
n_mels = 128
srate = 44100

# Number of data in train/dev/test sets
count_train = 0
count_dev = 0
count_test = 0

# data & label in train/dev/test sets
train_data = np.empty((n_mels, 431))
train_label = []
dev_data = np.empty((n_mels, 431))
dev_label = []
test_data = np.empty((n_mels, 431))
test_label = []

# Go through all files in the folder
for file_name in tqdm.tqdm(os.listdir(ESC50_RAW_DATA)):

	try:
		y, sr = lib.load(ESC50_RAW_DATA / file_name, sr=None)
		#logging.info('Processing %s' % file_name)

		if len(y.shape) > 1:
		#	logging.info('Mono Conversion') 
			y = lib.to_mono(y)

		if sr != srate:
		#	logging.info('Resampling to '+str(srate))
			y = lib.resample(y,sr,srate)

		mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
	
		#Filename decomposed: {Fold} - {ID} - {Take} - {Class}
		fold, fid, ftake, fclass = file_name.split('-') 
		fold = int(fold)
		fclass = int(fclass.split('.')[0])

		# Divide the data into three
		if fold < 4:
			train_data = np.dstack((train_data, mel_feat))
			train_label.append(fclass)
			count_train += 1
		elif fold == 4:
			dev_data = np.dstack((dev_data, mel_feat))
			dev_label.append(fclass)
			count_dev += 1
		else:
			test_data = np.dstack((test_data, mel_feat))
			test_label.append(fclass)
			count_test += 1
	except IOError: # Give me an audio file which I can read!!
		logging.error(file_name, "did not get coverted!")

# Remove the first element randomly initialized by np.empty
train_data = np.delete(train_data, 0, 2)
dev_data = np.delete(dev_data, 0, 2)
test_data = np.delete(test_data, 0, 2)

# Pickle data into appropriate folders
np.save(train_path/'train_data', train_data)
np.save(train_path/'train_label', train_label)
np.save(dev_path/'dev_data', dev_data)
np.save(dev_path/'dev_label', dev_label)
np.save(test_path/'test_data', test_data)
np.save(test_path/'test_label', test_label)

logging.info('train_data shape'+str(train_data.shape)+'   and train label shape' + str(len(train_label)))
logging.info('dev_data shape'+str(dev_data.shape)+'   and dev label shape' + str(len(dev_label)))
logging.info('test_data shape'+str(test_data.shape)+'   and test label shape' + str(len(test_label)))

# Output some stats
logging.info('# of training data: '+str(count_train) + ' which is: ' + str(count_train/(count_train+count_test+count_dev))+' of all the data')
logging.info('# of dev data: '+str(count_dev) + ' which is: ' + str(count_dev/(count_train+count_test+count_dev))+' of all the data')
logging.info('# of test data: '+str(count_test) + ' which is: ' + str(count_test/(count_train+count_test+count_dev))+' of all the data')
