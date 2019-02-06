# Extracting Data from ESC-50 Dataset
# Behrad Habib Afshar
# 5-Feb-2019

import numpy as np
import librosa as lib
import os
import pickle

# Audio clips stored in  local storage
esc50_folder = '/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio'
# Make the Train, Dev and Test directories
try:
	os.mkdir("/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio/Train")
except FileExistsError:
	print('Directory exists!')

try:
	os.mkdir("/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio/Dev")
except FileExistsError:
	print('Directory exists!')

try:
	os.mkdir("/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio/Test")
except FileExistsError:
	print('Directory exists!')


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
for file_name in os.listdir(esc50_folder):

	try:
		y, sr = lib.load(esc50_folder+'/'+file_name,sr=None)

		if len(y.shape) > 1:
			print('Mono Conversion') 
			y = lib.to_mono(y)

		if sr != srate:
			print('Resampling to '+str(srate))
			y = lib.resample(y,sr,srate)

		mel_feat = lib.feature.melspectrogram(y=y,sr=srate,n_fft=n_fft,hop_length=hop_length,n_mels=128)
	
		#Filename decomposed: {Fold} - {ID} - {Take} - {Class}
		f_name = file_name.split('-')

		# Divide the data into three
		if int(f_name[0]) < 4:
			train_data = np.dstack((train_data, mel_feat))
			train_label.append(int(f_name[3].split('.')[0]))
			count_train += 1
		elif int(f_name[0]) == 4:
			dev_data = np.dstack((dev_data, mel_feat))
			dev_label.append(int(f_name[3].split('.')[0]))
			count_dev += 1
		else:
			test_data = np.dstack((test_data, mel_feat))
			test_label.append(int(f_name[3].split('.')[0]))
			count_test += 1
	except:
		#raise IOError('Give me an audio  file which I can read!!')
		print(file_name+" did not get coverted!")
    
	

	

# Remove the first element randomly initialized by np.empty
train_data = np.delete(train_data, 0, 2)
dev_data = np.delete(dev_data, 0, 2)
test_data = np.delete(test_data, 0, 2)

# Pickle data into appropriate folders
with open('/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio/Processed_Data/train_data_label', 'wb') as f:
	pickle.dump(train_data, f)
	pickle.dump(train_label,f)
f.close()

with open('/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/audio/Processed_Data/dev_data_label', 'wb') as f:
	pickle.dump(dev_data, f)
	pickle.dump(dev_label,f)
f.close()

with open('/Users/behrad/Documents/Stanford/cs230/raw_data/ESC-50-master/Processed_Data/test_data_label', 'wb') as f:
	pickle.dump(test_data, f)
	pickle.dump(test_label,f)
f.close()

print('train_data shape'+str(train_data.shape)+'   and train label shape' + str(len(train_label)))
print('dev_data shape'+str(dev_data.shape)+'   and dev label shape' + str(len(dev_label)))
print('test_data shape'+str(test_data.shape)+'   and test label shape' + str(len(test_label)))

# Output some stats
print('# of training data: '+str(count_train) + ' which is: ' + str(count_train/(count_train+count_test+count_dev))+' of all the data')
print('# of dev data: '+str(count_dev) + ' which is: ' + str(count_dev/(count_train+count_test+count_dev))+' of all the data')
print('# of test data: '+str(count_test) + ' which is: ' + str(count_test/(count_train+count_test+count_dev))+' of all the data')


