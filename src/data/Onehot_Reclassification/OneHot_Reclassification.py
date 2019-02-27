# Restructuring Data to include 5 non overlapping classes

'''
The 5 classes chosen are:
      class      id        previous count
  1) Speech:  /m/09x0r        386025
  2) Bird:    /m/015p6        9592
  3) Engine:  /m/02mk9        6832
  4) Water:   /m/0838f        3208
  5) Siren:   /m/03kmc9       3029
  6) None of above
No Music:     /m/04rlf        387568

SQL Command
SELECT labels.display_name,  labels.mid, COUNT(labels_videos.id)
FROM labels_videos INNER JOIN labels ON labels.id = labels_videos.label_id
WHERE labels.display_name = "Music";

'''
# Youtube_dl
from __future__ import unicode_literals
import youtube_dl
# CSV file reading
import csv
# Audio Processing
import librosa
# VGGISH feature extractor
#import vggish_input
#import vggish_params
#import vggish_postprocess
#import vggish_slim

import numpy as np
import os

# Youtube Downloader Logger and Hook
class MyLogger(object):
    def debug(self, msg):
        pass
    
    def warning(self, msg):
        pass
    
    def error(self, msg):
        print(msg)

def my_hook(d):
    if d['status'] == 'finished':
        print('Done downloading, now converting ...')

# Download audiofile from youtube ID yt_id, Crop from t_start to t_end save original, convert to Fs sampling required for VGGish & convolve with b_n (FIR filter) return original and sampled
def yt_dl(yt_id, t_start, t_end, Fs, b_n):
  ydl_opts = {
        'outtmpl': '/Users/behrad/Google\ Drive/Stanford/CS230\ DeepLearningAI/cs230/src/data/onehot_Reclassification/'+yt_id+'.wav',
        'format': 'bestaudio/best',
        'postprocessors': [{
                           'key': 'FFmpegExtractAudio',
                           'preferredcodec': 'wav',
                           'preferredquality': '192',
                           }],
            'logger': MyLogger(),
        }    

  try:
      with youtube_dl.YoutubeDL(ydl_opts) as ydl:
          ydl.download(['https://www.youtube.com/watch?v='+yt_id])
  except:
      print('ID: ' + yt_id + ' failed to download!')

  # Read .wav
  y, sr = librosa.load(yt_id+'.wav')
  # Crop .wav
  y = y[int(t_start*sr):int(t_end*sr)]
  # Resample .wav
  y_resampled = librosa.resample(y,sr, Fs)
  # filter .wav
  y_filtered = np.convolve(y_resampled, b_n, 'same')

  # Store original and filtered file
  lib.output.write_wav(yt_id+'.wav', y_resampled, Fs)
  lib.output.write_wav(yt_id+';F.wav', y_filtered, Fs)


  return y_resampled, y_filtered




# Number of data points from each class
N_speech = 0
N_s = 0
N_speech_dev = 0
N_speech_test = 0
N_bird = 0
N_engine = 0
N_water = 0
N_siren = 0
N_other = 0

# Train/ Test / Dev
N_train = 5000
N_dev = 250
N_test = 250

# Fitler coefficients
b_n = [-0.0588665108099095,
-0.0701033848219016,  -0.0603921092226670,  -0.0156438329055447,  0.0358758948619559, 0.0545009492086415,
0.0217123281425785,  -0.0424689590048449,  -0.0945963278090251,  -0.102924304731349, -0.0707553131253208,
-0.0272119500341395,  0.00369577907399500,  0.0242395109906721, 0.0543234847106460,
0.103454011687007,  0.154487426983976, 0.176576210574062,
0.154487426983976,  0.103454011687007, 0.0543234847106460,
0.0242395109906721,  0.00369577907399500,  -0.0272119500341395,  -0.0707553131253208,  -0.102924304731349, -0.0945963278090251,
-0.0424689590048449,  0.0217123281425785, 0.0545009492086415,
0.0358758948619559,  -0.0156438329055447,  -0.0603921092226670,  -0.0701033848219016,  -0.0588665108099095]

Fs = 16000


# Go through the "unbalanced_train_segments.csv" and extract 6 non overlapping classes
with open('Speech.csv', 'w') as write1:
  w1 = csv.writer(write1, delimiter=',')
  with open('Bird.csv', 'w') as write2:
    w2 = csv.writer(write2, delimiter=',')
    with open('Engine.csv', 'w') as write3:
      w3 = csv.writer(write3, delimiter=',')
      with open('Water.csv', 'w') as write4:
        w4 = csv.writer(write4, delimiter=',')
        with open('Siren.csv', 'w') as write5:
          w5 = csv.writer(write5, delimiter=',')
          with open('balanced_train_segments.csv') as csvfile:
            readcsv = csv.reader(csvfile, delimiter='"')
            count = 0;
            for row in readcsv:
              count += 1
              if count <= 3:
                continue
              labels = row[1].split(",")
              y_id = row[0].split(",")

              # Speech
              if ('/m/09x0r' in labels and '/m/015p6' not in labels and '/m/02mk9' not in labels and '/m/0838f' not in labels and '/m/03kmc9' not in labels and '/m/04rlf' not in labels and N_speech<(N_train+N_dev+N_test)):
                if(N_speech < N_train):
                  y_resampled, y_filtered = yt_dl(y_id[0], y_id[1], y_id[2], Fs, b_n)
                  print('Shape of original recording: '+ str(y_resampled.shape))
                  print('Shape of filtered recording: '+ str(y_filtered.shape))
                  w1.writerows([row])
                elif(N_speech < N_train + N_dev):
                  N_speech_dev += 1
                elif(N_speech<N_train + N_dev + N_test):
                  N_speech_test += 1
                N_speech +=1
                print(y_id[0])
                print(y_id[1])
                print(y_id[2])

              # Bird
              if ('/m/09x0r' not in labels and '/m/015p6' in labels and '/m/02mk9' not in labels and '/m/0838f' not in labels and '/m/03kmc9' not in labels and '/m/04rlf' not in labels and N_bird<5500):
                N_bird +=1
                w2.writerows([row])

              # Engine
              if ('/m/09x0r' not in labels and '/m/015p6' not in labels and '/m/02mk9' in labels and '/m/0838f' not in labels and '/m/03kmc9' not in labels and '/m/04rlf' not in labels and N_engine<5500):
                N_engine +=1
                w3.writerows([row])

              # Water
              if ('/m/09x0r' not in labels and '/m/015p6' not in labels and '/m/02mk9' not in labels and '/m/0838f' in labels and '/m/03kmc9' not in labels and '/m/04rlf' not in labels and N_water<5500):
                N_water +=1
                w4.writerows([row])

              # Siren
              if ('/m/09x0r' not in labels and '/m/015p6' not in labels and '/m/02mk9' not in labels and '/m/0838f' not in labels and '/m/03kmc9' in labels and '/m/04rlf' not in labels and N_siren<5500):
                N_siren +=1
                w5.writerows([row])


write5.close()
write4.close()
write3.close()
write2.close()
write1.close()

print('Number of unique Speech (Train) clips is: ' + str(N_s))
print('Number of unique Speech (Dev) clips is: ' + str(N_speech_dev))
print('Number of unique Speech (Test) clips is: ' + str(N_speech_test))
print('Number of unique Bird clips is: ' + str(N_bird))
print('Number of unique Engine clips is: ' + str(N_engine))
print('Number of unique Water clips is: ' + str(N_water))
print('Number of unique Siren clips is: ' + str(N_siren))