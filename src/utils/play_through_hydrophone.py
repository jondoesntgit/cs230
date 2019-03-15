import shutil
from pathlib import Path
from optparse import OptionParser
import numpy as np

def main(raw_file, wav_file, numpy_file):
    """
    TODO: 

    Play the "raw file" audio through a speaker, and record it into a numpy
    file that should be saved, and then converted into a wav file using something
    like librosa.
    """
    print(raw_file)
    print(wav_file)
    shutil.copyfile(raw_file, wav_file)

    arr = np.array([1,2,3])
    np.save(numpy_file, arr)

if __name__ == '__main__':

    parser = OptionParser()

    parser.add_option('-f', '--from', dest='raw_file', type='string', help='The wav file to read from')
    parser.add_option('-t', '--to', dest='wav_file', type='string', help='The wav file to write to')
    parser.add_option('-n', '--numpy', dest='numpy_file', type='string', help='The wav file to write to')
    options ,args = parser.parse_args()
    main(raw_file=options.raw_file, 
        numpy_file=options.numpy_file,
        wav_file=options.wav_file)
