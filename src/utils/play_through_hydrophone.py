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

    Paramters
    ---------

    raw_file: str
        raw_file is a string that contains the file name of a .wav file that was
        downloaded from s3 and needs to be played through the hydrophone.

        For example, it's currently hardcoded such that
        raw_file = 'working_video.wav'

    wav_file: str
        wav_file is a string that contains the file name that the outside script that is
        calling this program is expecting to be written to. When this program
        is done, it will upload 'wav_file' to s3. So make sure that you have
        something that makes sense stored in hydrophone.wav by the time
        you exit this script

    npy_file: str
        npy_file is a string ('numpy.npy') that has the path to the numpy file
        that will be uploaded to amazon s3. Make sure that you run something like
        np.save(numpy_file, arr) by the time this exits.

    Returns
    -------

    None. There is nothing to return.

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
