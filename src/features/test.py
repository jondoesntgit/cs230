#!/usr/bin/env python


import sys
sys.path.append('..')
from features.vggish_input import wavfile_to_examples as w2e
from pathlib import Path
print('loaded')

p = Path('~/Downloads/example.wav').expanduser()

print(str(p))
print(w2e(str(p)))
