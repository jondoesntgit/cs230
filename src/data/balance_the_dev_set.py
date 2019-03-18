#!/usr/bin/env python
"""
A script that balances the dev set.
"""

import numpy as np
import h5py
from optparse import OptionParser
from pathlib import Path
import logging
logging.basicConfig(level=logging.INFO)

def main(input_file, output_file, labels, ratio, seed):
    h5file = h5py.File(input_file, mode='r')
    y_slice = h5file['y'][:, labels]
    
    num_hots = np.sum(y_slice, axis=1)
    positive_examples = np.argwhere(num_hots >= 1)[:,0]
    negative_examples = np.argwhere(num_hots == 0)[:,0]
    assert len(num_hots) == len(positive_examples) + len(negative_examples)

    np.random.seed(seed)
    print(len(positive_examples))
    print(len(negative_examples))

    if len(positive_examples) < ratio * len(negative_examples):
        positive_sample = positive_examples
        print('sample negative')
        size = int(len(positive_sample) / ratio)
        print(size)
        negative_sample = np.random.choice(negative_examples, size=size)
        
    else:
        negative_sample = negative_examples
        size = int(len(negative_sample) * ratio)
        positive_sample = np.random.choice(positive_examples, size=size)

    samples = np.concatenate((negative_sample, positive_sample))
    np.random.shuffle(samples) # Does the operation in-place
    
    outfile = h5py.File(output_file, mode='w')
    X = (h5file['X'][:])[samples, :, :, np.newaxis]
    outfile['X'] = X
    outfile['slugs'] = (h5file['slugs'][:])[samples]
    outfile['y'] = y_slice[samples, :]

    logging.info(f'Wrote {len(positive_sample)} positive examples and {len(negative_sample)} negative examples to {output_file}')


if __name__ == '__main__':
    parser = OptionParser()
    parser.add_option('-f', '--file', dest='input_file', help='Input file to write to')
    parser.add_option('-o', '--output', dest='output_file', help='Output file to write to')
    parser.add_option('-l', '--labels', dest='labels', type='string', help='List of the labels to extract')
    parser.add_option('-r', '--ratio', dest='ratio', default=1, type='float', help='Ratio of positive exampels to negative examples')
    parser.add_option('-s', '--seed', dest='seed', default=42, type='float' ,help='random number generator seed')

    options, args = parser.parse_args()

    try:
        input_file = options.input_file
        input_file = Path(input_file).expanduser()
    except TypeError:
        raise ValueError(f'Could not parse the input file')
    output_file = options.output_file
    labels = np.fromstring(options.labels, dtype=int, count=-1, sep=',')


    output_file = Path(output_file).expanduser()

    if not input_file.exists():
        raise ValueError(f'{input_file} does not exist')
    
    if len(labels) == 0:
        raise ValueError(f'Could not find any labels: {labels} ')

    ratio = options.ratio
    seed = options.seed

    main(input_file, output_file, labels, ratio, seed)