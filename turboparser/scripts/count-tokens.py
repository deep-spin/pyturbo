# -*- coding: utf-8 -*-

import argparse
import numpy as np

from turboparser.parser.dependency_reader import read_instances

"""
Script to count treebank tokens and instances
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('treebank')
    args = parser.parse_args()

    instances = read_instances(args.treebank)
    num_instances = len(instances)
    inst_lengths = np.array([len(instance) - 1 for instance in instances])
    num_tokens = inst_lengths.sum()

    sent_length_bins = 10 * np.arange(1, 6)
    bins = np.digitize(inst_lengths, sent_length_bins, right=True)
    counts = np.bincount(bins)
    bin_distribution = 100 * counts / counts.sum()

    print('%d instances' % num_instances)
    print('%d total tokens' % num_tokens)
    print('%.2f tokens per instance' % (num_tokens / num_instances))

    s = ' '.join('%.2f%%' % x for x in bin_distribution)
    print('Length distribution: %s' % s)
    print('Length bins: %s' % sent_length_bins)
