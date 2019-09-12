# -*- coding: utf-8 -*-

import argparse
import re
import numpy as np

"""
Analyze how much time batches took in log files.
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='Log file')
    args = parser.parse_args()

    with open(args.log, 'r') as f:
        text = f.read()
    lines = text.splitlines()

    match = re.search('log_interval=(\d+)', text)
    batches_per_report = int(match.group(1))

    pattern = 'Time to score: ([\d.]+)\s+Decode: ([\d.]+)\s+' \
              'Gradient step: ([\d.]+)'
    times = [re.search(pattern, line).groups()
             for line in lines
             if 'Time to score:' in line]

    times = np.array([[float(t1), float(t2), float(t3)]
                      for (t1, t2, t3) in times])
    time_scoring = times[:, 0]
    time_decoding = times[:, 1]
    time_gradient = times[:, 2]
    num_reports = time_scoring.shape[0]
    num_batches = num_reports * batches_per_report

    total_scoring = time_scoring.sum()
    total_decoding = time_decoding.sum()
    total_gradient = time_gradient.sum()

    print('%d batches run in total' % num_batches)
    print('Total time spent in scoring: %.2fs' % total_scoring)
    print('Total time spent in decoding: %.2fs' % total_decoding)
    print('Total time spent in weight gradients: %.2fs' % total_gradient)
    print('Total time: %.2fs\n' % times.sum())

    mean_scoring = total_scoring / num_batches
    mean_decoding = total_decoding / num_batches
    mean_gradient = total_gradient / num_batches

    print('Mean scoring time per batch: %.2fs' % mean_scoring)
    print('Mean decoding time per batch: %.2fs' % mean_decoding)
    print('Mean gradient time per batch: %.2fs' % mean_gradient)
    print('Mean time per batch: %.2fs' % (times.sum() / num_batches))
