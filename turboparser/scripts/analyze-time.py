# -*- coding: utf-8 -*-

import argparse
import re
import numpy as np

"""
Analyze how much time batches took in log files.
"""


def format_time(num_seconds):
    s = ''
    if num_seconds > 60:
        num_minutes = num_seconds // 60
        num_seconds = num_seconds % 60
        if num_minutes > 60:
            num_hours = num_minutes // 60
            num_minutes = num_minutes % 60
            s += '%dh ' % num_hours
        s += '%dm ' % num_minutes
    s += '%.2fs' % num_seconds

    return s


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('log', help='Log file')
    parser.add_argument('--mean', action='store_true', help='Show mean times')
    parser.add_argument('--detailed', action='store_true',
                        help='Show time for scoring/gradient/decoding')
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
    total_scoring_str = format_time(total_scoring)
    total_decoding_str = format_time(total_decoding)
    total_gradient_str = format_time(total_gradient)

    print('%d batches run in total' % num_batches)
    if args.detailed:
        print('Total time spent in scoring: %s' % total_scoring_str)
        print('Total time spent in decoding: %s' % total_decoding_str)
        print('Total time spent in weight gradients: %s' % total_gradient_str)
    print('Total time: %s\n' % format_time(times.sum()))

    mean_scoring = format_time(total_scoring / num_batches)
    mean_decoding = format_time(total_decoding / num_batches)
    mean_gradient = format_time(total_gradient / num_batches)
    mean_total = format_time(times.sum() / num_batches)

    if args.mean:
        if args.detailed:
            print('Mean scoring time per batch: %s' % mean_scoring)
            print('Mean decoding time per batch: %s' % mean_decoding)
            print('Mean gradient time per batch: %s' % mean_gradient)
        print('Mean time per batch: %s' % mean_total)
