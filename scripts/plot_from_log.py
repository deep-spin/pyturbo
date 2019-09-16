# -*- coding: utf-8 -*-

"""
Script to generate plots from Turbo Parser logs
"""

from __future__ import division, print_function, unicode_literals

import argparse
import re
import numpy as np
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as pl


def extract_accuracies(filename, uas_or_las='UAS'):
    accuracies = []
    uas_or_las = uas_or_las.upper()
    pattern = r'%s: (\d\.\d+)' % uas_or_las

    with open(filename, 'r') as f:
        for line in f:
            if 'Validation accuracies' not in line:
                continue

            match = re.search(pattern, line)
            acc = match.group(1)
            accuracies.append(float(acc))

    return np.array(accuracies)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('logs', help='Logs produced by the turbo parser',
                        nargs='+')
    parser.add_argument('output', help='File to print log')
    parser.add_argument('--title', help='Title of the plot', default='')
    parser.add_argument('-y', help='Vertical axis limits to show (2 values)',
                        nargs=2, default=[0, 1], type=float)
    parser.add_argument('-m', help='Metric', choices=['UAS', 'LAS'],
                        default='UAS', dest='metric')
    parser.add_argument('--names', nargs='+',
                        help='Names of the plots. If not given, filenames will '
                             'be used')
    args = parser.parse_args()

    names = args.logs if args.names is None else args.names

    all_mins = []
    all_maxs = []
    fig, ax = pl.subplots()
    fig.set_size_inches((9.6, 4.8))

    for log, name in zip(args.logs, names):
        accuracies = extract_accuracies(log, args.metric)
        length = len(accuracies)
        steps = np.arange(1, length + 1)
        ax.plot(steps, accuracies, label=name)

        # by default, set the lower y margins 2% below 92% of the accuracies
        # and the upper margin 2% above the highest
        index = int(0.08 * length)
        all_maxs.append(accuracies.max())
        all_mins.append(accuracies[index:].min())

    y_min = max(args.y[0], 0.98 * min(all_mins))
    y_max = min(args.y[1], 1.02 * max(all_maxs))

    ax.set_yticks(all_maxs, minor=True)
    ax.legend(loc='lower right')
    pl.title(args.title)
    pl.xlabel('Evaluation step')
    pl.ylabel(args.metric)
    pl.ylim(y_min, y_max)
    pl.grid(True, 'both', linestyle='--')
    pl.savefig(args.output, dpi=200, bbox_inches='tight')
