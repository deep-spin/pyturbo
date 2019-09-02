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
    parser.add_argument('-y', help='Vertical axis limits to show (2 values)',
                        nargs=2, default=[0, 1], type=float)
    parser.add_argument('--names', nargs='+',
                        help='Names of the plots. If not given, filenames will '
                             'be used')
    args = parser.parse_args()

    metric = 'UAS'
    names = args.logs if args.names is None else args.names

    all_mins = []
    all_maxs = []
    fig, ax = pl.subplots()

    for log, name in zip(args.logs, names):
        accuracies = extract_accuracies(log, metric)
        steps = np.arange(1, len(accuracies) + 1)
        ax.plot(steps, accuracies, label=name)

        all_maxs.append(accuracies.max())
        all_mins.append(accuracies.min())

    y_min = max(args.y[0], 0.96 * min(all_mins))
    y_max = min(args.y[1], 1.04 * max(all_maxs))

    ax.set_yticks(all_maxs, minor=True)
    ax.legend()
    pl.xlabel('Evaluation step')
    pl.ylabel(metric)
    pl.ylim(y_min, y_max)
    pl.grid(True, 'both', linestyle='--')
    pl.savefig(args.output, dpi=200)
