# -*- coding: utf-8 -*-

"""
Call and run the Turbo Parser on a given text.

It also serves as an example of API usage.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import sys

from turboparser import TurboParser

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Path to the parser model')
    parser.add_argument('--pruner', help='Path to the pruner model, if used')
    args = parser.parse_args()

    dep_parser = TurboParser.load(path=args.model, pruner_path=args.pruner)
    lines = [line.strip().split() for line in sys.stdin.readlines()]
    predictions = dep_parser.run_on_tokens(lines)

    for prediction in predictions:
        print(prediction)
