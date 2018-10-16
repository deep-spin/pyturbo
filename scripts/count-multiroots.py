# -*- coding: utf-8 -*-

"""
Script to count trees with more than one root in a treebank.
"""

import argparse

from turboparser.parser import DependencyReader


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('corpus', help='Corpus to count')
    args = parser.parse_args()

    reader = DependencyReader(args.corpus)
    with reader.open():
        instance = reader.next()
        print(instance)
