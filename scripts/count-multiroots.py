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

    no_root = 0
    one_root = 0
    more_roots = 0

    with DependencyReader(args.corpus) as reader:
        for instance in reader:
            output = instance.output
            num_roots = output.heads.count(0)

            if num_roots == 0:
                no_root += 1
            elif num_roots == 1:
                one_root += 1
            else:
                more_roots += 1

    total = no_root + one_root + more_roots
    prop_no_root = 100 * no_root / total
    prop_one_root = 100 * one_root / total
    prop_more_roots = 100 * more_roots / total

    print('%d sentences in the treebank' % total)
    print('%d (%.2f%%) sentences without root' % (no_root, prop_no_root))
    print('%d (%.2f%%) sentences with one root' % (one_root, prop_one_root))
    print('%d (%.2f%%) sentences with more than one root'
          % (more_roots, prop_more_roots))
