# -*- coding: utf-8 -*-

import argparse

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
    num_tokens = sum(len(instance) - 1 for instance in instances)

    print('%d instances' % num_instances)
    print('%d total tokens' % num_tokens)
    print('%.2f tokens per instance' % (num_tokens / num_instances))
