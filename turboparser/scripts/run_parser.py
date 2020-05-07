# -*- coding: utf-8 -*-

"""
Call and run the Turbo Parser on a given text.

It also serves as an example of API usage.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import sys

from turboparser import TurboParser, read_instances, DependencyInstance, \
    DependencyWriter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Path to the parser model')
    parser.add_argument('--pruner', help='Path to the pruner model, if used')
    parser.add_argument(
        '-f', '--file', help='Input conllu file. If not given, read from stdin')
    parser.add_argument(
        '-o', '--output', help='Output conllu file. If not given, write to '
                               'stdout')
    args = parser.parse_args()

    dep_parser = TurboParser.load(path=args.model, pruner_path=args.pruner)

    if args.file:
        # if the input is a conllu file, read from it
        instances = read_instances(args.file)
    else:
        # if not, read tokens from stdin and assume they are tokenized
        # by whitespace
        instances = []
        for line in sys.stdin.readlines():
            tokens = line.strip().split()
            instance = DependencyInstance.from_tokens(tokens)
            instances.append(instance)

    # get a list of dictionary mapping targets (tags and parse) to the outputs
    predictions = dep_parser.parse(instances)

    if args.output:
        # write to an output file
        writer = DependencyWriter()
        writer.open(args.output)
        for instance in instances:
            writer.write(instance)
        writer.close()
    else:
        # print each annotated instance to stdout
        for instance in instances:
            print(instance.to_conll())
            print()


if __name__ == '__main__':
    main()
