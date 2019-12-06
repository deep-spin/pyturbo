# -*- coding: utf-8 -*-

import argparse
import numpy as np

from turboparser.parser.dependency_reader import read_instances
from turboparser.parser.dependency_writer import DependencyWriter

"""
Find which sentences were correctly parsed by one model but not by the other. 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold', help='Gold treebank file')
    parser.add_argument('system1', help='Output of first system')
    parser.add_argument('system2', help='Output of second system')
    parser.add_argument('output1',
                        help='Output file for sentences only 1 got right')
    parser.add_argument('output2',
                        help='Output file for sentences only 2 got right')
    args = parser.parse_args()

    gold_instances = read_instances(args.gold)
    pred_instances1 = read_instances(args.system1)
    pred_instances2 = read_instances(args.system2)

    only_1_right = []
    only_2_right = []

    for gold, pred1, pred2 in zip(gold_instances,
                                  pred_instances1, pred_instances2):
        gold_heads = np.array(gold.heads[1:])
        pred_heads1 = np.array(pred1.heads[1:])
        pred_heads2 = np.array(pred2.heads[1:])

        if np.all(pred_heads1 == pred_heads2):
            # same answers for this instance
            continue

        system1_right = np.all(pred_heads1 == gold_heads)
        system2_right = np.all(pred_heads2 == gold_heads)

        if system1_right:
            only_1_right.append(gold)
        elif system2_right:
            only_2_right.append(gold)

    writer = DependencyWriter()
    writer.open(args.output1)
    for inst in only_1_right:
        writer.write(inst)
    writer.close()
    writer.open(args.output2)
    for inst in only_2_right:
        writer.write(inst)
    writer.close()
