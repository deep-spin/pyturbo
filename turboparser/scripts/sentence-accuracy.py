# -*- coding: utf-8 -*-

"""
Script to compute UAS and LAS by sentence
"""

from __future__ import division, print_function, unicode_literals

import argparse
import os
import numpy as np

from turboparser.parser.dependency_reader import read_instances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold')
    parser.add_argument('pred')
    parser.add_argument(
        '-o', help='Output directory', default='.', dest='out_dir')
    args = parser.parse_args()

    basename = os.path.splitext(os.path.basename(args.pred))[0]
    output_uas = os.path.join(args.out_dir, basename + '-uas.txt')
    output_las = os.path.join(args.out_dir, basename + '-las.txt')

    gold_instances = read_instances(args.gold)
    pred_instances = read_instances(args.pred)
    all_head_hits = []
    all_rel_hits = []

    sentence_uas = []
    sentence_las = []

    for gold_inst, pred_inst in zip(gold_instances, pred_instances):
        gold_heads = np.array(gold_inst.heads[1:])
        pred_heads = np.array(pred_inst.heads[1:])

        # ignore language-specific deprel subtypes (like in conll eval)
        gold_rels = [rel.split(':')[0] for rel in gold_inst.relations[1:]]
        pred_rels = [rel.split(':')[0] for rel in pred_inst.relations[1:]]

        head_hits = gold_heads == pred_heads
        rel_hits = np.array(
            [gold_rel == pred_rel
             for gold_rel, pred_rel in zip(gold_rels, pred_rels)])
        rel_hits *= head_hits

        sentence_uas.append(head_hits.mean())
        sentence_las.append(rel_hits.mean())

        all_head_hits.append(head_hits)
        all_rel_hits.append(rel_hits)

    # np.savetxt(output_uas, sentence_uas)
    # np.savetxt(output_las, sentence_las)
    np.savetxt(output_uas, np.concatenate(all_head_hits))
    np.savetxt(output_las, np.concatenate(all_rel_hits))
