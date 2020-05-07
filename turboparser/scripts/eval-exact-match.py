# -*- coding: utf-8 -*-

"""
Script to evaluate the ratio of complete sentence accuracy and full set of
modifiers accuracy.
"""

from __future__ import division, print_function, unicode_literals

import argparse
import numpy as np

from turboparser.parser.dependency_reader import read_instances

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold', help='Gold annotation file')
    parser.add_argument('pred', help='System output')
    args = parser.parse_args()

    gold_insts = read_instances(args.gold)
    pred_insts = read_instances(args.pred)

    full_sent_unlabeled_matches = 0.
    full_sent_labeled_matches = 0.
    full_head_unlabeld_matches = 0.
    full_head_labeled_matches = 0.
    total_heads = 0

    head_matches = 0
    total_tokens = 0

    for gold_inst, pred_inst in zip(gold_insts, pred_insts):
        assert len(gold_inst) == len(pred_inst)

        gold_heads = np.array(gold_inst.heads[1:])
        pred_heads = np.array(pred_inst.heads[1:])
        head_matches += np.sum(gold_heads == pred_heads)
        total_tokens += len(gold_heads)

        # ignore language-specific deprel subtypes (like in conll eval)
        gold_rels = [rel.split(':')[0] for rel in gold_inst.relations[1:]]
        pred_rels = [rel.split(':')[0] for rel in pred_inst.relations[1:]]

        if np.all(gold_heads == pred_heads):
            full_sent_unlabeled_matches += 1
            if all(g == p for g, p in zip(gold_rels, pred_rels)):
                full_sent_labeled_matches += 1

        # tokens which have at least one modifier, except root
        head_set = set(gold_heads) - {0}
        total_heads += len(head_set)
        for head in head_set:
            # positions that are modifiers of this head
            gold_modifiers = gold_heads == head
            pred_modifiers = pred_heads == head
            if np.all(gold_modifiers == pred_modifiers):
                full_head_unlabeld_matches += 1

                modifier_inds = np.where(gold_modifiers)[0]
                if all(gold_rels[i] == pred_rels[i] for i in modifier_inds):
                    full_head_labeled_matches += 1

    n = len(gold_insts)
    uas = 100 * full_sent_unlabeled_matches / n
    las = 100 * full_sent_labeled_matches / n
    print('Full sentence UAS/LAS: %.2f\t%.2f' % (uas, las))

    acc_u = 100 * full_head_unlabeld_matches / total_heads
    acc_l = 100 * full_head_labeled_matches / total_heads
    print('Full modifier set unlabaled/labeled match: %.2f\t%.2f'
          % (acc_u, acc_l))

    print('UAS: %.2f' % (100 * head_matches / total_tokens))
