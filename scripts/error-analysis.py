# -*- coding: utf-8 -*-

import argparse
import numpy as np
from collections import Counter
import matplotlib
matplotlib.use('TKAgg')
from matplotlib import pyplot as pl

from turboparser.parser.dependency_reader import read_instances

"""
Script to analyze errors of the dependency parser by various criteria.
"""


def get_depths(heads):
    """
    Compute the depth of each token. Head indices are 1-based, 0 points to root.
    """
    depths = np.zeros_like(heads)
    found_root = heads == 0
    next_heads = heads
    while not found_root.all():
        depths[~found_root] += 1

        # go to the next depth
        next_heads = heads[next_heads - 1]
        found_root = np.where(found_root, True, next_heads == 0)

    return depths


def compute_scores(gold_instances, pred_instances, sent_length_bins,
                   max_dist=10, max_depth=10,
                   compute_distance=False, compute_depth=False,
                   compute_length=False, compute_pos=False):
    """
    Return a dictionary with arrays for the UAS and LAS per distance and depth.
    """
    assert len(gold_instances) == len(pred_instances)

    # root, dists 1 to (max - 1), and ≥ max
    num_bins_dist = max_dist + 1
    head_hits_per_distance = np.zeros(num_bins_dist, np.int)
    rel_hits_per_distance = np.zeros_like(head_hits_per_distance)

    num_bins_depth = max_depth + 1
    head_hits_per_depth = np.zeros(num_bins_depth, np.int)
    rel_hits_per_depth = np.zeros_like(head_hits_per_depth)

    # +1 because the bin indicators are actually the thresholds
    head_hits_per_sent_size = np.zeros(len(sent_length_bins) + 1, np.int)
    rel_hits_per_sent_size = np.zeros_like(head_hits_per_sent_size)

    head_hits_per_pos = Counter()
    rel_hits_per_pos = Counter()

    head_hits_per_gold_rel = Counter()
    rel_hits_per_gold_rel = Counter()
    head_hits_per_pred_rel = Counter()
    rel_hits_per_pred_rel = Counter()

    tokens_per_distance = np.zeros_like(head_hits_per_distance)
    tokens_per_depth = np.zeros_like(head_hits_per_depth)
    tokens_per_sent_size = np.zeros_like(head_hits_per_sent_size)
    tokens_per_pos = Counter()
    tokens_per_gold_rel = Counter()
    tokens_per_pred_rel = Counter()

    for gold_inst, pred_inst in zip(gold_instances, pred_instances):
        assert len(gold_inst) == len(pred_inst)
        gold_heads = np.array(gold_inst.heads[1:])
        pred_heads = np.array(pred_inst.heads[1:])

        positions = np.arange(1, len(gold_inst))
        distances = np.abs(gold_heads - positions)
        # distance 0 indicates root; others indicate absolute distance to head
        distances = np.where(gold_heads == 0, 0, distances)
        np.clip(distances, 0, max_dist, distances)

        depths = get_depths(gold_heads)
        np.clip(depths, 0, max_depth, depths)

        # ignore language-specific deprel subtypes (like in conll eval)
        gold_rels = [rel.split(':')[0] for rel in gold_inst.relations[1:]]
        pred_rels = [rel.split(':')[0] for rel in pred_inst.relations[1:]]

        head_hits = gold_heads == pred_heads
        rel_hits = np.array(
            [gold_rel == pred_rel
             for gold_rel, pred_rel in zip(gold_rels, pred_rels)])
        rel_hits *= head_hits
        assert isinstance(head_hits, np.ndarray)

        # place the head hits in distance bins
        bins = np.bincount(distances[head_hits], minlength=num_bins_dist)
        head_hits_per_distance += bins

        bins = np.bincount(distances[rel_hits],
                           minlength=num_bins_dist)
        rel_hits_per_distance += bins

        distance_bins = np.bincount(distances, minlength=num_bins_dist)
        tokens_per_distance += distance_bins

        # place hits in the depth bins
        bins = np.bincount(depths[head_hits], minlength=num_bins_depth)
        head_hits_per_depth += bins
        bins = np.bincount(depths[rel_hits],
                           minlength=num_bins_depth)
        rel_hits_per_depth += bins
        depth_bins = np.bincount(depths, minlength=num_bins_depth)
        tokens_per_depth += depth_bins

        # place hits in the sentence size bins
        bin = np.digitize(len(gold_heads), sent_length_bins, right=True)
        head_hits_per_sent_size[bin] += head_hits.sum()
        rel_hits_per_sent_size[bin] += np.sum(rel_hits)
        tokens_per_sent_size[bin] += len(gold_heads)

        # place hits in the pos tag counters
        pos_tags = gold_inst.upos[1:]
        tokens_per_pos.update(pos_tags)
        tokens_per_gold_rel.update(gold_rels)
        tokens_per_pred_rel.update(pred_rels)

        # -1 for root
        for i in range(len(gold_inst) - 1):
            tag = pos_tags[i]
            gold_rel = gold_rels[i]
            pred_rel = pred_rels[i]
            head_hit = head_hits[i]
            rel_hit = rel_hits[i]
            head_hits_per_pos[tag] += head_hit
            rel_hits_per_pos[tag] += rel_hit

            head_hits_per_gold_rel[gold_rel] += head_hit
            rel_hits_per_gold_rel[gold_rel] += rel_hit

            head_hits_per_pred_rel[pred_rel] += head_hit
            rel_hits_per_pred_rel[pred_rel] += rel_hit

    dist_uas = 100 * head_hits_per_distance / tokens_per_distance
    dist_las = 100 * rel_hits_per_distance / tokens_per_distance
    depth_uas = 100 * head_hits_per_depth / tokens_per_depth
    depth_las = 100 * rel_hits_per_depth / tokens_per_depth
    length_uas = 100 * head_hits_per_sent_size / tokens_per_sent_size
    length_las = 100 * rel_hits_per_sent_size / tokens_per_sent_size
    pos_uas = {}
    pos_las = {}
    for tag in tokens_per_pos:
        pos_uas[tag] = 100 * head_hits_per_pos[tag] / tokens_per_pos[tag]
        pos_las[tag] = 100 * rel_hits_per_pos[tag] / tokens_per_pos[tag]

    rel_precision = {}
    rel_recall = {}
    for rel in tokens_per_gold_rel:
        num_pred = tokens_per_pred_rel[rel]
        if num_pred:
            rel_precision[rel] = 100 * rel_hits_per_pred_rel[rel] / num_pred
        else:
            rel_precision[rel] = 100

        rel_recall[rel] = \
            100 * rel_hits_per_gold_rel[rel] / tokens_per_gold_rel[rel]

    data = {'dist_uas': dist_uas,
            'dist_las': dist_las,
            'depth_uas': depth_uas,
            'depth_las': depth_las,
            'sent_length_uas': length_uas,
            'sent_length_las': length_las,
            'pos_uas': pos_uas,
            'pos_las': pos_las,
            'rel_precision': rel_precision,
            'rel_recall': rel_recall}

    return data


def plot(results_per_run, metric, xlabel, ylabel, xticks, output,
         legend_position='lower left'):
    fig, ax = pl.subplots()
    fig.set_size_inches((9.6, 4.8))

    r = np.arange(len(xticks))

    line_styles = ['--', '-', '-.', '--', '-', '-.']
    marker_styles = ['o', 'v', '^', 'x', 'D', 'h']

    for i, name in enumerate(results_per_run):
        data = results_per_run[name]
        results = data[metric]
        line = line_styles[i]
        marker = marker_styles[i]
        ax.plot(r, results, label=name, linestyle=line, marker=marker)

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    ax.legend(loc=legend_position)
    pl.xticks(r, xticks)
    pl.grid(True, 'both', linestyle='--')
    pl.savefig(output, dpi=200, bbox_inches='tight')


def barplot(results_per_run, metric, xlabel, ylabel, output, labels, bottom):
    """
    Bar plot for accuracy in categories, such as POS tags
    """
    fig, ax = pl.subplots()
    fig.set_size_inches((6, 10))
    y = np.arange(len(labels))
    height = 0.1
    half_height = height / 2
    num_bars = len(results_per_run)
    current_y = y - (num_bars - 1) * half_height

    for name in results_per_run:
        data = results_per_run[name]
        results = data[metric]
        ordered_results = np.array([results[label] for label in labels])
        ax.barh(current_y, ordered_results - bottom,
                height, label=name, left=bottom, zorder=3)
        current_y += height

    pl.xlabel(xlabel)
    pl.ylabel(ylabel)
    ax.set_yticklabels(labels)
    ax.set_yticks(y)
    ax.legend(loc='upper right')
    pl.grid(True, 'both', axis='x', linestyle=':', zorder=0)
    pl.savefig(output, dpi=200, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('gold', help='Gold annotation file')
    parser.add_argument('pred', help='Predicted annotations from one or more '
                                     'systems', nargs='+')
    parser.add_argument('output', help='Base name for output png files')
    parser.add_argument('--names', help='Names for the plots', nargs='+')
    # parser.add_argument('--dist', help='Plot accuracy by h/m distance',
    #                     action='store_true')
    # parser.add_argument('--depth', help='Plot accuracy by m depth',
    #                     action='store_true')
    # parser.add_argument('--length', help='Plot accuracy by sentence length',
    #                     action='store_true')
    # parser.add_argument('--pos', help='Plot accuracy by m UPOS',
    #                     action='store_true')
    args = parser.parse_args()

    gold_instances = read_instances(args.gold)
    results_per_run = {}
    sent_length_bins = 10 * np.arange(1, 6)

    bottom_pos = 100
    bottom_rel_recall = 100
    bottom_rel_precision = 100
    for name, pred_file in zip(args.names, args.pred):
        print('Reading file %s' % pred_file)
        pred_instances = read_instances(pred_file)
        results = compute_scores(gold_instances, pred_instances,
                                 sent_length_bins=sent_length_bins)
        results_per_run[name] = results

        bottom_pos = min(bottom_pos, 0.96 * min(results['pos_las'].values()))
        bottom_rel_precision = min(
            bottom_rel_precision, 0.96 * min(results['rel_precision'].values()))
        bottom_rel_recall = min(
            bottom_rel_recall, 0.96 * min(results['rel_recall'].values()))

    xticks = ['root'] + list(range(1, 10)) + ['10+']
    output = args.output + '-dist.png'
    plot(results_per_run, 'dist_las', 'Dependency length', 'LAS', xticks,
         output)

    output = args.output + '-depth.png'
    plot(results_per_run, 'depth_las', 'Distance to root', 'LAS', xticks,
         output, 'upper right')

    xticks = ['1-10', '11-20', '21-30', '31-40', '41-50', '51+']
    output = args.output + '-sent-length.png'
    plot(results_per_run, 'sent_length_las', 'Sentence length', 'LAS', xticks,
         output)

    output = args.output + '-pos.png'
    tag_names = list(results_per_run.values())[0]['pos_las'].keys()
    barplot(results_per_run, 'pos_las', 'LAS', '', output,
            sorted(tag_names), bottom_pos)

    # output = args.output + '-rel-precision.png'
    # rel_names = list(results_per_run.values())[0]['rel_precision'].keys()
    # rel_names = sorted(rel_names)
    # barplot(results_per_run, 'rel_precision', 'Precision', '', output,
    #         rel_names, bottom_rel_precision)
    #
    # output = args.output + '-rel-recall.png'
    # barplot(results_per_run, 'rel_recall', 'Recall', '', output,
    #         rel_names, bottom_rel_recall)
