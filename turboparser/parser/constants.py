# -*- coding: utf-8 -*-

from enum import Enum, auto


class Target(Enum):
    """Enum class for targets in the parser/tagger"""
    DEPENDENCY_PARTS = auto()
    HEADS = auto()
    RELATIONS = auto()
    XPOS = auto()
    UPOS = auto()
    MORPH = auto()
    LEMMA = auto()
    BEST_RELATION = auto()
    SIGN = auto()
    DISTANCE = auto()

    NEXT_SIBLINGS = auto()
    GRANDPARENTS = auto()
    GRANDSIBLINGS = auto()


class ParsingObjective(Enum):
    LOCAL = auto()
    GLOBAL_MARGIN = auto()
    GLOBAL_PROBABILITY = auto()


higher_order_parts = {Target.NEXT_SIBLINGS, Target.GRANDPARENTS,
                      Target.GRANDSIBLINGS}
dependency_targets = {Target.HEADS, Target.RELATIONS, Target.DISTANCE,
                      Target.SIGN}
dependency_targets.update(higher_order_parts)
structured_objectives = {ParsingObjective.GLOBAL_MARGIN,
                         ParsingObjective.GLOBAL_PROBABILITY}

target2string = {Target.DEPENDENCY_PARTS: 'Dependency parts',
                 Target.XPOS: 'XPOS', Target.UPOS: 'UPOS',
                 Target.MORPH: 'UFeats', Target.LEMMA: 'Lemma',
                 Target.SIGN: 'Linearization',
                 Target.DISTANCE: 'Head distance',
                 Target.GRANDSIBLINGS: 'Grandsiblings',
                 Target.GRANDPARENTS: 'Grandparent',
                 Target.NEXT_SIBLINGS: 'Consecutive siblings',
                 Target.RELATIONS: 'LAS',
                 Target.HEADS: 'UAS'}

string2objective = {'local': ParsingObjective.LOCAL,
                    'global-margin': ParsingObjective.GLOBAL_MARGIN,
                    'global-prob': ParsingObjective.GLOBAL_PROBABILITY}

ROOT = '_root_'
UNKNOWN = '_unknown_'
PADDING = '_padding_'
EMPTY = '_empty_'
SPECIAL_SYMBOLS = [PADDING, UNKNOWN, EMPTY, ROOT]

BOS = '_bos_'
EOS = '_eos_'

bert_model_name = 'bert-base-multilingual-cased'
