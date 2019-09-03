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
    BEST_RELATION = auto()
    SIGN = auto()
    DISTANCE = auto()

    NEXT_SIBLINGS = auto()
    GRANDPARENTS = auto()
    GRANDSIBLINGS = auto()


higher_order_parts = {Target.NEXT_SIBLINGS, Target.GRANDPARENTS,
                      Target.GRANDSIBLINGS}
dependency_targets = {Target.HEADS, Target.RELATIONS, Target.DISTANCE,
                      Target.SIGN}
dependency_targets.update(higher_order_parts)

target2string = {Target.DEPENDENCY_PARTS: 'Dependency parts',
                 Target.XPOS: 'XPOS', Target.UPOS: 'UPOS',
                 Target.MORPH: 'UFeats', Target.SIGN: 'Linearization',
                 Target.DISTANCE: 'Head distance',
                 Target.GRANDSIBLINGS: 'Grandsiblings',
                 Target.GRANDPARENTS: 'Grandparent',
                 Target.NEXT_SIBLINGS: 'Consecutive siblings'}


ROOT = '_root_'
UNKNOWN = '_unknown_'
PADDING = '_padding_'
NONE = '_none_'
SPECIAL_SYMBOLS = [PADDING, UNKNOWN, NONE, ROOT]
