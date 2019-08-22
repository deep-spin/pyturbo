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


dependency_targets = {Target.HEADS, Target.RELATIONS, Target.DISTANCE,
                      Target.SIGN}
target2string = {Target.DEPENDENCY_PARTS: 'Dependency parts',
                 Target.XPOS: 'XPOS', Target.UPOS: 'UPOS',
                 Target.MORPH: 'UFeats', Target.SIGN: 'Linearization',
                 Target.DISTANCE: 'Head distance'}


ROOT = '_root_'
UNKNOWN = '_unknown_'
PADDING = '_padding_'
NONE = '_none_'
SPECIAL_SYMBOLS = [PADDING, UNKNOWN, NONE, ROOT]
