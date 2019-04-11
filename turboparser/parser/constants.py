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

    NEXT_SIBLINGS = auto()
    GRANDPARENTS = auto()
    GRANDSIBLINGS = auto()
