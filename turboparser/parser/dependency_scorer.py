from ..classifier.neural_scorer import NeuralScorer


class DependencyNeuralScorer(NeuralScorer):
    """
    Subclass of neural scorer that can compute loss on both dependency parsing
    and POS tagging.
    """
