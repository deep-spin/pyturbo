from classifier.structured_decoder import StructuredDecoder
from parser.dependency_instance import DependencyInstance

class DependencyDecoder(StructuredDecoder):
    def __init__(self):
        StructuredDecoder.__init__(self)
