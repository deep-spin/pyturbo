class DependencyFeatures(object):
    def __init__(self, classifier, parts):
        self.classifier = classifier
        self.input_features = [None for part in parts]

    def __getitem__(self, r):
        return self.input_features[r]

    def add_arc_features_light(self, instance, r, head, modifier):
        self.input_features[r] = []

    def add_arc_features(self, instance, r, head, modifier):
        return self.add_arc_features_light(instance, r, head, modifier)

