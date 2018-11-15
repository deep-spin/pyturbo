class Instance(object):
    '''An abstract instance.'''
    def __init__(self, input, output=None):
        self.input = input
        self.output = output


class InstanceData(object):
    """
    Class for storing a list of instances, their corresponding parts, features
    and gold labels.
    """
    def __init__(self, instances, parts, features=None, gold_labels=None):
        self.instances = instances
        self.parts = parts
        self.features = features
        self.gold_labels = gold_labels

    def __getitem__(self, item):
        features = None if self.features is None else self.features[item]
        labels = None if self.gold_labels is None else self.gold_labels[item]
        return InstanceData(self.instances[item], self.parts[item],
                            features, labels)

    def __len__(self):
        return len(self.instances)
