import random


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
    def __init__(self, instances, parts, features=None, gold_parts=None,
                 gold_labels=None):
        """
        :param instances: a list of instances
        :type instances: list
        :param parts: a list of the parts of each instance
        :type parts: list
        :param gold_parts: list of numpy arrays with gold labels (1 and 0)
            for the parts of each sentence. It is a list, not a matrix, since
            each instance has a different number of parts.
        :type gold_parts: list
        :param gold_labels: list of dictionaries. Each dictionary maps the name
            of a target (such as upos) to a gold numpy array. If there are no
            labels to classify besides the parts, this should be None.
        :type gold_labels: list[dict]
        """
        self.instances = instances
        self.parts = parts
        self.features = features
        self.gold_parts = gold_parts
        self.gold_labels = gold_labels

    def __getitem__(self, item):
        features = None if self.features is None else self.features[item]
        gold_parts = None if self.gold_parts is None else self.gold_parts[item]
        labels = None if self.gold_labels is None else self.gold_labels[item]
        return InstanceData(self.instances[item], self.parts[item],
                            features, gold_parts, labels)

    def __len__(self):
        return len(self.instances)

    def sort_by_size(self, descending=False):
        """
        Sort the instances in-place from longest to shortest (or the opposite if
        descending is True).
        """
        zipped = self._zip_data()
        sorted_data = sorted(zipped, key=lambda x: len(x[0]),
                             reverse=descending)
        self._unzip_data(sorted_data)

    def _zip_data(self):
        """Auxiliary internal function"""
        # zip the attributes together so they are shuffled in the same order
        data = [self.instances, self.parts]
        if self.features is not None:
            data.append(self.features)
        if self.gold_parts is not None:
            data.append(self.gold_parts)
        if self.gold_labels is not None:
            data.append(self.gold_labels)

        zipped = list(zip(*data))
        return zipped

    def _unzip_data(self, zipped_data):
        """Auxiliary internal function"""
        unzipped_data = zip(*zipped_data)
        it = iter(unzipped_data)

        self.instances = list(next(it))
        self.parts = list(next(it))
        if self.features is not None:
            self.features = list(next(it))
        if self.gold_parts is not None:
            self.gold_parts = list(next(it))
        if self.gold_labels is not None:
            self.gold_labels = list(next(it))

    def shuffle(self):
        """
        Shuffle the data stored by this object.
        """
        zipped = self._zip_data()
        random.shuffle(zipped)
        self._unzip_data(zipped)
