import random


class Instance(object):
    '''An abstract instance.'''
    def __init__(self, input_, output=None):
        self.input = input_
        self.output = output


class InstanceData(object):
    """
    Class for storing a list of instances, their corresponding parts and gold
    labels.
    """
    def __init__(self, instances, parts, gold_labels=None):
        """
        :param instances: a list of instances
        :type instances: list
        :param parts: a list of the parts of each instance
        :type parts: list
        :param gold_labels: list of dictionaries. Each dictionary maps the name
            of a target (such as upos) to a gold numpy array. If there are no
            labels to classify besides the parts, this should be None.
        :type gold_labels: list[dict]
        """
        self.instances = instances
        self.parts = parts
        self.gold_labels = gold_labels

    def __getitem__(self, item):
        labels = None if self.gold_labels is None else self.gold_labels[item]
        return InstanceData(self.instances[item], self.parts[item], labels)

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
        if self.gold_labels is not None:
            self.gold_labels = list(next(it))

    def shuffle(self):
        """
        Shuffle the data stored by this object.
        """
        zipped = self._zip_data()
        random.shuffle(zipped)
        self._unzip_data(zipped)
