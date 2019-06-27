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

        self.batches = None
        self.next_batch_pointer = 0

    def __getitem__(self, item):
        labels = None if self.gold_labels is None else self.gold_labels[item]
        return InstanceData(self.instances[item], self.parts[item], labels)

    def __len__(self):
        return len(self.instances)

    def prepare_batches(self, words_per_batch, sort=False):
        """
        Split the data into a sequence of batches such that each one has at most
        `words_per_batch` words.

        This is done preemptively to avoid repetition. After calling this
        function, next_batch() becomes usable.

        :param words_per_batch: maximum words per batch (summing all sentences)
        :param sort: whether to sort sentences by size before batching
        """
        # if sort:
        #     self.sort_by_size()

        self.batches = []
        last_index = 0
        accumulated_size = 0

        for i, inst in enumerate(self.instances):
            if len(inst) + accumulated_size > words_per_batch:
                # this won't fit the last batch; finish it and start a new one
                batch = self[last_index:i]
                self.batches.append(batch)
                last_index = i
                accumulated_size = 0

            accumulated_size += len(inst)

        last_batch = self[last_index:]
        self.batches.append(last_batch)

    def shuffle_batches(self):
        """
        Shuffle the batches so that their ordering is not dependant on instance
        sizes.

        Each batch will still have instances of similar sizes.
        """
        random.shuffle(self.batches)

    def get_next_batch(self):
        """
        Return the next batch in the data. It keeps track of batches internally
        and will wrap around them once they are finished.
        """
        batch = self.batches[self.next_batch_pointer]
        self.next_batch_pointer += 1
        self.next_batch_pointer %= len(self.batches)

        return batch

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

    def shuffle_instances(self):
        """
        Shuffle the data stored by this object. It has no effect on batches if
        they have already been prepared.
        """
        zipped = self._zip_data()
        random.shuffle(zipped)
        self._unzip_data(zipped)
