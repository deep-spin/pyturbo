'''This implements a dictionary of labels.'''

class Alphabet(dict):
    '''This class implements a dictionary of labels. Labels as mapped to
    integers, and it is efficient to retrieve the label name from its
    integer representation, and vice-versa.'''
    def __init__(self, label_names=None):
        dict.__init__(self)
        self.locked = False
        self.names = []
        if label_names != None:
            for name in label_names:
                self.add(name)

    def clear(self):
        dict.clear(self)
        self.names = []

    def insert(self, name):
        '''Add new label.'''
        if name in self:
            return self[name]
        elif self.locked:
            return -1
        else:
            label_id = len(self.names)
            self[name] = label_id
            self.names.append(name)
            return label_id

    def lookup(self, name):
        '''Lookup label.'''
        if name in self:
            return self[name]
        else:
            return -1

    def stop_growth(self):
        self.locked = True

    def allow_growth(self):
        self.locked = False

    def get_label_name(self, label_id):
        '''Get label name from id.'''
        return self.names[label_id]

    def get_label_id(self, name):
        '''Get label id from name.'''
        return self[name]

    def save(self, label_file):
        '''Save labels to a file.'''
        f = open(label_file, 'w')
        for name in self.names:
            f.write(name + '\n')
        f.close()

    def load(self, label_file):
        '''Load labels from a file.'''
        self.names = []
        self.clear()
        f = open(label_file)
        for line in f:
            name = line.rstrip('\n')
            self.add(name)
        f.close()
