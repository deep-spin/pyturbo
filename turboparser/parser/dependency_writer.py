
class DependencyWriter(object):
    def __init__(self):
        self.file = None

    def open(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def close(self):
        self.file.close()

    def write(self, instance):
        self.file.write(instance.to_conll())
        self.file.write('\n')
