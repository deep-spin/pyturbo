from classifier.writer import Writer
from parser.dependency_instance import DependencyInstance

class DependencyWriter(Writer):
    def __init__(self):
        self.file = None

    def open(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def close(self):
        self.file.close()

    def write(self, instance):
        for i in range(1, len(instance)):
            self.file.write('\t'.join([instance.input.forms[i],
                                       instance.input.lemmas[i],
                                       instance.input.tag[i],
                                       instance.input.tag[i],
                                       '_',
                                       str(instance.output.heads[i]),
                                       instance.input.relations[i]]) + '\n')
        self.file.write('\n')
