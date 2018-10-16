from ..classifier.reader import Reader
from .dependency_instance import DependencyInstance, \
    DependencyInstanceInput, DependencyInstanceOutput


class DependencyReader(Reader):

    def open(self):
        self.file = open(self.path, encoding='utf-8')

    def close(self):
        return self.file.close()

    def __next__(self):
        sentence_fields = []
        for line in self.file:
            line = line.strip()
            if line == '':
                if len(sentence_fields) == 0:
                    # ignore multiple empty lines
                    continue
                break
            # Ignore comment lines (necessary for CONLLU files).
            if line[0] == '#':
                continue
            fields = line.split('\t')
            # Ignore multi-word tokens (necessary for CONLLU files)
            # or ellipsed words (as in UD English EWT)
            if '-' in fields[0] or '.' in fields[0]:
                continue
            sentence_fields.append(fields)

        if not len(sentence_fields):
            raise StopIteration()

        # Sentence length (the first token is the root symbol).
        length = 1 + len(sentence_fields)
        root_string = '_root_'
        forms = [root_string] * length
        lemmas = [root_string] * length
        tags = [root_string] * length
        fine_tags = [root_string] * length
        morph_tags = [[]] * length
        morph_tags[0] = [root_string]
        heads = [-1] * length
        relations = [root_string] * length
        for i in range(1, length):
            info = sentence_fields[i-1]
            forms[i] = info[1]
            lemmas[i] = info[2]
            tags[i] = info[3]
            fine_tags[i] = info[4]
            morph = info[5]
            if morph == '_':
                morph_tags[i] = []
            else:
                morph_tags[i] = morph.split('|')
            heads[i] = int(info[6])
            if heads[i] < 0 or heads[i] >= length:
                raise ValueError(
                    'Invalid value of head (%d) not in range [0..%d]'
                    % (heads[i], length-1))
            relations[i] = info[7]
        input = DependencyInstanceInput(forms, lemmas, tags, morph_tags)
        output = DependencyInstanceOutput(heads, relations)

        return DependencyInstance(input, output)
