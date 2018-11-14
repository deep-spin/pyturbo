from ..classifier.reader import AuxiliaryReader
from .dependency_instance import DependencyInstance, \
    DependencyInstanceInput, DependencyInstanceOutput, MultiwordSpan


class DependencyReader(AuxiliaryReader):

    def __next__(self):
        sentence_fields = []
        multiwords = []

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

            # Ignore ellipsed words (as in UD English EWT)
            if '.' in fields[0]:
                continue

            # Store multiword tokens (necessary for CONLLU output files)
            if '-' in fields[0]:
                span = fields[0].split('-')
                first = int(span[0])
                last = int(span[1])
                form = fields[1]
                multiword = MultiwordSpan(first, last, form)
                multiwords.append(multiword)
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
        input_ = DependencyInstanceInput(forms, lemmas, tags, morph_tags,
                                         multiwords)
        output = DependencyInstanceOutput(heads, relations)

        return DependencyInstance(input_, output)
