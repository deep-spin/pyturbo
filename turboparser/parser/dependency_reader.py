from ..classifier.reader import AuxiliaryReader, Reader
from .dependency_instance import DependencyInstance, MultiwordSpan
from .constants import ROOT


class ConllReader(Reader):
    """
    Reader class for reading dependency trees from conllu files.
    """
    def __init__(self):
        super(ConllReader, self).__init__(AuxiliaryConllReader)


class AuxiliaryConllReader(AuxiliaryReader):

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
        root_string = ROOT
        forms = [root_string] * length
        lemmas = [root_string] * length
        upos = [root_string] * length
        xpos = [root_string] * length
        # list comprehension creates different dicts
        morph_tags = [{} for _ in range(length)]
        morph_singletons = [root_string] * length
        heads = [-1] * length
        relations = [root_string] * length
        for i in range(1, length):
            info = sentence_fields[i-1]
            forms[i] = info[1]
            lemmas[i] = info[2]
            upos[i] = info[3]
            xpos[i] = info[4]
            morph = info[5]
            morph_singletons[i] = morph
            if morph != '_':
                pairs = morph.split('|')
                for pair in pairs:
                    key, value = pair.split('=')
                    morph_tags[i][key] = value

            head = info[6]
            if head != '_':
                heads[i] = int(info[6])
                if heads[i] < 0 or heads[i] >= length:
                    raise ValueError(
                        'Invalid value of head (%d) not in range [1..%d]'
                        % (heads[i], length))
            relations[i] = info[7]

        instance = DependencyInstance(forms, lemmas, upos, xpos, morph_tags,
                                      morph_singletons, heads, relations,
                                      multiwords)

        return instance


def read_instances(path):
    """
    Read instances from the given path and change them to the format
    used internally.

    Returned instances are not formatted; i.e., they have non-numeric
    attributes.

    :param path: path to a file
    :return: list of instances (not formatted)
    """
    instances = []
    reader = ConllReader()

    with reader.open(path) as r:
        for instance in r:
            instances.append(instance)

    return instances
