from ..classifier.writer import Writer

num_fields = 10
multiword_blanks = '\t'.join(['_'] * 8)


class DependencyWriter(Writer):
    def __init__(self):
        self.file = None

    def open(self, path):
        self.file = open(path, 'w', encoding='utf-8')

    def close(self):
        self.file.close()

    def write(self, instance):
        # keep track of multiword tokens
        multiwords = instance.input.multiwords
        multiword = multiwords[0] if len(multiwords) else None
        multiword_idx = 0

        for i in range(1, len(instance)):
            if multiword and i == multiword.first:
                span = '%d-%d' % (multiword.first, multiword.last)
                line = '%s\t%s\t%s\n' % (span, multiword.form, multiword_blanks)
                self.file.write(line)

                multiword_idx += 1
                if multiword_idx >= len(multiwords):
                    multiword = None
                else:
                    multiword = multiwords[multiword_idx]

            if instance.output.tags is not None:
                tags = instance.output.tags
            else:
                tags = instance.input.tags
            line = '\t'.join([str(i),
                              instance.input.forms[i],
                              instance.input.lemmas[i],
                              tags[i],
                              instance.input.fine_tags[i],
                              '_',
                              str(instance.output.heads[i]),
                              instance.output.relations[i],
                              '_', '_'],) + '\n'
            self.file.write(line)
        self.file.write('\n')
