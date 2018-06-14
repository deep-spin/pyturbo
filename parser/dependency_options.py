from classifier.options import Options

class DependencyOptions(Options):
    '''Options for the dependency parser.'''
    def __init__(self, parser):
        Options.__init__(self, parser)
        parser.add_argument('--form_cutoff', type=int, default=0,
                            help="""Ignore word forms whose frequency is less
                            than this.""")
        parser.add_argument('--lemma_cutoff', type=int, default=0,
                            help="""Ignore word lemmas whose frequency is less
                            than this.""")
        parser.add_argument('--tag_cutoff', type=int, default=0,
                            help="""Ignore POS tags whose frequency is less
                            than this.""")
        parser.add_argument('--morph_tag_cutoff', type=int, default=0,
                            help="""Ignore morph tags whose frequency is less
                            than this.""")
        parser.add_argument('--prefix_length', type=int, default=4,
                            help='Length of prefixes')
        parser.add_argument('--suffix_length', type=int, default=4,
                            help='Length of suffixes')
        parser.add_argument('--form_case_sensitive', action='store_const',
                            default=0, const=1,
                            help='Distinguish upper/lower case of word forms.')

    def parse_args(self, args):
        Options.parse_args(self, args)
        self.form_cutoff = args['form_cutoff']
        self.lemma_cutoff = args['lemma_cutoff']
        self.tag_cutoff = args['tag_cutoff']
        self.morph_tag_cutoff = args['morph_tag_cutoff']
        self.prefix_length = args['prefix_length']
        self.suffix_length = args['suffix_length']
        self.form_case_sensitive = bool(args['form_case_sensitive'])
