from ..classifier.options import OptionParser


class DependencyOptionParser(OptionParser):
    '''Options for the dependency parser.'''
    def __init__(self):
        super(DependencyOptionParser, self).__init__(
            prog='Turbo parser.',
            description='Trains/test a dependency parser.')
        parser = self.parser
        
        # Token options.
        parser.add_argument('--char_cutoff', type=int, default=5,
                            help="""Ignore characters whose frequency is less
                            than this, when using char level embeddings.""")
        parser.add_argument('--form_cutoff', type=int, default=2,
                            help="""Ignore word forms whose frequency is less
                            than this.""")
        parser.add_argument('--lemma_cutoff', type=int, default=2,
                            help="""Ignore word lemmas whose frequency is less
                            than this.""")
        parser.add_argument('--tag_cutoff', type=int, default=2,
                            help="""Ignore POS tags whose frequency is less
                            than this.""")
        parser.add_argument('--morph_tag_cutoff', type=int, default=2,
                            help="""Ignore morph tags whose frequency is less
                            than this.""")
        parser.add_argument('--prefix_length', type=int, default=4,
                            help='Length of prefixes')
        parser.add_argument('--suffix_length', type=int, default=4,
                            help='Length of suffixes')
        parser.add_argument('--form_case_sensitive', action='store_true',
                            help='Distinguish upper/lower case of word forms.')

        # Parser options.
        parser.add_argument('--model_type', type=str, default='standard',
                            help="""Model type. This a string formed by the one
                            or several of the following pieces:
                            af enables arc-factored parts (required),
                            +cs enables consecutive sibling parts,
                            +gp enables grandparent parts,
                            +gs enables grandsibling parts,
                            +ts enables trisibling parts,
                            +as enables arbitrary sibling parts,
                            +hb enables head bigram parts,
                            +gs enables grand-sibling (third-order) parts,
                            +ts enables tri-sibling (third-order) parts.
                            The following alias are predefined:
                            basic is af,
                            standard is af+cs+gp,
                            full is af+cs+gp+as+hb+gs+ts.""")
        parser.add_argument('--unlabeled', action='store_const',
                            default=0, const=1,
                            help="""Make the parser output just the backbone
                            dependencies.""")
        parser.add_argument('--upos', action='store_true',
                            help='Predict UPOS tags')
        parser.add_argument('--xpos', action='store_true',
                            help='Predict XPOS tags')
        parser.add_argument('--morph', action='store_true',
                            help='Predict UMorph tags')
        parser.add_argument('--single_root', action='store_true',
                            help='When running the parser, enforce that there '
                                 'is only one root per sentence.')
        parser.add_argument('--prune_relations', action='store_true',
                            help="""Prune the set of possible relations for
                            each pair of POS tags in the training data.""")
        parser.add_argument('--prune_tags', action='store_true',
                            help="""Prune arcs with a combination of POS tags 
                            unseen in training data.""")
        parser.add_argument('--pruner_path',
                            help="""Path to a pretrained model to be used as
                            pruner. This is independent from the main model; it
                            should be called at inference time again.""")
        parser.add_argument('--pruner_posterior_threshold', type=float,
                            default=0.0001,
                            help="""Posterior probability threshold for an arc
                            to be pruned, in basic pruning. For each  word m,
                            if P(h,m) < pruner_posterior_threshold * P(h',m),
                            where h' is the best scored head, then (h,m) will be
                            pruned out.""")
        parser.add_argument('--pruner_max_heads', type=int,
                            default=10,
                            help="""Maximum number of possible head words for a
                            given word, in basic pruning.""")
        parser.add_argument('--embeddings', help="""File with text embeddings,
                            optionally xzipped.  
                            First line must have number of words and number 
                            of dimensions; each other line must have a word
                            followed by the values of its vector. These are
                            kept frozen.""")
        parser.add_argument('--embedding_size', help="""Dimension of trainable
                            embeddings.""", default=100, type=int)
        parser.add_argument('--char_embedding_size', type=int, default=100,
                            help='Size of char embeddings')
        parser.add_argument('--tag_embedding_size', type=int, default=20,
                            help='Size of tag embeddings')
        parser.add_argument('--distance_embedding_size', type=int, default=20,
                            help='Size of distance embeddings')
        parser.add_argument('--rnn_size', type=int, default=100,
                            help='Size of hidden RNN layers')
        parser.add_argument('--arc_mlp_size', type=int, default=100,
                            help='Size of dependency arc MLP layers')
        parser.add_argument('--label_mlp_size', type=int, default=100,
                            help='Size of dependency label MLP layers')
        parser.add_argument('--ho_mlp_size', type=int, default=100,
                            help='Size of dependency higher-order MLP layers')
        parser.add_argument('--tag_mlp_size', type=int, default=100,
                            help='Size of MLP layer for tagging')
        parser.add_argument('--rnn_layers', type=int, default=1,
                            help='Number of RNN layers')
        parser.add_argument('--mlp_layers', type=int, default=1,
                            help='Number of MLP layers')
        parser.add_argument('--dropout', type=float, default=0,
                            help='Dropout rate')
        parser.add_argument('--word_dropout', type=float, default=0,
                            help='Word dropout rate (replace by unknown)')
        parser.add_argument('--tag_dropout', type=float, default=0,
                            help='Tag dropout rate (replace by unknown)')

    def parse_args(self):
        options = super(DependencyOptionParser, self).parse_args()

        if options.model_type == 'basic':
            options.model_type = 'af'
        elif options.model_type == 'standard':
            options.model_type = 'af+cs+gp'
        elif options.model_type == 'full':
            options.model_type = 'af+cs+gp+as+hb+gs+ts'

        return options
