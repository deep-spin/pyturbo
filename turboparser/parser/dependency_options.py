from ..classifier.options import OptionParser
from .constants import string2objective


class ModelType(object):
    """Dummy class to store the types of parts used by a parser"""
    def __init__(self, type_string):
        """
        :param type_string: a string encoding multiple types of parts:
            af: arc factored (always used)
            cs: consecutive siblings
            gp: grandparents
            as: arbitrary siblings
            hb: head bigrams
            gs: grandsiblings
            ts: trisiblings

            More than one type must be concatenated by +, e.g., af+cs+gp
        """
        codes = type_string.lower().split('+')
        self.consecutive_siblings = 'cs' in codes
        self.grandparents = 'gp' in codes
        self.grandsiblings = 'gs' in codes
        self.arbitrary_siblings = 'as' in codes
        self.head_bigrams = 'hb' in codes
        self.trisiblings = 'ts' in codes
        self.first_order = not any(
            [self.consecutive_siblings, self.grandparents, self.grandsiblings,
             self.arbitrary_siblings, self.head_bigrams, self.trisiblings])
        self.code = '+'.join(sorted(codes))

    def __str__(self):
        return self.code

    def __repr__(self):
        return self.code


class DependencyOptionParser(OptionParser):
    '''Options for the dependency parser.'''
    def __init__(self):
        super(DependencyOptionParser, self).__init__(
            prog='Turbo parser.',
            description='Trains/test a dependency parser.')
        parser = self.parser
        
        # Token options.
        parser.add_argument('--char_cutoff', type=int, default=1,
                            help="""Ignore characters whose frequency is less
                            than this, when using char level embeddings.""")
        parser.add_argument('--form_cutoff', type=int, default=7,
                            help="""Ignore word forms whose frequency is less
                            than this.""")
        parser.add_argument('--lemma_cutoff', type=int, default=7,
                            help="""Ignore word lemmas whose frequency is less
                            than this. This does not affect the lemmatizer.""")
        parser.add_argument('--tag_cutoff', type=int, default=1,
                            help="""Ignore POS tags whose frequency is less
                            than this.""")
        parser.add_argument('--morph_tag_cutoff', type=int, default=1,
                            help="""Ignore morph tags whose frequency is less
                            than this.""")
        parser.add_argument('--case_sensitive', action='store_true',
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
        parser.add_argument('--parsing_loss', default='global-margin',
                            choices=['local', 'global-margin', 'global-prob'],
                            help="""Type of parse loss function. 
                            local treats the head of each word as an independent
                            softmax; global-margin applies a margin loss over 
                            the complete structure; global-prob maximizies the 
                            probability of the whole structure.
                            global-margin is slower than local but tends to give
                            better results.""")
        parser.add_argument('--upos', action='store_true',
                            help='Predict UPOS tags')
        parser.add_argument('--xpos', action='store_true',
                            help='Predict XPOS tags')
        parser.add_argument('--morph', action='store_true',
                            help='Predict UMorph tags')
        parser.add_argument('--lemma', action='store_true',
                            help='Predict lemmas')
        parser.add_argument('--parse', action='store_true',
                            help='Predict parse trees')
        parser.add_argument('--single_root', action='store_true',
                            help='When running the parser, enforce that there '
                                 'is only one root per sentence.')
        parser.add_argument('--pruner_path',
                            help="""Path to a pretrained model to be used as
                            pruner. This is independent from the main model; it
                            should be called at inference time again.""")
        parser.add_argument('--pruner_batch_size', default=0, type=int,
                            help="""Batch size to be used in the pruner. If not 
                            given, the one used in the training of the pruner will
                            be used.""")
        parser.add_argument('--pruner_posterior_threshold', type=float,
                            default=0.0001,
                            help="""Posterior probability threshold for an arc
                            to be pruned, in basic pruning. For each  word m,
                            if P(h,m) < pruner_posterior_threshold * P(h',m),
                            where h' is the best scored head, then (h,m) will be
                            pruned out.""")
        parser.add_argument('--lemma_embedding_size', type=int, default=0,
                            help='Dimension of the lemma embeddings, if used')
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
        parser.add_argument('--char_embedding_size', type=int, default=100,
                            help='Size of char embeddings')
        parser.add_argument('--char_hidden_size', default=200, type=int,
                            help='''Size of the hidden char RNN (each 
                            direction)''')
        parser.add_argument('--tag_embedding_size', type=int, default=0,
                            help='Size of tag embeddings')
        parser.add_argument('--transform_size', type=int, default=125,
                            help='''Size of the linear transformation for 
                            char-based and pretrained representations''')
        parser.add_argument('--rnn_size', type=int, default=400,
                            help='Size of hidden RNN layers '
                                 '(0 to not to use RNN)')
        parser.add_argument('--arc_mlp_size', type=int, default=400,
                            help='Size of dependency arc MLP layers')
        parser.add_argument('--label_mlp_size', type=int, default=400,
                            help='Size of dependency label MLP layers')
        parser.add_argument('--ho_mlp_size', type=int, default=100,
                            help='Size of dependency higher-order MLP layers')
        parser.add_argument('--tag_mlp_size', type=int, default=100,
                            help='Size of MLP layer for tagging')
        parser.add_argument('--rnn_layers', type=int, default=2,
                            help='Number of shared RNN layers for all tasks')
        parser.add_argument('--mlp_layers', type=int, default=1,
                            help='Number of MLP layers')
        parser.add_argument('--dropout', type=float, default=0.5,
                            help='Dropout rate')
        parser.add_argument('--word_dropout', type=float, default=0.33,
                            help='Word dropout rate (replace by unknown)')
        parser.add_argument('--bert_model',
                            help='Name of the BERT model to use (blank not to '
                                 'use BERT)')
        parser.add_argument('--bert_learning_rate', default=0.00005, type=float,
                            help='Learning rate for BERT (will undergo warmup)')

    def parse_args(self):
        options = super(DependencyOptionParser, self).parse_args()

        if options.model_type == 'basic':
            options.model_type = 'af'
        elif options.model_type == 'standard':
            options.model_type = 'af+cs+gp'
        elif options.model_type == 'full':
            options.model_type = 'af+cs+gp+as+hb+gs+ts'

        options.model_type = ModelType(options.model_type)
        options.parsing_loss = string2objective[options.parsing_loss]

        return options
