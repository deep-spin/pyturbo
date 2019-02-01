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
        parser.add_argument('--projective', action='store_true',
                            help="""Force the parser output single-rooted
                            projective trees.""")
        parser.add_argument('--upos', action='store_true',
                            help='Predict UPOS tags')
        parser.add_argument('--xpos', action='store_true',
                            help='Predict XPOS tags')
        parser.add_argument('--morph', action='store_true',
                            help='Predict UMorph tags')
        parser.add_argument('--single_root', action='store_true',
                            help='When running the parser, enforce that there '
                                 'is only one root per sentence.')
        parser.add_argument('--prune_relations', type=int, default=0,
                            help="""1 for pruning the set of possible relations
                            taking into account the labels that have occured for
                            each pair of POS tags in the training data.""")
        parser.add_argument('--prune_distances', type=int, default=1,
                            help="""1 for pruning the set of possible left/right
                            distances taking into account the distances that
                            have occured for each pair of POS tags in the
                            training data.""")
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
                            followed by the values of its vector.""")
        parser.add_argument('--embedding_size', help="""If an embeddings file is
                            not given, specify the size of randomly generated 
                            embeddings.""", default=100, type=int)
        parser.add_argument('--char_embedding_size', type=int, default=100,
                            help='Size of char embeddings')
        parser.add_argument('--tag_embedding_size', type=int, default=20,
                            help='Size of tag embeddings')
        parser.add_argument('--distance_embedding_size', type=int, default=20,
                            help='Size of distance embeddings')
        parser.add_argument('--rnn_size', type=int, default=100,
                            help='Size of hidden RNN layers')
        parser.add_argument('--mlp_size', type=int, default=100,
                            help='Size of hidden head MLP layers')
        parser.add_argument('--label_mlp_size', type=int, default=100,
                            help='Size of hidden dependency label MLP layers')
        parser.add_argument('--pos_mlp_size', type=int, default=100,
                            help='Size of hidden POS MLP layer')
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
        parser.add_argument('--learning_rate', type=float, default=0.001,
                            help='Neural model learning rate')

        """
// Options for pruner training.
// TODO: implement these options.
DEFINE_string(pruner_train_algorithm, "crf_mira",
              "Training algorithm for the pruner. Options are perceptron, mira, "
              "svm_mira, crf_mira, svm_sgd, crf_sgd.");
DEFINE_bool(pruner_only_supported_features, true,
            "True for the pruner to use supported features only (should be true"
            "for CRFs).");
DEFINE_bool(pruner_use_averaging, true,
            "True for the pruner to average the weight vector at the end of"
            "training.");
DEFINE_int32(pruner_train_epochs, 10,
             "Number of training epochs for the pruner.");
DEFINE_double(pruner_train_regularization_constant, 0.001,
              "Regularization parameter C for the pruner.");
DEFINE_bool(pruner_labeled, false,
            "True if pruner is a labeled parser. Currently, must be set to false.");
DEFINE_double(pruner_train_initial_learning_rate, 0.01,
              "Initial learning rate of pruner (for SGD only).");
DEFINE_string(pruner_train_learning_rate_schedule, "invsqrt",
              "Learning rate annealing schedule of pruner (for SGD only). "
              "Options are fixed, lecun, invsqrt, inv.");
DEFINE_bool(pruner_large_feature_set, false,
            "True for using a large feature set in the pruner.");
        """

    def parse_args(self):
        options = super(DependencyOptionParser, self).parse_args()
        args = self.args

        options.char_cutoff = args['char_cutoff']
        options.form_cutoff = args['form_cutoff']
        options.lemma_cutoff = args['lemma_cutoff']
        options.tag_cutoff = args['tag_cutoff']
        options.morph_tag_cutoff = args['morph_tag_cutoff']
        options.prefix_length = args['prefix_length']
        options.suffix_length = args['suffix_length']
        options.form_case_sensitive = args['form_case_sensitive']

        options.model_type = args['model_type']
        options.unlabeled = bool(args['unlabeled'])
        options.predict_upos = args['upos']
        options.predict_xpos = args['xpos']
        options.predict_morph = args['morph']
        options.projective = args['projective']
        options.prune_relations = bool(args['prune_relations'])
        options.prune_distances = bool(args['prune_distances'])
        options.pruner_path = args['pruner_path']
        options.pruner_posterior_threshold = args['pruner_posterior_threshold']
        options.pruner_max_heads = args['pruner_max_heads']
        options.single_root = args['single_root']

        options.embeddings = args['embeddings']
        options.embedding_size = args['embedding_size']
        options.char_embedding_size = args['char_embedding_size']
        options.tag_embedding_size = args['tag_embedding_size']
        options.distance_embedding_size = args['distance_embedding_size']
        options.rnn_size = args['rnn_size']
        options.mlp_size = args['mlp_size']
        options.pos_mlp_size = args['pos_mlp_size']
        options.label_mlp_size = args['label_mlp_size']
        options.rnn_layers = args['rnn_layers']
        options.mlp_layers = args['mlp_layers']
        options.dropout = args['dropout']
        options.word_dropout = args['word_dropout']
        options.tag_dropout = args['tag_dropout']
        options.learning_rate = args['learning_rate']

        if options.model_type == 'basic':
            options.model_type = 'af'
        elif options.model_type == 'standard':
            options.model_type = 'af+cs+gp'
        elif options.model_type == 'full':
            options.model_type = 'af+cs+gp+as+hb+gs+ts'

        return options
