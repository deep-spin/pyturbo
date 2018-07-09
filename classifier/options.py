class Options(object):
    '''Options for a general structured classifier.'''
    def __init__(self, parser):
        parser.add_argument('--train', action='store_const',
                            default=0, const=1,
                            help='1 for training the classifier.')
        parser.add_argument('--test', action='store_const',
                            default=0, const=1,
                            help='1 for testing the classifier.')
        parser.add_argument('--evaluate', action='store_const',
                            default=0, const=1,
                            help="""1 for evaluating the classifier 
                            (requires --test).""")
        parser.add_argument('--training_path', type=str, default=None,
                            help='Path to the training data.')
        parser.add_argument('--test_path', type=str, default=None,
                            help='Path to the test data.')
        parser.add_argument('--model_path', type=str, default=None,
                            help='Path to the model.')
        parser.add_argument('--output_path', type=str, default=None,
                            help='Path to the output predictions.')
        parser.add_argument('--neural', action='store_const',
                            default=0, const=1,
                            help='1 for using a neural classifier.')
        parser.add_argument('--training_algorithm', type=str,
                            default='svm_mira',
                            help="""Training algorithm. Options are 
                            'perceptron', 'mira', 'svm_mira', 'crf_mira', 
                            'svm_sgd', 'crf_sgd'.""")
        parser.add_argument('--training_initial_learning_rate', type=float,
                            default=.01,
                            help='Initial learning rate (SGD only).')
        parser.add_argument('--training_learning_rate_schedule', type=str,
                            default='invsqrt',
                            help="""Learning rate annealing schedule (SGD 
                            only). Options are 'fixed', 'lecun', 'invsqrt', 
                            'inv'.""")
        parser.add_argument('--only_supported_features', action='store_const',
                            default=0, const=1,
                            help="""1 for using supported features only 
                            (should be 1 for CRFs).""")
        parser.add_argument('--use_averaging', type=int, default=1,
                            help="""1 for using averaging the weight vector
                            at the end of training.""")
        parser.add_argument('--training_epochs', type=int, default=10,
                            help='Number of training epochs.''')
        parser.add_argument('--regularization_constant', type=float,
                            default=1e12,
                            help='Regularization parameter C.')

    def parse_args(self, args):
        import sys
        print(args, file=sys.stderr)
        self.train = bool(args['train'])
        self.test = bool(args['test'])
        self.evaluate = bool(args['evaluate'])
        self.training_path = args['training_path']
        self.test_path = args['test_path']
        self.model_path = args['model_path']
        self.output_path = args['output_path']
        self.neural = bool(args['neural'])
        self.training_algorithm = args['training_algorithm']
        self.training_initial_learning_rate = args[
            'training_initial_learning_rate']
        self.training_learning_rate_schedule = args[
            'training_learning_rate_schedule']
        self.only_supported_features = args['only_supported_features']
        self.use_averaging = args['use_averaging']
        self.training_epochs = args['training_epochs']
        self.regularization_constant = args['regularization_constant']

