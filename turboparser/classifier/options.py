import argparse


class Options(object):
    """
    Class for holding hyperparameters and setup options for structured
    classifiers.
    """
    pass


class OptionParser(object):
    '''Parser for the command line arguments used in structured classifiers.'''
    def __init__(self, prog=None, description=None):
        """
        `prog` and `description` are arguments for the argparse parser.
        """
        parser = argparse.ArgumentParser(prog=prog, description=description)
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
        parser.add_argument('--valid_path', type=str,
                            help='Path to validation data.')
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
        parser.add_argument('--batch_size', type=int, default=16,
                            help='Batch size for neural models')

        self.parser = parser

    def parse_args(self):
        import sys
        args = self.parser.parse_args()
        args = vars(args)
        self.args = args
        print(args, file=sys.stderr)

        options = Options()
        options.train = bool(args['train'])
        options.test = bool(args['test'])
        options.evaluate = bool(args['evaluate'])
        options.training_path = args['training_path']
        options.valid_path = args['valid_path']
        options.test_path = args['test_path']
        options.model_path = args['model_path']
        options.output_path = args['output_path']
        options.neural = bool(args['neural'])
        options.training_algorithm = args['training_algorithm']
        options.training_initial_learning_rate = args[
            'training_initial_learning_rate']
        options.training_learning_rate_schedule = args[
            'training_learning_rate_schedule']
        options.only_supported_features = args['only_supported_features']
        options.use_averaging = args['use_averaging']
        options.training_epochs = args['training_epochs']
        options.regularization_constant = args['regularization_constant']
        options.batch_size = args['batch_size']

        return options
