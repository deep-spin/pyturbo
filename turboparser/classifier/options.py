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
        parser.add_argument('--decay', type=float, default=0.99,
                            help="""Decay value to multiply learning rate after 
                            an epoch without improvement in the validation 
                            set.""")
        parser.add_argument('--beta1', type=float, default=0.9,
                            help="""Beta1 parameter of the adam optimizer""")
        parser.add_argument('--beta2', type=float, default=0.999,
                            help="""Beta2 parameter of the adam optimizer""")
        parser.add_argument('--training_epochs', type=int, default=10,
                            help='Number of training epochs.''')
        parser.add_argument('--patience', type=int, default=5,
                            help='Number of epochs without improvements in the'
                                 ' validation set to wait before terminating '
                                 'training.')
        parser.add_argument('--regularization_constant', type=float,
                            default=1e12,
                            help='Regularization parameter C.')
        parser.add_argument('--batch_size', type=int, default=16,
                            help='Batch size for neural models')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose mode with some extra information '
                                 'about the models')

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
        options.training_path = args['training_path']
        options.valid_path = args['valid_path']
        options.test_path = args['test_path']
        options.model_path = args['model_path']
        options.output_path = args['output_path']
        options.training_epochs = args['training_epochs']
        options.regularization_constant = args['regularization_constant']
        options.batch_size = args['batch_size']
        options.patience = args['patience']
        options.decay = args['decay']
        options.beta1 = args['beta1']
        options.beta2 = args['beta2']
        options.verbose = args['verbose']

        return options
