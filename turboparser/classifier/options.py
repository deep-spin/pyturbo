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

        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument('--train', action='store_true',
                          help='Training mode')
        mode.add_argument('--test', action='store_true',
                          help='Test/running mode')
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
        parser.add_argument('--decay', type=float, default=0.9,
                            help="""Decay value to multiply learning rate after 
                            an epoch without improvement in the validation 
                            set.""")
        parser.add_argument('--log_interval', type=int, default=20,
                            help="""Steps between each log report""")
        parser.add_argument('--eval_interval', type=int, default=100,
                            help="""Steps between each model evaluation""")
        parser.add_argument('--learning_rate', type=float, default=0.003,
                            help='Neural model learning rate')
        parser.add_argument('--beta1', type=float, default=0.9,
                            help="""Beta1 parameter of the adam optimizer""")
        parser.add_argument('--beta2', type=float, default=0.95,
                            help="""Beta2 parameter of the adam optimizer""")
        parser.add_argument('--max_steps', type=int, default=50000,
                            help='''Maximum number of training steps (batches). 
                            If the model stops improving it stops earlier.''')
        parser.add_argument('--patience', type=int, default=5,
                            help='''Number of evaluations without 
                            improvement in the validation set to wait before 
                            terminating training.''')
        parser.add_argument('--regularization_constant', type=float,
                            default=1e12,
                            help='Regularization parameter C.')
        parser.add_argument('--batch_size', type=int, default=3000,
                            help='Number of words per batch')
        parser.add_argument('-v', '--verbose', action='store_true',
                            help='Verbose mode with some extra information '
                                 'about the models')
        parser.add_argument('--seed', type=int, default=6,
                            help='Random seed')

        self.parser = parser

    def parse_args(self):
        return self.parser.parse_args()
