import logging
import torch
import random
import numpy as np

from turboparser.parser import DependencyOptionParser, TurboParser
from turboparser.classifier.utils import get_logger


logger = get_logger()


def main():
    """Main function for the dependency parser."""
    # Parse arguments.
    option_parser = DependencyOptionParser()
    options = option_parser.parse_args()
    set_seeds(options.seed)

    if options.train:
        train_parser(options)
    elif options.test:
        test_parser(options)


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_parser(options):
    logger.info('Training the parser')
    dependency_parser = TurboParser(options)
    log_options(dependency_parser)
    dependency_parser.train()


def test_parser(options):
    logger.info('Running the parser')
    dependency_parser = TurboParser.load(options)
    log_options(dependency_parser)
    dependency_parser.run()


def log_options(parser):
    """Log parser options"""
    msg = 'Parser options: ' + str(parser.options)
    logger.info(msg)


if __name__ == '__main__':
    main()
