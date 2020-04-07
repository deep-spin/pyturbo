import torch
import random
import numpy as np

from turboparser.parser import DependencyOptionParser, TurboParser
from turboparser.commons.utils import get_logger, configure_logger


def main():
    """Main function for the dependency parser."""
    # Parse arguments.
    option_parser = DependencyOptionParser()
    options = option_parser.parse_args()

    configure_logger(options.verbose)
    set_seeds(options.seed)

    if options.train:
        train_parser(options)
    elif options.test:
        test_parser(options)


def set_seeds(seed):
    np.seterr(over='warn', under='warn')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train_parser(options):
    logger = get_logger()
    logger.info('Starting parser in training mode')
    dependency_parser = TurboParser(options)
    log_options(dependency_parser)
    dependency_parser.train()


def test_parser(options):
    logger = get_logger()
    logger.info('Starting parser in inference mode')
    dependency_parser = TurboParser.load(options)
    log_options(dependency_parser)
    dependency_parser.run()


def log_options(parser):
    """Log parser options"""
    logger = get_logger()
    msg = 'Parser options: ' + str(parser.options)
    logger.debug(msg)


if __name__ == '__main__':
    main()
