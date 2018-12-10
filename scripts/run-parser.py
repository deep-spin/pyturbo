import logging

from turboparser.parser import DependencyOptionParser, TurboParser


def main():
    """Main function for the dependency parser."""
    # Parse arguments.
    parser = DependencyOptionParser()
    options = parser.parse_args()

    if options.train:
        train_parser(options)
    elif options.test:
        test_parser(options)


def train_parser(options):
    logging.info('Training the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.train()


def test_parser(options):
    logging.info('Running the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.load()
    dependency_parser.run()


if __name__ == '__main__':
    main()
