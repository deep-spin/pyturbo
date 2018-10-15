import logging

from turboparser.parser import DependencyOptions, TurboParser


def main():
    """Main function for the dependency parser."""
    # Parse arguments.
    import argparse
    parser = argparse. \
        ArgumentParser(prog='Turbo parser.',
                       description='Trains/test a dependency parser.')
    options = DependencyOptions(parser)
    args = vars(parser.parse_args())
    options.parse_args(args)

    if options.train:
        train_parser(options)
    elif options.test:
        test_parser(options)


def train_parser(options):
    logging.info('Training the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.train()
    dependency_parser.save()


def test_parser(options):
    logging.info('Running the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.load()
    dependency_parser.run()


if __name__ == '__main__':
    main()