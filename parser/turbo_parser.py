from classifier.structured_classifier import StructuredClassifier
from parser.dependency_options import DependencyOptions
from parser.dependency_reader import DependencyReader
from parser.dependency_writer import DependencyWriter
from parser.dependency_decoder import DependencyDecoder
from parser.dependency_dictionary import DependencyDictionary
from parser.dependency_instance_numeric import DependencyInstanceNumeric
from parser.token_dictionary import TokenDictionary
from parser.dependency_parts import DependencyParts
import logging

class TurboParser(StructuredClassifier):
    '''Dependency parser.'''
    def __init__(self, options):
        StructuredClassifier.__init__(self, options)
        self.token_dictionary = TokenDictionary(self)
        self.dictionary = DependencyDictionary(self)
        self.reader = DependencyReader()
        self.writer = DependencyWriter()
        self.decoder = DependencyDecoder()
        self.parameters = None
        self.token_dictionary.initialize(self.reader)
        self.dictionary.create_relation_dictionary(self.reader)

    def get_formatted_instance(self, instance):
        return DependencyInstanceNumeric(instance, self.dictionary)

    def make_parts(self, instance):
        return self.make_parts_basic(instance, add_relation_parts=False)

    def make_parts_basic(self, instance, add_relation_parts=True):
        make_gold = instance.output != None # Check this.
        if make_gold:
            gold_outputs = []

        parts = DependencyParts()
        if add_relation_parts and not self.options.prune_relations:
            allowed_relations = range(len(
                self.dictionary.get_relation_alphabet()))

        num_parts_initial = len(parts)
        for h in range(len(instance)):
            for m in range(1, len(instance)):
                if h == m:
                    continue
                if add_relation_parts:
                    # If no unlabeled arc is there, just skip it.
                    # This happens if that arc was pruned out.
                    if 0 > parts.find_arc(h, m):
                        continue
                else:
                    if h and self.options.prune_distances:
                        modifier_tag = instance.get_tag(m)
                        head_tag = instance.get_tag(h)
                        if h < m:
                            # Right attachment.
                            if m - h > \
                               self.dictionary.get_maximum_right_distance(
                                   modifier_tag, head_tag):
                                continue
                        else:
                            # Left attachment.
                            if h - m > \
                               self.dictionary.get_maximum_left_distance(
                                   modifier_tag, head_tag):
                                continue
                if self.options.prune_relations:
                    modifier_tag = instance.get_tag(m)
                    head_tag = instance.get_tag(h)
                    allowed_relations = []
                    allowed_relations = self.dictionary.get_existing_relations(
                        modifier_tag, head_tag)
                    if not add_relation_parts and not allowed_relations:
                        continue

                # Add parts for labeled/unlabeled arcs.
                if add_relation_parts:
                    # If there is no allowed relation for this arc, but the
                    # unlabeled arc was added, then it was forced to be present
                    # to maintain connectivity of the graph. In that case (which
                    # should be pretty rare) consider all the possible
                    # relations.
                    if not allowed_relations:
                        allowed_relations = range(len(
                            self.dictionary.get_relation_alphabet()))
                    for l in allowed_relations:
                        part = parts.create_part_labeled_arc(h, m, l)
                        parts.append(part)
                        if make_gold:
                            if instance.get_head(m) == h and \
                               instance.get_relation(m) == l:
                                gold_outputs.append(1.)
                            else:
                                gold_outputs.append(0.)
                else:
                    part = parts.create_part_arc(h, m)
                    parts.append(part)
                    if make_gold:
                        if instance.get_head(m) == h:
                            gold_outputs.append(1.)
                        else:
                            gold_outputs.append(0.)

        # When adding unlabeled arcs, make sure the graph stays connected.
        # Otherwise, enforce connectedness by adding some extra arcs
        # that connect words to the root.
        # NOTE: if --projective, enforcing connectedness is not enough,
        # so we add arcs of the form m-1 -> m to make sure the sentence
        # has a projective parse.
        if not add_relation_parts:
            arcs = parts[num_parts_initial:]
            inserted_arcs = self.enforce_well_formed_graph(instance, arcs)
            for h, m in inserted_arcs:
                part = parts.create_part_arc(h, m)
                parts.append(part)
                if make_gold:
                    if instance.get_head(m) == h:
                        gold_outputs.append(1.)
                    else:
                        gold_outputs.append(0.)
            parts.set_offset_arc(num_parts_initial,
                                 len(parts) - num_parts_initial)
        else:
            parts.set_offset_labeled_arc(num_parts_initial,
                                 len(parts) - num_parts_initial)

    def enforce_well_formed_graph(self, instance, arcs):
        raise NotImplementedError
        return []


def main():
    '''Main function for the dependency parser.'''
    # Parse arguments.
    import argparse
    parser = argparse. \
        ArgumentParser(prog='Turbo parser.',
                       description='Trains/test a dependency parser.')
    options = DependencyOptions(parser)
    args = vars(parser.parse_args())
    options.parse_args(args)

    if options.train:
        logging.info('Training parser...')
        train_parser(options)
    elif options.test:
        logging.info('Running parser...')
        test_parser(options)

def train_parser(options):
    logging.info('Training the parser...')
    dependency_parser = TurboParser(options)
    dependency_parser.train()
    dependency_parser.save_model()

if __name__ == "__main__":
    main()
