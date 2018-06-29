from classifier.features import FeatureEncoder

class DependencyFeatureTypes(object):
    # Constants for feature template parts.
    ARC, NEXTSIBL, ALLSIBL, GRANDPAR, NONPROJARC, PATH, BIGRAM, NEXTSIBL_M_S, \
        ALLSIBL_M_S, GRANDPAR_G_M, GRANDSIBL, TRISIBL, GRANDSIBL_G_S = range(13)

class DependencyFeatureArc(object):
    HPMP, HWMW = range(2)

class DependencyFeatures(object):
    def __init__(self, classifier, parts):
        self.classifier = classifier
        self.input_features = [None for part in parts]

    def __getitem__(self, r):
        return self.input_features[r]

    def add_arc_features_light(self, instance, r, head, modifier):
        self.input_features[r] = []
        self._add_word_pair_features(instance, DependencyFeatureTypes.ARC,
                                     head, modifier,
                                     self.input_features[r],
                                     use_lemma_features=True,
                                     use_morph_features=True)

    def add_arc_features(self, instance, r, head, modifier):
        return self.add_arc_features_light(instance, r, head, modifier)

    def _add_word_pair_features(self, instance, type, head, modifier,
                                features,
                                use_lemma_features=False,
                                use_morph_features=False):
        # Only 4 bits are allowed for type.
        assert 0 <= type < 16

        # Words/POS.
        HLID = instance.get_lemma(head)
        MLID = instance.get_lemma(modifier)
        HWID = instance.get_form(head)
        MWID = instance.get_form(modifier)
        HPID = instance.get_tag(head)
        MPID = instance.get_tag(modifier)

        flags = type

        key = FeatureEncoder.create_key_PP(DependencyFeatureArc.HPMP,
                                           flags, HPID, MPID)
        features.append(key)

        key = FeatureEncoder.create_key_WW(DependencyFeatureArc.HWMW,
                                           flags, HWID, MWID)
        features.append(key)
