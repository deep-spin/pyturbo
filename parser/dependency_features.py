from classifier.features import FeatureEncoder
import numpy as np

class DependencyFeatureTypes(object):
    # Constants for feature template parts.
    ARC, NEXTSIBL, ALLSIBL, GRANDPAR, NONPROJARC, PATH, BIGRAM, NEXTSIBL_M_S, \
        ALLSIBL_M_S, GRANDPAR_G_M, GRANDSIBL, TRISIBL, GRANDSIBL_G_S = range(13)

class DependencyFeatureArc(object):
    # There is a cross-product between these and direction, distance,
    # {word form}, {pos cpos}.
    # Features for head and modifier [add prefixes and suffixes]
    BIAS, \
    DIST, \
    BFLAG, \
    HQ, \
    MQ, \
    HP, \
    MP, \
    HW, \
    MW, \
    HWP, \
    MWP, \
    HP_MP, \
    HP_MW, \
    HP_MWP, \
    HW_MW, \
    HW_MP, \
    HWP_MP, \
    HWP_MWP, \
    HP_MP_BFLAG, \
    HW_MW_BFLAG, \
    HP_MA, \
    HP_MAP, \
    HW_MA, \
    HW_MAP, \
    HA_MP, \
    HAP_MP, \
    HA_MW, \
    HAP_MW, \
    HP_MZ, \
    HP_MZP, \
    HW_MZ, \
    HW_MZP, \
    HZ_MP, \
    HZP_MP, \
    HZ_MW, \
    HZP_MW, \
    HF, \
    HFP, \
    MF, \
    MFP, \
    HF_MF, \
    HF_MP, \
    HF_MFP, \
    HP_MF, \
    HFP_MF, \
    HFP_MFP, \
    HFP_MP, \
    HP_MFP, \
    HWF, \
    MWF, \
    HL, \
    ML, \
    pHP, \
    nHP, \
    ppHP, \
    nnHP, \
    pHQ, \
    nHQ, \
    ppHQ, \
    nnHQ, \
    pHW, \
    nHW, \
    ppHW, \
    nnHW, \
    pHL, \
    nHL, \
    ppHL, \
    nnHL, \
    pHWP, \
    nHWP, \
    ppHWP, \
    nnHWP, \
    pMP, \
    nMP, \
    ppMP, \
    nnMP, \
    pMQ, \
    nMQ, \
    ppMQ, \
    nnMQ, \
    pMW, \
    nMW, \
    ppMW, \
    nnMW, \
    pML, \
    nML, \
    ppML, \
    nnML, \
    pMWP, \
    nMWP, \
    ppMWP, \
    nnMWP, \
    HP_pHP, \
    HP_nHP, \
    HP_pHP_ppHP, \
    HP_nHP_nnHP, \
    MP_pMP, \
    MP_nMP, \
    MP_pMP_ppMP, \
    MP_nMP_nnMP, \
    HP_MP_pHP, \
    HP_MP_nHP, \
    HP_MP_pMP, \
    HP_MP_nMP, \
    HP_MP_pHP_pMP, \
    HP_MP_nHP_nMP, \
    HP_MP_pHP_nMP, \
    HP_MP_nHP_pMP, \
    HP_MP_pHP_nHP_pMP_nMP, \
    HW_MP_pHP, \
    HW_MP_nHP, \
    HW_MP_pMP, \
    HW_MP_nMP, \
    HW_MP_pHP_pMP, \
    HW_MP_nHP_nMP, \
    HW_MP_pHP_nMP, \
    HW_MP_nHP_pMP, \
    HP_MW_pHP, \
    HP_MW_nHP, \
    HP_MW_pMP, \
    HP_MW_nMP, \
    HP_MW_pHP_pMP, \
    HP_MW_nHP_nMP, \
    HP_MW_pHP_nMP, \
    HP_MW_nHP_pMP, \
    HW_MW_pHP, \
    HW_MW_nHP, \
    HW_MW_pMP, \
    HW_MW_nMP, \
    HW_MW_pHP_pMP, \
    HW_MW_nHP_nMP, \
    HW_MW_pHP_nMP, \
    HW_MW_nHP_pMP, \
    HP_MP_pHW, \
    HP_MP_nHW, \
    HP_MP_pMW, \
    HP_MP_nMW, \
    HW_MP_pHW, \
    HW_MP_nHW, \
    HW_MP_pMW, \
    HW_MP_nMW, \
    HP_MW_pHW, \
    HP_MW_nHW, \
    HP_MW_pMW, \
    HP_MW_nMW, \
    HW_MW_pHW, \
    HW_MW_nHW, \
    HW_MW_pMW, \
    HW_MW_nMW, \
    BP, \
    HP_MP_BP, \
    HW_MW_BP, \
    HW_MP_BP, \
    HP_MW_BP, \
    HP_MP_BW, \
    HW_MW_BW, \
    SHAPE = range(157)

class DependencyFeatures(object):
    def __init__(self, classifier, parts):
        self.classifier = classifier
        self.input_features = [None for part in parts]
        self.token_context = 1

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
        length = len(instance)
        labeled = not self.classifier.options.unlabeled
        feature_arc = DependencyFeatureArc()

        # Only 4 bits are allowed for type.
        assert 0 <= type < 16

        if modifier < head:
            left_position = modifier
            right_position = head
            # 0x1 if right attachment, 0x0 otherwise.
            direction_code = (0x0)
        else:
            left_position = head
            right_position = modifier
            # 0x1 if right attachment, 0x0 otherwise.
            direction_code = (0x1)
        arc_length = right_position - left_position
        # Exact arc length.
        exact_arc_length = (arc_length) if arc_length <= 0xff else 0xff
        # 7 possible values for binned_length_code (3 bits).
        if arc_length > 40:
            binned_length_code = (0x6)
        elif arc_length > 30:
            binned_length_code = (0x5)
        elif arc_length > 20:
            binned_length_code = (0x4)
        elif arc_length > 10:
            binned_length_code = (0x3)
        elif arc_length > 5:
            binned_length_code = (0x2)
        elif arc_length > 2:
            binned_length_code = (0x1)
        else:
            binned_length_code = (0x0)

        # Several flags.
        # 4 bits to denote the kind of flag.
        # Maximum will be 16 flags.
        flag_between_verb = (0x0)
        flag_between_punc = (0x1)
        flag_between_coord = (0x2)
        # TODO: This is expensive and could be precomputed.
        num_between_verb = 0
        num_between_punc = 0
        num_between_coord = 0
        for i in range(left_position + 1, right_position):
            if instance.is_verb(i):
                num_between_verb += 1
            elif instance.is_punc(i):
                num_between_punc += 1
            elif instance.is_coord(i):
                num_between_coord += 1
        # 4 bits to denote the number of occurrences for each flag.
        # Maximum will be 15 occurrences.
        max_occurrences = 15;
        if num_between_verb > max_occurrences:
            num_between_verb = max_occurrences
        if num_between_punc > max_occurrences:
            num_between_punc = max_occurrences
        if num_between_coord > max_occurrences:
            num_between_coord = max_occurrences
        flag_between_verb |= ((num_between_verb << 4))
        flag_between_punc |= ((num_between_punc << 4))
        flag_between_coord |= ((num_between_coord << 4))

        # Words/POS.
        HLID = (instance.get_lemma(head))
        MLID = (instance.get_lemma(modifier))
        HWID = (instance.get_form(head))
        MWID = (instance.get_form(modifier))
        HPID = (instance.get_tag(head))
        MPID = (instance.get_tag(modifier))

        # Contextual information.
        token_start = self.classifier.token_dictionary.token_start
        token_stop = self.classifier.token_dictionary.token_stop
        # Context size = 1:
        pHLID = (instance.get_lemma(head - 1)) \
                if head > 0 else token_start
        pMLID = (instance.get_lemma(modifier - 1)) \
                if modifier > 0 else token_start
        pHWID = (instance.get_form(head - 1)) \
                if head > 0 else token_start
        pMWID = (instance.get_form(modifier - 1)) \
                if modifier > 0 else token_start
        pHPID = (instance.get_tag(head - 1)) \
                if head > 0 else token_start
        pMPID = (instance.get_tag(modifier - 1)) \
                if modifier > 0 else token_start
        nHLID = (instance.get_lemma(head + 1)) \
                if head < length - 1 else token_stop
        nMLID = (instance.get_lemma(modifier + 1)) \
                if modifier < length - 1 else token_stop
        nHWID = (instance.get_form(head + 1)) \
                if head < length - 1 else token_stop
        nMWID = (instance.get_form(modifier + 1)) \
                if modifier < length - 1 else token_stop
        nHPID = (instance.get_tag(head + 1)) \
                if head < length - 1 else token_stop
        nMPID = (instance.get_tag(modifier + 1)) \
                if modifier < length - 1 else token_stop

        # Context size = 2:
        ppHLID = (instance.get_lemma(head - 2)) \
                if head > 1 else token_start
        ppMLID = (instance.get_lemma(modifier - 2)) \
                if modifier > 1 else token_start
        ppHWID = (instance.get_form(head - 2)) \
                if head > 1 else token_start
        ppMWID = (instance.get_form(modifier - 2)) \
                if modifier > 1 else token_start
        ppHPID = (instance.get_tag(head - 2)) \
                if head > 1 else token_start
        ppMPID = (instance.get_tag(modifier - 2)) \
                if modifier > 1 else token_start
        nnHLID = (instance.get_lemma(head + 2)) \
                if head < length - 2 else token_stop
        nnMLID = (instance.get_lemma(modifier + 2)) \
                if modifier < length - 2 else token_stop
        nnHWID = (instance.get_form(head + 2)) \
                if head < length - 2 else token_stop
        nnMWID = (instance.get_form(modifier + 2)) \
                if modifier < length - 2 else token_stop
        nnHPID = (instance.get_tag(head + 2)) \
                if head < length - 2 else token_stop
        nnMPID = (instance.get_tag(modifier + 2)) \
                if modifier < length - 2 else token_stop

        flags = (type) # 4 bits
        flags |= (direction_code << 4) # 1 more bit.

        # Bias feature (not in EGSTRA).
        key = FeatureEncoder.create_key_NONE(feature_arc.BIAS,
                                             flags)
        features.append(key)

        """
        Token features.
        """
        # Note: in EGSTRA (but not here), token and token contextual features go
        # without direction flags.
        # POS features.
        key = FeatureEncoder.create_key_P(feature_arc.HP, flags, HPID)
        features.append(key)
        # Lexical features.
        key = FeatureEncoder.create_key_W(feature_arc.HW, flags, HWID)
        features.append(key)
        if use_lemma_features:
            key = FeatureEncoder.create_key_W(feature_arc.HL,
                                              flags, HLID)
            features.append(key)

        # Features involving words and POS.
        key = FeatureEncoder.create_key_WP(feature_arc.HWP,
                                           flags, HWID, HPID)
        features.append(key)
        # Morpho-syntactic features.
        # Technically should add context here too to match egstra, but I don't
        # think it would add much relevant information.
        if use_morph_features:
            for j in range(instance.get_num_morph_tags(head)):
                HFID = (instance.get_morph_tag(head, j))
                assert HFID < 0xfff
                if j >= 0xf:
                    logging.warning('Too many morphological features (%d)' %  j)
                    HFID = (HFID << 4) | (0xf)
                else:
                    HFID = (HFID << 4) | (j)
                key = FeatureEncoder.create_key_W(feature_arc.HF,
                                                  flags, HFID)
                features.append(key)
                key = FeatureEncoder.create_key_WW(feature_arc.HWF,
                                                   flags, HWID, HFID)
                features.append(key)

        # If labeled parsing, features involving the modifier only are still
        # useful, since they will be conjoined with the label.
        if labeled:
            key = FeatureEncoder.create_key_P(feature_arc.HP,
                                              flags, MPID)
            features.append(key)
            key = FeatureEncoder.create_key_W(feature_arc.HW,
                                              flags, MWID)
            features.append(key)
            if use_lemma_features:
                key = FeatureEncoder.create_key_W(feature_arc.ML,
                                                  flags, MLID)
                features.append(key)

            key = FeatureEncoder.create_key_WP(feature_arc.MWP,
                                               flags, MWID, MPID)
            features.append(key)
            if use_morph_features:
                for j in range(instance.get_num_morph_tags(modifier)):
                    MFID = (instance.get_morph_tag(modifier, j))
                    assert MFID < 0xfff
                    if j >= 0xf:
                        logging.warning('Too many morphological features (%d)' \
                                        %  j)
                        MFID = (MFID << 4) | (0xf)
                    else:
                        MFID = (MFID << 4) | (j)
                    key = FeatureEncoder.create_key_W(feature_arc.MF,
                                                      flags, MFID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WW(feature_arc.MWF,
                                                       flags, MWID, MFID)
                    features.append(key)

        """
        Token contextual features.
        """
        if self.token_context >= 1:
            key = FeatureEncoder.create_key_P(feature_arc.pHP,
                                              flags, pHPID)
            features.append(key)
            key = FeatureEncoder.create_key_P(feature_arc.nHP,
                                              flags, nHPID)
            features.append(key)
            key = FeatureEncoder.create_key_W(feature_arc.pHW,
                                              flags, pHWID)
            features.append(key)
            key = FeatureEncoder.create_key_W(feature_arc.nHW,
                                              flags, nHWID)
            features.append(key)
            if use_lemma_features:
                key = FeatureEncoder.create_key_W(feature_arc.pHL,
                                                  flags, pHLID)
                features.append(key)
                key = FeatureEncoder.create_key_W(feature_arc.nHL,
                                                  flags, nHLID)
                features.append(key)
            key = FeatureEncoder.create_key_WP(feature_arc.pHWP,
                                               flags, pHWID, pHPID)
            features.append(key)
            key = FeatureEncoder.create_key_WP(feature_arc.nHWP,
                                               flags, nHWID, nHPID)
            features.append(key)
            # If labeled parsing, features involving the modifier only are still
            # useful, since they will be conjoined with the label.
            if labeled:
                key = FeatureEncoder.create_key_P(feature_arc.pMP,
                                                  flags, pMPID)
                features.append(key)
                key = FeatureEncoder.create_key_P(feature_arc.nMP,
                                                  flags, nMPID)
                features.append(key)
                key = FeatureEncoder.create_key_W(feature_arc.pMW,
                                                  flags, pMWID)
                features.append(key)
                key = FeatureEncoder.create_key_W(feature_arc.nMW,
                                                  flags, nMWID)
                features.append(key)
                if use_lemma_features:
                    key = FeatureEncoder.create_key_W(feature_arc.pML,
                                                      flags, pMLID)
                    features.append(key)
                    key = FeatureEncoder.create_key_W(feature_arc.nML,
                                                      flags, nMLID)
                    features.append(key)
                key = FeatureEncoder.create_key_WP(feature_arc.pMWP,
                                                   flags, pMWID, pMPID)
                features.append(key)
                key = FeatureEncoder.create_key_WP(feature_arc.nMWP,
                                                   flags, nMWID, nMPID)
                features.append(key)

        # Contextual bigram and trigram features involving POS.
        key = FeatureEncoder.create_key_PP(feature_arc.HP_pHP,
                                           flags, HPID, pHPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_pHP_ppHP,
                                            flags, HPID, pHPID, ppHPID)
        features.append(key)
        key = FeatureEncoder.create_key_PP(feature_arc.HP_nHP,
                                           flags, HPID, nHPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_nHP_nnHP,
                                            flags, HPID, nHPID, nnHPID)
        features.append(key)
        # If labeled parsing, features involving the modifier only are still
        # useful, since they will be conjoined with the label.
        if labeled:
            key = FeatureEncoder.create_key_PP(feature_arc.MP_pMP,
                                               flags, MPID, pMPID)
            features.append(key)
            key = FeatureEncoder.create_key_PPP(
                feature_arc.MP_pMP_ppMP, flags, MPID, pMPID, ppMPID)
            features.append(key)
            key = FeatureEncoder.create_key_PP(feature_arc.MP_nMP,
                                               flags, MPID, nMPID)
            features.append(key)
            key = FeatureEncoder.create_key_PPP(
                feature_arc.MP_nMP_nnMP, flags, MPID, nMPID, nnMPID)
            features.append(key)

        """
        Dependency features.
        Everything goes with direction flags and with coarse POS.
        """
        # POS features.
        key = FeatureEncoder.create_key_PP(feature_arc.HP_MP,
                                           flags, HPID, MPID)
        features.append(key)

        # Lexical/Bilexical features.
        key = FeatureEncoder.create_key_WW(feature_arc.HW_MW,
                                           flags, HWID, MWID)
        features.append(key)

        # Features involving words and POS.
        key = FeatureEncoder.create_key_WP(feature_arc.HP_MW,
                                           flags, MWID, HPID)
        features.append(key)
        key = FeatureEncoder.create_key_WPP(feature_arc.HP_MWP,
                                            flags, MWID, MPID, HPID)
        features.append(key)
        key = FeatureEncoder.create_key_WP(feature_arc.HW_MP,
                                           flags, HWID, MPID)
        features.append(key)
        key = FeatureEncoder.create_key_WPP(feature_arc.HWP_MP,
                                            flags, HWID, HPID, MPID)
        features.append(key)
        key = FeatureEncoder.create_key_WWPP(feature_arc.HWP_MWP,
                                             flags, HWID, MWID, HPID, MPID)
        features.append(key)

        # Morpho-syntactic features.
        if use_morph_features:
            for j in range(instance.get_num_morph_tags(head)):
                HFID = (instance.get_morph_tag(head, j))
                assert HFID < 0xfff
                if j >= 0xf:
                    logging.warning('Too many morphological features (%d)' \
                                    %  j)
                    HFID = (HFID << 4) | (0xf)
                else:
                    HFID = (HFID << 4) | (j)
                for k in range(instance.get_num_morph_tags(modifier)):
                    MFID = (instance.get_morph_tag(modifier, k))
                    assert MFID < 0xfff
                    if k >= 0xf:
                        logging.warning('Too many morphological features (%d)' \
                                        %  k)
                        MFID = (MFID << 4) | (0xf)
                    else:
                        MFID = (MFID << 4) | (k)
                    # Morphological features.
                    key = FeatureEncoder.create_key_WW(
                        feature_arc.HF_MF, flags, HFID, MFID)
                    features.append(key)

                    # Morphological features conjoined with POS.
                    key = FeatureEncoder.create_key_WP(
                        feature_arc.HF_MP, flags, HFID, MPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WWP(
                        feature_arc.HF_MFP, flags, HFID, MFID, MPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WP(
                        feature_arc.HP_MF, flags, MFID, HPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WWP(
                        feature_arc.HFP_MF, flags, HFID, MFID, HPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WPP(
                        feature_arc.HFP_MP, flags, HFID, HPID, MPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WPP(
                        feature_arc.HP_MFP, flags, MFID, HPID, MPID)
                    features.append(key)
                    key = FeatureEncoder.create_key_WWPP(
                        feature_arc.HP_MFP, flags,
                        HFID, MFID, HPID, MPID)
                    features.append(key)

        # Contextual features.
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_pHP,
                                            flags, HPID, MPID, pHPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_nHP,
                                            flags, HPID, MPID, nHPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_pMP,
                                            flags, HPID, MPID, pMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_nMP,
                                            flags, HPID, MPID, nMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_pHP_pMP,
                                             flags, HPID, MPID, pHPID, pMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_nHP_nMP,
                                             flags, HPID, MPID, nHPID, nMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_pHP_nMP,
                                             flags, HPID, MPID, pHPID, nMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_nHP_pMP,
                                             flags, HPID, MPID, nHPID, pMPID)
        features.append(key)
        key = FeatureEncoder.create_key_PPPPPP(
            feature_arc.HP_MP_pHP_nHP_pMP_nMP,
            flags, HPID, MPID, pHPID, nHPID, pMPID, nMPID)
        features.append(key)

        # Features for adjacent dependencies.
        if head != 0 and head == modifier - 1:
            key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_pHP,
                                                 flags, HPID, MPID, pHPID, 0x1)
            features.append(key)
            key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_nMP,
                                                 flags, HPID, MPID, nMPID, 0x1)
            features.append(key)
            key = FeatureEncoder.create_key_PPPPP(
                feature_arc.HP_MP_pHP_nMP,
                flags, HPID, MPID, pHPID, nMPID, 0x1)
            features.append(key)
        elif head != 0 and head == modifier + 1:
            key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_nHP,
                                                 flags, HPID, MPID, nHPID, 0x1)
            features.append(key)
            key = FeatureEncoder.create_key_PPPP(feature_arc.HP_MP_pMP,
                                                 flags, HPID, MPID, pMPID, 0x1)
            features.append(key)
            key = FeatureEncoder.create_key_PPPPP(
                feature_arc.HP_MP_nHP_pMP,
                flags, HPID, MPID, nHPID, pMPID, 0x1)
            features.append(key)

        # Exact arc length.
        key = FeatureEncoder.create_key_P(feature_arc.DIST, flags,
                                          exact_arc_length)
        features.append(key)

        # Binned arc length.
        # POS features conjoined with binned arc length.
        for bin in range(binned_length_code + 1):
            bin = (bin)
            key = FeatureEncoder.create_key_P(feature_arc.BIAS, flags,
                                              bin)
            features.append(key)
            key = FeatureEncoder.create_key_PP(feature_arc.HP, flags,
                                               HPID, bin)
            features.append(key)
            key = FeatureEncoder.create_key_PP(feature_arc.MP, flags,
                                               MPID, bin)
            features.append(key)
            key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP,
                                                flags, HPID, MPID, bin)
            features.append(key)

        # In-between flags.
        key = FeatureEncoder.create_key_P(feature_arc.BFLAG, flags,
                                          flag_between_verb)
        features.append(key)
        key = FeatureEncoder.create_key_P(feature_arc.BFLAG, flags,
                                          flag_between_punc)
        features.append(key)
        key = FeatureEncoder.create_key_P(feature_arc.BFLAG, flags,
                                          flag_between_coord)
        features.append(key)

        # POS features conjoined with in-between flags.
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_BFLAG,
                                            flags, HPID, MPID,
                                            flag_between_verb)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_BFLAG,
                                            flags, HPID, MPID,
                                            flag_between_punc)
        features.append(key)
        key = FeatureEncoder.create_key_PPP(feature_arc.HP_MP_BFLAG,
                                            flags, HPID, MPID,
                                            flag_between_coord)
        features.append(key)

        BPIDs = set()
        BWIDs = set()
        for i in range(left_position + 1, right_position):
            BPID = instance.get_tag(i)
            if BPID not in BPIDs:
                BPIDs.add(BPID)

                # POS in the middle.
                key = FeatureEncoder.create_key_PPP(
                    feature_arc.HP_MP_BP, flags, HPID, MPID, BPID)
                features.append(key)
                key = FeatureEncoder.create_key_WWP(
                    feature_arc.HW_MW_BP, flags, HWID, MWID, BPID)
                features.append(key)
                key = FeatureEncoder.create_key_WPP(
                    feature_arc.HW_MP_BP, flags, HWID, MPID, BPID)
                features.append(key)
                key = FeatureEncoder.create_key_WPP(
                    feature_arc.HP_MW_BP, flags, MWID, HPID, BPID)
                features.append(key)

