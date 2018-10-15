import numpy as np

class FeatureEncoder(object):
    def create_key_NONE(type, flags):
        key = (type) | ((flags) << (8))
        return key

    def create_key_W(type, flags, w):
        key = (type) | ((flags) << (8))
        key |= ((w) << (16))
        return key

    def create_key_WP(type, flags, w, p):
        key = (type) | ((flags) << (8))
        key |= ((w) << (16)) | ((p) << (32))
        return key

    def create_key_WPP(type, flags, w, p1, p2):
        key = (type) | ((flags) << (8))
        key |= ((w) << (16)) | ((p1) << (32)) \
               | ((p2) << (40))
        return key

    def create_key_WPPP(type, flags, w, p1, p2, p3):
        key = (type) | ((flags) << (8))
        key |= ((w) << (16)) | ((p1) << (32)) \
               | ((p2) << (40)) | ((p3) << (48))
        return key

    def create_key_WPPPP(type, flags, w, p1, p2, p3, p4):
        key = (type) | ((flags) << (8))
        key |= ((w) << (16)) | ((p1) << (32)) \
               | ((p2) << (40)) \
               | ((p3) << (48)) \
               | ((p4) << (56))
        return key

    def create_key_WW(type, flags, w1, w2):
        key = (type) | ((flags) << (8))
        key |= ((w1) << (16)) | ((w2) << (32))
        return key

    def create_key_WWW(type, flags, w1, w2, w3):
        key = (type) | ((flags) << (8))
        key |= ((w1) << (16)) | ((w2) << (32)) \
               | ((w3) << (48))
        return key

    def create_key_WWPP(type, flags, w1, w2, p1, p2):
        key = (type) | ((flags) << (8))
        key |= ((w1) << (16)) | ((w2) << (32)) \
               | ((p1) << (48)) | ((p2) << (56))
        return key

    def create_key_WWP(type, flags, w1, w2, p):
        key = (type) | ((flags) << (8))
        key |= ((w1) << (16)) | ((w2) << (32)) \
               | ((p) << (48))
        return key

    def create_key_P(type, flags, p):
        key = (type) | ((flags) << (8))
        key |= ((p) << (16))
        return key

    def create_key_PP(type, flags, p1, p2):
        key = (type) | ((flags) << (8))
        key |= ((p1) << (16)) | ((p2) << (24))
        return key

    def create_key_PPP(type, flags, p1, p2, p3):
        key = (type) | ((flags) << (8))
        key |= ((p1) << (16)) | ((p2) << (24)) \
               | ((p3) << (32))
        return key

    def create_key_PPPP(type, flags, p1, p2, p3, p4):
        key = (type) | ((flags) << (8))
        key |= ((p1) << (16)) | ((p2) << (24)) \
               | ((p3) << (32)) | ((p4) << (40))
        return key

    def create_key_PPPPP(type, flags, p1, p2, p3, p4, p5):
        key = (type) | ((flags) << (8))
        key |= ((p1) << (16)) | ((p2) << (24)) \
               | ((p3) << (32)) \
               | ((p4) << (40)) \
               | ((p5) << (48))
        return key

    def create_key_PPPPPP(type, flags, p1, p2, p3, p4, p5, p6):
        key = (type) | ((flags) << (8))
        key |= ((p1) << (16)) | ((p2) << (24)) \
               | ((p3) << (32)) \
               | ((p4) << (40)) \
               | ((p5) << (48)) \
               | ((p6) << (56))
        return key

    def create_key_S(type, flags, s):
        key = (type) | ((flags) << (8))
        key |= ((s) << (16))
        return key
