import numpy as np

class FeatureEncoder(object):
    def create_key_PP(type, flags, p1, p2):
        key = np.uint64(type) | (np.uint64(flags) << np.uint(8))
        key |= (np.uint64(p1) << np.uint(16)) | (np.uint64(p2) << np.uint(24))
        return key

    def create_key_WW(type, flags, w1, w2):
        key = np.uint64(type) | (np.uint64(flags) << np.uint(8))
        key |= (np.uint64(w1) << np.uint(16)) | (np.uint64(w2) << np.uint(32))
        return key
