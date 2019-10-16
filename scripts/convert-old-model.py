# -*- coding: utf-8 -*-

import argparse
import pickle
import torch

import turboparser
from turboparser.parser import TokenDictionary
from turboparser.parser.dependency_options import ModelType
from turboparser.parser.constants import EMPTY, string2objective
import pickle

"""
Convert old models to a new format. 
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model', help='Model file')
    parser.add_argument('-o', help='Parsing objective',
                        default='global-margin', dest='objective',
                        choices=string2objective.keys())
    parser.add_argument('output', help='New model file')
    args = parser.parse_args()

    turboparser.parser.turbo_parser.ModelType = ModelType

    with open(args.model, 'rb') as f:
        data = []
        for i in range(31):
            item = pickle.load(f)
            data.append(item)
        state_dict = torch.load(f, map_location='cpu')

    options = data[0]
    del options.distance_embedding_size
    del options.regularization_constant

    options.l2 = 0.
    options.lemma = False
    options.model_type= ModelType(options.model_type)
    options.num_jobs = 1
    options.parse = True
    options.parsing_loss = string2objective[args.objective]
    options.pruner_batch_size = 0

    token_dict = TokenDictionary()
    for i in range(1, 10):
        alphabet = data[i]
        if i == 5:
            for key in alphabet:
                subalphabet = alphabet[key]
                subalphabet[EMPTY] = subalphabet['_none_']
                del subalphabet['_none_']
        else:
            alphabet[EMPTY] = alphabet['_none_']
            del alphabet['_none_']

    token_dict.character_alphabet = data[1]
    token_dict.pretrain_alphabet = data[2]
    token_dict.form_alphabet = data[3]
    token_dict.lemma_alphabet = data[4]
    token_dict.morph_tag_alphabets = data[5]
    token_dict.morph_singleton_alphabet = data[6]
    token_dict.upos_alphabet = data[7]
    token_dict.xpos_alphabet = data[8]
    token_dict.deprel_alphabet = data[9]

    embeddings = state_dict['fixed_word_embeddings.weight']
    vocab, dim = embeddings.shape
    metadata = {'fixed_embedding_vocabulary': vocab,
                'fixed_embedding_size': dim}

    data = {'options': options, 'dictionary': token_dict, 'metadata': metadata}
    with open(args.output, 'wb') as f:
        pickle.dump(data, f)
        torch.save(state_dict, f)
