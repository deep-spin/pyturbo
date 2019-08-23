import torch
import torch.nn as nn
from torch.nn import functional as F
from .token_dictionary import TokenDictionary, UNKNOWN
from .constants import Target, SPECIAL_SYMBOLS
from ..classifier.lstm import LSTM, CharLSTM, HighwayLSTM
from ..classifier.biaffine import DeepBiaffineScorer
import numpy as np
import pickle


def create_padding_mask(lengths):
    """
    Create a mask with 1 for padding values and 0 in real values.

    :param lengths: length of each sequence
    :return: 2d tensor
    """
    batch_size = len(lengths)
    max_len = lengths.max()
    positions = torch.arange(max_len, device=lengths.device)\
        .expand(batch_size, max_len)
    mask = positions >= lengths.unsqueeze(1)

    return mask


class DependencyNeuralModel(nn.Module):
    def __init__(self,
                 model_type,
                 token_dictionary,
                 fixed_word_embeddings,
                 trainable_word_embedding_size,
                 char_embedding_size,
                 char_hidden_size,
                 transform_size,
                 tag_embedding_size,
                 distance_embedding_size,
                 rnn_size,
                 arc_mlp_size,
                 label_mlp_size,
                 ho_mlp_size,
                 rnn_layers,
                 mlp_layers,
                 dropout,
                 word_dropout,
                 predict_upos,
                 predict_xpos,
                 predict_morph,
                 tag_mlp_size):
        """
        :param model_type: a ModelType object
        :param token_dictionary: TokenDictionary object
        :type token_dictionary: TokenDictionary
        :param fixed_word_embeddings: numpy or torch embedding matrix
            (kept fixed)
        :param trainable_word_embedding_size: size for trainable word embeddings
        :param word_dropout: probability of replacing a word with the unknown
            token
        """
        super(DependencyNeuralModel, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.char_hidden_size = char_hidden_size
        self.tag_embedding_size = tag_embedding_size
        self.distance_embedding_size = distance_embedding_size
        self.transform_size = transform_size
        self.rnn_size = rnn_size
        self.arc_mlp_size = arc_mlp_size
        self.tag_mlp_size = tag_mlp_size
        self.ho_mlp_size = ho_mlp_size
        self.label_mlp_size = label_mlp_size
        self.rnn_layers = rnn_layers
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout
        self.word_dropout_rate = word_dropout
        self.on_gpu = torch.cuda.is_available()
        self.predict_upos = predict_upos
        self.predict_xpos = predict_xpos
        self.predict_morph = predict_morph
        self.predict_tags = predict_upos or predict_xpos or predict_morph
        self.model_type = model_type

        self.unknown_fixed_word = token_dictionary.get_embedding_id(UNKNOWN)
        self.unknown_trainable_word = token_dictionary.get_form_id(UNKNOWN)
        self.unknown_upos = token_dictionary.get_upos_id(UNKNOWN)
        self.unknown_xpos = token_dictionary.get_xpos_id(UNKNOWN)
        self.unknown_lemma = token_dictionary.get_lemma_id(UNKNOWN)
        morph_alphabets = token_dictionary.morph_tag_alphabets
        self.unknown_morphs = [0] * len(morph_alphabets)
        for i, feature_name in enumerate(morph_alphabets):
            alphabet = morph_alphabets[feature_name]
            self.unknown_morphs[i] = alphabet.lookup(UNKNOWN)

        rnn_input_size = 0

        if trainable_word_embedding_size:
            num_words = token_dictionary.get_num_forms()
            self.trainable_word_embeddings = nn.Embedding(
                num_words, trainable_word_embedding_size)
            num_lemmas = token_dictionary.get_num_lemmas()
            self.lemma_embeddings = nn.Embedding(
                num_lemmas, trainable_word_embedding_size)
            rnn_input_size += 2 * trainable_word_embedding_size
        else:
            self.trainable_word_embeddings = None

        if tag_embedding_size:
            # only use tag embeddings if there are actual tags, not only special
            # symbols for root, unknown, etc
            num_upos = token_dictionary.get_num_upos_tags()
            if num_upos > len(SPECIAL_SYMBOLS):
                self.upos_embeddings = nn.Embedding(num_upos,
                                                    tag_embedding_size)
            else:
                self.upos_embeddings = None

            # also check if UPOS and XPOS are not the same
            num_xpos = token_dictionary.get_num_xpos_tags()
            xpos_tags = token_dictionary.get_xpos_tags()
            upos_tags = token_dictionary.get_upos_tags()
            if num_xpos > len(SPECIAL_SYMBOLS) and \
                    upos_tags != xpos_tags:
                self.xpos_embeddings = nn.Embedding(num_xpos,
                                                    tag_embedding_size)
            else:
                self.xpos_embeddings = None

            if self.upos_embeddings is not None or \
                    self.xpos_embeddings is not None:
                # both types of POS embeddings are summed
                rnn_input_size += tag_embedding_size
            self.morph_embeddings = nn.ModuleList()
            for feature_name in morph_alphabets:
                alphabet = morph_alphabets[feature_name]
                embeddings = nn.Embedding(len(alphabet), tag_embedding_size)
                self.morph_embeddings.append(embeddings)
            rnn_input_size += tag_embedding_size
        else:
            self.upos_embeddings = None
            self.xpos_embeddings = None
            self.morph_embeddings = None

        if self.char_embedding_size:
            num_chars = token_dictionary.get_num_characters()
            self.char_rnn = CharLSTM(
                num_chars, char_embedding_size, char_hidden_size,
                dropout=dropout, bidirectional=False)

            num_directions = 1
            self.char_projection = nn.Linear(
                num_directions * char_hidden_size, transform_size, bias=False)
            rnn_input_size += transform_size

            # # tensor to replace char representation with word dropout
            # self.char_dropout_replacement = self._create_parameter_tensor(
            #     num_directions * char_hidden_size)
        else:
            self.char_rnn = None

        fixed_word_embeddings = torch.tensor(fixed_word_embeddings,
                                             dtype=torch.float)
        self.fixed_word_embeddings = nn.Embedding.from_pretrained(
            fixed_word_embeddings, freeze=True)
        self.fixed_embedding_projection = nn.Linear(
            fixed_word_embeddings.shape[1], transform_size, bias=False)
        # self.fixed_dropout_replacement = self._create_parameter_tensor(
        #     fixed_word_embeddings.shape[1])
        rnn_input_size += transform_size

        self.shared_rnn = HighwayLSTM(rnn_input_size, rnn_size, rnn_layers,
                                      dropout=self.dropout_rate)
        self.parser_rnn = HighwayLSTM(2 * rnn_size, rnn_size,
                                      dropout=self.dropout_rate)

        self.dropout_replacement = nn.Parameter(
            torch.randn(rnn_input_size) / np.sqrt(rnn_input_size))

        if self.predict_tags:
            self.tagger_rnn = LSTM(2 * rnn_size, rnn_size, dropout=dropout)
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # POS tags
        if predict_upos:
            self.upos_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=self.relu)
            num_tags = token_dictionary.get_num_upos_tags()
            self.upos_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                   bias=True)
        if predict_xpos:
            self.xpos_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=self.relu)
            num_tags = token_dictionary.get_num_xpos_tags()
            self.xpos_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                   bias=True)
        if predict_morph:
            self.morph_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=self.relu)
            num_tags = token_dictionary.get_num_morph_singletons()
            self.morph_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                    bias=True)

        # first order layers
        num_labels = token_dictionary.get_num_deprels()
        self.arc_scorer = DeepBiaffineScorer(
            2 * rnn_size, 2 * rnn_size, arc_mlp_size, 1, dropout=dropout)
        self.label_scorer = DeepBiaffineScorer(
            2 * rnn_size, 2 * rnn_size, label_mlp_size, num_labels,
            dropout=dropout)
        self.linearization_scorer = DeepBiaffineScorer(
            2 * rnn_size, 2 * rnn_size, arc_mlp_size, 1, dropout=dropout)
        self.distance_scorer = DeepBiaffineScorer(
            2 * rnn_size, 2 * rnn_size, arc_mlp_size, 1, dropout=dropout)

        # self.head_mlp = self._create_mlp()
        # self.modifier_mlp = self._create_mlp()
        # self.arc_scorer = self._create_scorer()

        # self.label_head_mlp = self._create_mlp(
        #     hidden_size=self.label_mlp_size)
        # self.label_modifier_mlp = self._create_mlp(
        #     hidden_size=self.label_mlp_size)
        # self.label_scorer = self._create_scorer(self.label_mlp_size,
        #                                         num_labels)

        # Higher order layers
        if model_type.grandparents:
            self.gp_grandparent_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.gp_head_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.gp_modifier_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.grandparent_scorer = self._create_scorer(self.ho_mlp_size)

        if model_type.consecutive_siblings:
            self.sib_head_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.sib_modifier_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.sib_sibling_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.sibling_scorer = self._create_scorer(self.ho_mlp_size)

        if model_type.consecutive_siblings or model_type.grandsiblings \
                or model_type.trisiblings or model_type.arbitrary_siblings:
            self.null_sibling_tensor = self._create_parameter_tensor(
                2 * rnn_size)

        if model_type.grandsiblings:
            self.gsib_head_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.gsib_modifier_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.gsib_sibling_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.gsib_grandparent_mlp = self._create_mlp(
                hidden_size=self.ho_mlp_size)
            self.grandsibling_scorer = self._create_scorer(self.ho_mlp_size)

        # Clear out the gradients before the next batch.
        self.zero_grad()

    def _packed_dropout(self, states):
        """Apply dropout to packed states"""
        # shared_states is a packed tuple; (data, lengths)
        states_data = states.data
        states_lengths = states.batch_sizes
        states_data = self.dropout(states_data)
        states = nn.utils.rnn.PackedSequence(states_data, states_lengths)
        return states

    def _create_parameter_tensor(self, shape):
        """
        Create a tensor for representing some special token. It is included in
        the model parameters.
        """
        tensor = torch.randn(shape) / np.sqrt(shape)
        if self.on_gpu:
            tensor = tensor.cuda()
        
        parameter = nn.Parameter(tensor)

        return parameter

    def _create_scorer(self, input_size=None, output_size=1, bias=False):
        """
        Create the weights for scoring a given tensor representation to a
        single number.

        :param input_size: expected input size. If None, arc_mlp_size
            is used.
        :return: an nn.Linear object
        """
        if input_size is None:
            input_size = self.arc_mlp_size
        linear = nn.Linear(input_size, output_size, bias=bias)
        scorer = nn.Sequential(self.dropout, linear)

        return scorer

    def _create_mlp(self, input_size=None, hidden_size=None, num_layers=None,
                    output_activation=None):
        """
        Create the weights for a fully connected subnetwork.

        The output has a linear activation; if num_layers > 1, hidden layers
        will use a non-linearity. If output_activation is given, it will be
        applied to the output.

        The first layer will have a weight matrix (input x hidden), subsequent
        layers will be (hidden x hidden).

        :param input_size: if not given, will be assumed rnn_size * 2
        :param hidden_size: if not given, will be assumed arc_mlp_size
        :param num_layers: number of hidden layers (including the last one). If
            not given, will be mlp_layers
        :return: an nn.Linear object, mapping an input with 2*hidden_units
            to hidden_units.
        """
        if input_size is None:
            input_size = self.rnn_size * 2
        if hidden_size is None:
            hidden_size = self.arc_mlp_size
        if num_layers is None:
            num_layers = self.mlp_layers

        layers = []
        for i in range(num_layers):
            if i > 0:
                layers.append(self.relu)

            linear = nn.Linear(input_size, hidden_size)
            layers.extend([self.dropout, linear])
            input_size = hidden_size

        if output_activation is not None:
            layers.append(output_activation)

        mlp = nn.Sequential(*layers)
        return mlp

    def save(self, file):
        pickle.dump(self.fixed_word_embeddings.weight.shape[0], file)
        pickle.dump(self.fixed_word_embeddings.weight.shape[1], file)
        dim = 0 if self.trainable_word_embeddings is None \
            else self.trainable_word_embeddings.weight.shape[1]
        pickle.dump(dim, file)
        pickle.dump(self.char_embedding_size, file)
        pickle.dump(self.tag_embedding_size, file)
        pickle.dump(self.distance_embedding_size, file)
        pickle.dump(self.char_hidden_size, file)
        pickle.dump(self.transform_size, file)
        pickle.dump(self.rnn_size, file)
        pickle.dump(self.arc_mlp_size, file)
        pickle.dump(self.tag_mlp_size, file)
        pickle.dump(self.label_mlp_size, file)
        pickle.dump(self.ho_mlp_size, file)
        pickle.dump(self.rnn_layers, file)
        pickle.dump(self.mlp_layers, file)
        pickle.dump(self.dropout_rate, file)
        pickle.dump(self.word_dropout_rate, file)
        pickle.dump(self.predict_upos, file)
        pickle.dump(self.predict_xpos, file)
        pickle.dump(self.predict_morph, file)
        pickle.dump(self.model_type, file)
        torch.save(self.state_dict(), file)

    @classmethod
    def load(cls, file, token_dictionary):
        fixed_embedding_vocab_size = pickle.load(file)
        fixed_embedding_size = pickle.load(file)
        trainable_embedding_size = pickle.load(file)
        char_embedding_size = pickle.load(file)
        tag_embedding_size = pickle.load(file)
        distance_embedding_size = pickle.load(file)
        char_hidden_size = pickle.load(file)
        transform_size = pickle.load(file)
        rnn_size = pickle.load(file)
        arc_mlp_size = pickle.load(file)
        tag_mlp_size = pickle.load(file)
        label_mlp_size = pickle.load(file)
        ho_mlp_size = pickle.load(file)
        rnn_layers = pickle.load(file)
        mlp_layers = pickle.load(file)
        dropout = pickle.load(file)
        word_dropout = pickle.load(file)
        predict_upos = pickle.load(file)
        predict_xpos = pickle.load(file)
        predict_morph = pickle.load(file)
        model_type = pickle.load(file)

        dummy_embeddings = np.empty([fixed_embedding_vocab_size,
                                     fixed_embedding_size], np.float32)
        model = DependencyNeuralModel(
            model_type, token_dictionary, dummy_embeddings,
            trainable_embedding_size,
            char_embedding_size,
            tag_embedding_size=tag_embedding_size,
            distance_embedding_size=distance_embedding_size,
            char_hidden_size=char_hidden_size,
            transform_size=transform_size,
            rnn_size=rnn_size,
            arc_mlp_size=arc_mlp_size,
            tag_mlp_size=tag_mlp_size,
            label_mlp_size=label_mlp_size,
            ho_mlp_size=ho_mlp_size,
            rnn_layers=rnn_layers,
            mlp_layers=mlp_layers,
            dropout=dropout,
            word_dropout=word_dropout,
            predict_upos=predict_upos, predict_xpos=predict_xpos,
            predict_morph=predict_morph)

        if model.on_gpu:
            state_dict = torch.load(file)
        else:
            state_dict = torch.load(file, map_location='cpu')

        # kind of a hack to allow compatibility with previous versions
        own_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           k in own_state_dict}
        own_state_dict.update(pretrained_dict)
        model.load_state_dict(own_state_dict)

        return model

    def _compute_arc_scores(self, states, lengths):
        """
        Compute the first order scores and store them in the appropriate
        position in the `scores` tensor.

        The score matrices have shape (modifer, head), and do not have the row
        corresponding to the root as a modifier. Thus they have shape
        (num_words - 1, num_words).

        :param states: hidden states returned by the RNN; one for each word
        :param lengths: length of each sentence in the batch (including root)
        """
        batch_size, max_sent_size, _ = states.size()

        # apply dropout separately to get different masks
        # head_scores is interpreted as (batch, modifier, head)
        head_scores = self.arc_scorer(self.dropout(states),
                                      self.dropout(states)).squeeze(3)
        s1 = self.dropout(states)
        s2 = self.dropout(states)
        label_scores = self.label_scorer(s1, s2)

        # set arc scores from each word to itself as -inf
        diag = torch.eye(max_sent_size, dtype=torch.uint8, device=states.device)
        diag = diag.unsqueeze(0)
        head_scores.masked_fill_(diag, -np.inf)

        # set padding head scores to -inf
        # during training, label loss is computed with respect to the gold
        # arcs, so there's no need to set -inf scores to invalid positions
        # in label scores.
        padding_mask = create_padding_mask(lengths)
        head_scores = head_scores.masked_fill(padding_mask.unsqueeze(1),
                                              -np.inf)

        # linearization (scoring heads after/before modifier)
        arange = torch.arange(max_sent_size, device=states.device)
        position1 = arange.view(1, 1, -1).expand(batch_size, -1, -1)
        position2 = arange.view(1, -1, 1).expand(batch_size, -1, -1)
        head_offset = position1 - position2
        sign_scores = self.linearization_scorer(self.dropout(states),
                                                self.dropout(states)).squeeze(3)
        sign_sigmoid = F.logsigmoid(
            sign_scores * torch.sign(head_offset).float()).detach()
        head_scores += sign_sigmoid

        # score distances between head and modifier
        dist_scores = self.distance_scorer(self.dropout(states),
                                           self.dropout(states)).squeeze(3)
        dist_pred = 1 + F.softplus(dist_scores)
        dist_target = torch.abs(head_offset)

        # KL divergence between predicted distances and actual ones
        dist_kld = -torch.log((dist_target.float() - dist_pred) ** 2 / 2 + 1)
        head_scores += dist_kld.detach()

        # exclude attachment for the root symbol
        head_scores = head_scores[:, 1:]
        label_scores = label_scores[:, 1:]
        sign_scores = sign_scores[:, 1:]
        dist_kld = dist_kld[:, 1:]

        self.scores[Target.HEADS] = head_scores
        self.scores[Target.RELATIONS] = label_scores
        self.scores[Target.SIGN] = sign_scores
        self.scores[Target.DISTANCE] = dist_kld

    def _compute_grandparent_scores(self, states, parts):
        """`
        Compute the grandparent scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        """
        # there may be no grandparent parts in some cases
        if parts.get_num_type(Target.GRANDPARENTS) == 0:
            empty = torch.tensor([], device=states.device)
            self.scores[Target.GRANDPARENTS].append(empty)
            return

        head_tensors = self.gp_head_mlp(states)
        grandparent_tensors = self.gp_grandparent_mlp(states)
        modifier_tensors = self.gp_modifier_mlp(states)

        head_indices = []
        modifier_indices = []
        grandparent_indices = []

        for part in parts.part_lists[Target.GRANDPARENTS]:
            # list all indices, then feed the corresponding tensors to the net
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        part_states = self.tanh(heads + modifiers + grandparents)
        part_scores = self.grandparent_scorer(part_states)

        self.scores[Target.GRANDPARENTS].append(part_scores.view(-1))

    def _compute_consecutive_sibling_scores(self, states, parts):
        """
        Compute the consecutive sibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        """
        # include the vector for null sibling
        # word_sibling_tensors is (num_words=1, hidden_units)
        states_and_sibling = torch.cat([states,
                                        self.null_sibling_tensor.view(1, -1)])

        head_tensors = self.sib_head_mlp(states)
        modifier_tensors = self.sib_modifier_mlp(states)
        sibling_tensors = self.sib_sibling_mlp(states_and_sibling)

        head_indices = []
        modifier_indices = []
        sibling_indices = []

        for part in parts.part_lists[Target.NEXT_SIBLINGS]:
            # list all indices to the candidate head/modifier/siblings, then
            # process them all at once for faster execution.
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            if part.sibling == 0:
                # sibling == 0 or -1 indicates there's no sibling to the left
                # (to the right, sibling == len(states))
                sibling_indices.append(len(states))
            else:
                sibling_indices.append(part.sibling)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        siblings = sibling_tensors[sibling_indices]
        sibling_states = self.tanh(heads + modifiers + siblings)
        sibling_scores = self.sibling_scorer(sibling_states)

        self.scores[Target.NEXT_SIBLINGS].append(sibling_scores.view(-1))

    def _compute_grandsibling_scores(self, states, parts):
        """
        Compute the consecutive grandsibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        """
        # include the vector for null sibling
        # word_sibling_tensors is (num_words=1, hidden_units)
        states_and_sibling = torch.cat([states,
                                        self.null_sibling_tensor.view(1, -1)])

        head_tensors = self.gsib_head_mlp(states)
        modifier_tensors = self.gsib_modifier_mlp(states)
        sibling_tensors = self.gsib_sibling_mlp(states_and_sibling)
        grandparent_tensors = self.gsib_grandparent_mlp(states)

        head_indices = []
        modifier_indices = []
        sibling_indices = []
        grandparent_indices = []

        for part in parts.part_lists[Target.GRANDSIBLINGS]:
            # list all indices to the candidate head/mod/sib/grandparent
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)
            if part.sibling == 0:
                # sibling == 0 or -1 indicates there's no sibling to the left
                # (to the right, sibling == len(states))
                sibling_indices.append(len(states))
            else:
                sibling_indices.append(part.sibling)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        siblings = sibling_tensors[sibling_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        gsib_states = self.tanh(heads + modifiers + siblings + grandparents)
        gsib_scores = self.grandsibling_scorer(gsib_states)

        self.scores[Target.GRANDSIBLINGS].append(gsib_scores.view(-1))

    def get_word_representations(self, instances, max_length):
        """
        Get the full embedding representations of words in the batch, including
        word type embeddings, char level and POS tag embeddings.

        :param instances: list of instance objects
        :param max_length: length of the longest instance in the batch
        :return: a tensor with shape (batch, max_num_tokens, embedding_size)
        """
        all_embeddings = []
        word_embeddings = self._get_embeddings(
            instances, max_length, 'fixedword')
        projection = self.fixed_embedding_projection(word_embeddings)
        all_embeddings.append(projection)

        if self.trainable_word_embeddings is not None:
            trainable_embeddings = self._get_embeddings(instances, max_length,
                                                        'trainableword')
            all_embeddings.append(trainable_embeddings)
        if self.lemma_embeddings is not None:
            lemma_embeddings = self._get_embeddings(instances, max_length,
                                                    'lemma')
            all_embeddings.append(lemma_embeddings)

        upos = 0 if self.upos_embeddings is None \
            else self._get_embeddings(instances, max_length, 'upos')
        xpos = 0 if self.xpos_embeddings is None \
            else self._get_embeddings(instances, max_length, 'xpos')
        pos_embeddings = upos + xpos
        if pos_embeddings is not 0:
            all_embeddings.append(pos_embeddings)

        if self.morph_embeddings is not None:
            morph_embeddings = self._get_embeddings(instances, max_length,
                                                    'morph')
            all_embeddings.append(morph_embeddings)

        if self.char_rnn is not None:
            char_embeddings = self._run_char_rnn(instances, max_length)
            projection = self.char_projection(self.dropout(char_embeddings))
            all_embeddings.append(projection)

        # each embedding tensor is (batch, num_tokens, embedding_size)
        embeddings = torch.cat(all_embeddings, dim=2)

        if self.word_dropout_rate:
            if self.training:
                # apply word dropout -- replace by a random tensor
                dropout_draw = torch.rand_like(embeddings[:, :, 0])
                inds = dropout_draw < self.word_dropout_rate
                embeddings[inds] = self.dropout_replacement
            else:
                # weight embeddings by the training dropout rate
                embeddings *= (1 - self.word_dropout_rate)
                embeddings += self.word_dropout_rate * self.dropout_replacement

        return embeddings

    def _get_embeddings(self, instances, max_length, type_):
        """
        Get the word or tag embeddings for all tokens in the instances.

        This function takes care of padding.

        :param type_: 'fixedword', 'trainableword', 'upos' or 'xpos'
        :param max_length: length of the longest instance
        :return: a tensor with shape (batch, sequence, embedding size)
        """
        if type_ == 'morph':
            # morph features have multiple embeddings (one for each feature)
            num_features = len(self.morph_embeddings)
            shape = (len(instances), max_length, num_features)
            index_tensor = torch.zeros(
                shape, dtype=torch.long,
                device=self.morph_embeddings[0].weight.device)

            for i, instance in enumerate(instances):
                indices = instance.get_all_morph_tags()
                index_tensor[i, :len(instance)] = torch.tensor(indices)

            embedding_sum = 0
            for i, matrix in enumerate(self.morph_embeddings):
                indices = index_tensor[:, :, i]
                # if self.training and self.word_dropout_rate:
                #     #TODO: avoid some repeated code
                #     indices = indices.clone()
                #     dropout_draw = torch.rand_like(indices,
                #                                    dtype=torch.float)
                #     drop_indices = dropout_draw < self.word_dropout_rate
                #     indices[drop_indices] = self.unknown_morphs[i]

                # embeddings is (batch, max_length, num_units)
                embeddings = matrix(indices)
                embedding_sum = embedding_sum + embeddings

            return embedding_sum

        shape = (len(instances), max_length)
        index_matrix = torch.zeros(shape, dtype=torch.long)
        for i, instance in enumerate(instances):
            if type_ == 'fixedword':
                indices = instance.get_all_embedding_ids()
            elif type_ == 'trainableword':
                indices = instance.get_all_forms()
            elif type_ == 'lemma':
                indices = instance.get_all_lemmas()
            elif type_ == 'upos':
                indices = instance.get_all_upos()
            elif type_ == 'xpos':
                indices = instance.get_all_xpos()
            else:
                raise ValueError('Invalid embedding type: %s' % type_)

            index_matrix[i, :len(instance)] = torch.tensor(indices)

        if type_ == 'fixedword':
            embedding_matrix = self.fixed_word_embeddings
            unknown_symbol = self.unknown_fixed_word
        elif type_ == 'trainableword':
            embedding_matrix = self.trainable_word_embeddings
            unknown_symbol = self.unknown_trainable_word
        elif type_ == 'lemma':
            embedding_matrix = self.lemma_embeddings
            unknown_symbol = self.unknown_lemma
        else:
            if type_ == 'upos':
                embedding_matrix = self.upos_embeddings
                unknown_symbol = self.unknown_upos
            else:
                embedding_matrix = self.xpos_embeddings
                unknown_symbol = self.unknown_xpos

        # if self.training and self.word_dropout_rate and type_ != 'fixedword':
        #     dropout_draw = torch.rand_like(index_matrix, dtype=torch.float)
        #     inds = dropout_draw < self.word_dropout_rate
        #     index_matrix[inds] = unknown_symbol

        if self.on_gpu:
            index_matrix = index_matrix.cuda()

        embeddings = embedding_matrix(index_matrix)

        # if self.training and self.word_dropout_rate and type_ == 'fixedword':
        #     # since the embedding matrix is fixed, we use a separate
        #     # trainable dropout tensor
        #     dropout_draw = torch.rand_like(embeddings[:, :, 0])
        #     inds = dropout_draw < self.word_dropout_rate
        #     embeddings[inds] = self.fixed_dropout_replacement

        return embeddings

    def _run_char_rnn(self, instances, max_sentence_length):
        """
        Run a RNN over characters in all words in the batch instances.

        :param instances:
        :return: a tensor with shape (batch, sequence, embedding size)
        """
        batch_size = len(instances)
        token_lengths_ = [[len(inst.get_characters(i))
                           for i in range(len(inst))]
                          for inst in instances]
        max_token_length = max(max(inst_lengths)
                               for inst_lengths in token_lengths_)

        shape = [batch_size, max_sentence_length]
        token_lengths = torch.zeros(shape, dtype=torch.long)

        shape = [batch_size, max_sentence_length, max_token_length]
        char_indices = torch.zeros(shape, dtype=torch.long)

        for i, instance in enumerate(instances):
            token_lengths[i, :len(instance)] = torch.tensor(token_lengths_[i])

            for j in range(len(instance)):
                # each j is a token
                chars = instance.get_characters(j)
                char_indices[i, j, :len(chars)] = torch.tensor(chars)

        char_representation = self.char_rnn(char_indices, token_lengths)
        # if self.training and self.word_dropout_rate:
        #     # sample a dropout mask and replace the dropped representations
        #     shape = [batch_size, max_sentence_length]
        #     dropout_mask = torch.rand(shape) < self.word_dropout_rate
        #     char_representation[dropout_mask] = self.char_dropout_replacement

        return char_representation

    def convert_arc_scores_to_parts(self, instances, parts):
        """
        Convert the stored matrices with arc scores and label scores to 1d
        arrays, in the same order as in parts. Masks are also applied.

        :param instances: list of DependencyInstanceNumeric
        :param parts: a DependencyParts object
        """
        new_head_scores = []
        new_label_scores = []

        # arc_mask has shape (head, modifier) but scores are
        # (modifier, head); so we transpose
        all_head_scores = torch.transpose(self.scores[Target.HEADS], 1, 2)
        all_label_scores = torch.transpose(self.scores[Target.RELATIONS], 1, 2)

        for i, instance in enumerate(instances):
            inst_parts = parts[i]
            mask = inst_parts.arc_mask
            mask = torch.tensor(mask.astype(np.uint8))
            length = len(instance)

            # get a matrix [inst_length, inst_length - 1]
            # (root has already been discarded as a modifier)
            head_scores = all_head_scores[i, :length, :length - 1]

            # get a tensor [inst_length, inst_length - 1, num_labels]
            label_scores = all_label_scores[i, :length, :length - 1]

            mask = mask[:, 1:]
            head_scores1d = head_scores[mask]
            label_scores1d = label_scores[mask].view(-1)

            if self.training:
                # apply the margin on the scores of gold parts
                gold_arc_parts = torch.tensor(
                    inst_parts.gold_parts[:inst_parts.num_arcs])

                offset = inst_parts.offsets[Target.RELATIONS]
                num_labeled = inst_parts.num_labeled_arcs
                gold_label_parts = torch.tensor(
                    inst_parts.gold_parts[offset:offset + num_labeled])
                head_scores1d = head_scores1d - gold_arc_parts
                label_scores1d = label_scores1d - gold_label_parts

            new_head_scores.append(head_scores1d)
            new_label_scores.append(label_scores1d)

        self.scores[Target.HEADS] = new_head_scores
        self.scores[Target.RELATIONS] = new_label_scores

    def forward(self, instances, parts, normalization='local'):
        """
        :param instances: a list of DependencyInstance objects
        :param parts: a list of DependencyParts objects
        :param normalization: either "local" or "global". It only affects
            first order parts (arcs and labeled arcs).

            If "local", the losses for each word (as a modifier) is computed
            independently. The model will store a tensor with all arc scores
            (including padding) for efficient loss computation.

            If "global", the loss is a hinge margin over the global structure.
            The model will store scores as a list of 1d arrays (without padding)
            that can easily be used with AD3 decoding functions.

        :param gold_heads: tensor (batch, max_num_words) with gold heads. Only
            used if training with global margin loss
        :param gold_labels: same as above, with gold label for each token.
        :return: a score matrix with shape (num_instances, longest_length)
        """
        self.scores = {Target.HEADS: []}
        if parts[0].labeled:
            self.scores[Target.RELATIONS] = []
        for type_ in parts[0].part_lists:
            self.scores[type_] = []

        batch_size = len(instances)
        lengths = torch.tensor([len(instance) for instance in instances],
                               dtype=torch.long)
        if self.on_gpu:
            lengths = lengths.cuda()

        # packed sequences must be sorted by decreasing length
        sorted_lengths, inds = lengths.sort(descending=True)
        _, rev_inds = inds.sort()
        if self.on_gpu:
            sorted_lengths = sorted_lengths.cuda()

        # instances = [instances[i] for i in inds]
        max_length = sorted_lengths[0].item()
        embeddings = self.get_word_representations(instances, max_length)
        sorted_embeddings = embeddings[inds]

        # pack to account for variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            sorted_embeddings, sorted_lengths, batch_first=True)

        shared_states, _ = self.shared_rnn(packed_embeddings)
        # shared_states = self._packed_dropout(shared_states)
        parser_packed_states, _ = self.parser_rnn(shared_states)

        # batch_states is (batch, num_tokens, hidden_size)
        parser_batch_states, _ = nn.utils.rnn.pad_packed_sequence(
            parser_packed_states, batch_first=True)

        # return to the original ordering
        parser_batch_states = parser_batch_states[rev_inds]

        if self.predict_tags:
            tagger_packed_states, _ = self.tagger_rnn(shared_states)
            tagger_batch_states, _ = nn.utils.rnn.pad_packed_sequence(
                tagger_packed_states, batch_first=True)
            tagger_batch_states = tagger_batch_states[rev_inds]

            if self.predict_upos:
                # ignore root
                hidden = self.upos_mlp(tagger_batch_states[:, 1:])
                self.scores[Target.UPOS] = self.upos_scorer(hidden)

            if self.predict_xpos:
                hidden = self.xpos_mlp(tagger_batch_states[:, 1:])
                self.scores[Target.XPOS] = self.xpos_scorer(hidden)

            if self.predict_morph:
                hidden = self.morph_mlp(tagger_batch_states[:, 1:])
                self.scores[Target.MORPH] = self.morph_scorer(hidden)

        self._compute_arc_scores(parser_batch_states, lengths)

        # now go through each batch item
        for i in range(batch_size):
            length = lengths[i].item()
            states = parser_batch_states[i, :length]
            sent_parts = parts[i]

            if self.model_type.consecutive_siblings:
                self._compute_consecutive_sibling_scores(states, sent_parts)

            if self.model_type.grandparents:
                self._compute_grandparent_scores(states, sent_parts)

            if self.model_type.grandsiblings:
                self._compute_grandsibling_scores(states, sent_parts)

        if normalization == 'global':
            self.convert_arc_scores_to_parts(instances, parts)

        return self.scores
