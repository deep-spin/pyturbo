import torch
import torch.nn as nn
from .token_dictionary import TokenDictionary, UNKNOWN
from .constants import Target
from ..classifier.lstm import LSTM
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
                 word_embeddings,
                 char_embedding_size,
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
                 tag_dropout,
                 predict_upos,
                 predict_xpos,
                 predict_morph,
                 tag_mlp_size):
        """
        :param model_type: a ModelType object
        :param token_dictionary: TokenDictionary object
        :type token_dictionary: TokenDictionary
        :param word_embeddings: numpy or torch embedding matrix
        :param word_dropout: probability of replacing a word with the unknown
            token
        :param tag_dropout: probability of replacing a POS tag with the unknown
            tag
        """
        super(DependencyNeuralModel, self).__init__()
        self.embedding_vocab_size = word_embeddings.shape[0]
        self.word_embedding_size = word_embeddings.shape[1]
        self.char_embedding_size = char_embedding_size
        self.tag_embedding_size = tag_embedding_size
        self.distance_embedding_size = distance_embedding_size
        self.rnn_size = rnn_size
        self.arc_mlp_size = arc_mlp_size
        self.tag_mlp_size = tag_mlp_size
        self.ho_mlp_size = ho_mlp_size
        self.label_mlp_size = label_mlp_size
        self.rnn_layers = rnn_layers
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout
        self.word_dropout_rate = word_dropout
        self.tag_dropout_rate = tag_dropout
        self.unknown_word = token_dictionary.get_embedding_id(UNKNOWN)
        self.unknown_upos = token_dictionary.get_upos_id(UNKNOWN)
        self.unknown_xpos = token_dictionary.get_xpos_id(UNKNOWN)
        self.on_gpu = torch.cuda.is_available()
        self.predict_upos = predict_upos
        self.predict_xpos = predict_xpos
        self.predict_morph = predict_morph
        self.predict_tags = predict_upos or predict_xpos or predict_morph
        self.model_type = model_type

        word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings,
                                                            freeze=False)

        if self.char_embedding_size:
            char_vocab = token_dictionary.get_num_characters()
            self.char_embeddings = nn.Embedding(char_vocab, char_embedding_size)

            self.char_rnn = LSTM(
                input_size=char_embedding_size, hidden_size=char_embedding_size)
            char_based_embedding_size = 2 * char_embedding_size

            # tensor to replace char representation with word dropout
            self.char_dropout_replacement = self._create_parameter_tensor(
                char_based_embedding_size)
        else:
            self.char_embeddings = None
            self.char_rnn = None
            char_based_embedding_size = 0

        total_tag_embedding_size = 0
        if tag_embedding_size:
            # only use tag embeddings if there are actual tags
            # 3 means root, unk and the placeholder "_" when there are no tags
            num_upos = token_dictionary.get_num_upos_tags()
            if num_upos > 3:
                self.upos_embeddings = nn.Embedding(num_upos,
                                                    tag_embedding_size)
                total_tag_embedding_size += tag_embedding_size
            else:
                self.upos_embeddings = None

            # also check if UPOS and XPOS are not the same
            num_xpos = token_dictionary.get_num_xpos_tags()
            xpos_tags = token_dictionary.get_xpos_tags()
            upos_tags = token_dictionary.get_upos_tags()
            if num_xpos > 3 and \
                    upos_tags != xpos_tags:
                self.xpos_embeddings = nn.Embedding(num_xpos,
                                                    tag_embedding_size)
                total_tag_embedding_size += tag_embedding_size
            else:
                self.xpos_embeddings = None
        else:
            self.upos_embeddings = None
            self.xpos_embeddings = None

        if self.distance_embedding_size:
            bins = np.array(list(range(1, 10)) + list(range(10, 31, 5)))
            self.distance_bins = np.concatenate([-bins[::-1], bins])
            self.distance_embeddings = nn.Embedding(len(self.distance_bins) * 2,
                                                    distance_embedding_size)
        else:
            self.distance_bins = None
            self.distance_embeddings = None

        input_size = self.word_embedding_size + total_tag_embedding_size + \
                     char_based_embedding_size
        self.shared_rnn = LSTM(input_size, rnn_size, rnn_layers, dropout)
        self.parser_rnn = LSTM(2 * rnn_size, rnn_size, dropout=dropout)
        if self.predict_tags:
            self.tagger_rnn = LSTM(2 * rnn_size, rnn_size, dropout=dropout)
        self.rnn_hidden_size = 2 * rnn_size
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
        self.head_mlp = self._create_mlp()
        self.modifier_mlp = self._create_mlp()
        self.arc_scorer = self._create_scorer()

        self.label_head_mlp = self._create_mlp(
            hidden_size=self.label_mlp_size)
        self.label_modifier_mlp = self._create_mlp(
            hidden_size=self.label_mlp_size)
        num_labels = token_dictionary.get_num_deprels()
        self.label_scorer = self._create_scorer(self.label_mlp_size,
                                                num_labels)

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
            self.null_sibling_tensor = self._create_parameter_tensor()

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

        if self.distance_embedding_size:
            self.distance_projector = nn.Linear(
                distance_embedding_size,
                arc_mlp_size,
                bias=True)
            self.label_distance_projector = nn.Linear(
                distance_embedding_size,
                label_mlp_size,
                bias=True
            )
        else:
            self.distance_mlp = None

        # Clear out the gradients before the next batch.
        self.zero_grad()

    def _packed_dropout(self, states):
        """Apply dropout to packed states"""
        # shared_states is a packed tuple; (data, lengths)
        states_data, states_lengths = states
        states_data = self.dropout(states_data)
        states = nn.utils.rnn.PackedSequence(states_data, states_lengths)
        return states

    def _create_parameter_tensor(self, shape=None):
        """
        Create a tensor for representing some special token. It is included in
        the model parameters.

        If shape is None, it will have shape equal to rnn_hidden_size.
        """
        if shape is None:
            shape = self.rnn_hidden_size

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
        pickle.dump(self.embedding_vocab_size, file)
        pickle.dump(self.word_embedding_size, file)
        pickle.dump(self.char_embedding_size, file)
        pickle.dump(self.tag_embedding_size, file)
        pickle.dump(self.distance_embedding_size, file)
        pickle.dump(self.rnn_size, file)
        pickle.dump(self.arc_mlp_size, file)
        pickle.dump(self.tag_mlp_size, file)
        pickle.dump(self.label_mlp_size, file)
        pickle.dump(self.ho_mlp_size, file)
        pickle.dump(self.rnn_layers, file)
        pickle.dump(self.mlp_layers, file)
        pickle.dump(self.dropout_rate, file)
        pickle.dump(self.word_dropout_rate, file)
        pickle.dump(self.tag_dropout_rate, file)
        pickle.dump(self.predict_upos, file)
        pickle.dump(self.predict_xpos, file)
        pickle.dump(self.predict_morph, file)
        pickle.dump(self.model_type, file)
        torch.save(self.state_dict(), file)

    @classmethod
    def load(cls, file, token_dictionary):
        embedding_vocab_size = pickle.load(file)
        word_embedding_size = pickle.load(file)
        char_embedding_size = pickle.load(file)
        tag_embedding_size = pickle.load(file)
        distance_embedding_size = pickle.load(file)
        rnn_size = pickle.load(file)
        mlp_size = pickle.load(file)
        tag_mlp_size = pickle.load(file)
        label_mlp_size = pickle.load(file)
        ho_mlp_size = pickle.load(file)
        rnn_layers = pickle.load(file)
        mlp_layers = pickle.load(file)
        dropout = pickle.load(file)
        word_dropout = pickle.load(file)
        tag_dropout = pickle.load(file)
        predict_upos = pickle.load(file)
        predict_xpos = pickle.load(file)
        predict_morph = pickle.load(file)
        model_type = pickle.load(file)

        dummy_embeddings = np.empty([embedding_vocab_size,
                                     word_embedding_size], np.float32)
        model = DependencyNeuralModel(
            model_type, token_dictionary, dummy_embeddings, char_embedding_size,
            tag_embedding_size=tag_embedding_size,
            distance_embedding_size=distance_embedding_size,
            rnn_size=rnn_size,
            arc_mlp_size=mlp_size,
            tag_mlp_size=tag_mlp_size,
            label_mlp_size=label_mlp_size,
            ho_mlp_size=ho_mlp_size,
            rnn_layers=rnn_layers,
            mlp_layers=mlp_layers,
            dropout=dropout,
            word_dropout=word_dropout, tag_dropout=tag_dropout,
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

    def _compute_arc_scores(self, states, parts):
        """
        Compute the first order scores and store them in the appropriate
        position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object
        :type parts: DependencyParts
        """
        head_tensors = self.head_mlp(states)
        modifier_tensors = self.modifier_mlp(states)
        head_indices, modifier_indices = parts.get_arc_indices()

        if self.distance_embedding_size:
            distance_diff = head_indices - modifier_indices
            distance_indices = np.digitize(distance_diff, self.distance_bins)
            distance_indices = torch.tensor(distance_indices, dtype=torch.long)
            if self.on_gpu:
                distance_indices = distance_indices.cuda()

            distances = self.distance_embeddings(distance_indices)
            distance_projections = self.distance_projector(distances)
            distance_projections = distance_projections.view(
                -1, self.arc_mlp_size)
        else:
            distance_projections = 0

        # now index all of them to process at once
        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]

        arc_states = self.tanh(heads + modifiers + distance_projections)
        arc_scores = self.arc_scorer(arc_states)
        self.scores[Target.HEADS].append(arc_scores.view(-1))

        if not parts.labeled:
            return

        head_tensors = self.label_head_mlp(states)
        modifier_tensors = self.label_modifier_mlp(states)

        # we can reuse indices -- every LabeledArc must also appear as Arc
        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        label_states = self.tanh(heads + modifiers)
        label_scores = self.label_scorer(label_states)
        self.scores[Target.RELATIONS].append(label_scores.view(-1))

    def _compute_grandparent_scores(self, states, parts):
        """
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
        word_embeddings = self._get_embeddings(instances, max_length, 'word')
        all_embeddings.append(word_embeddings)
        if self.upos_embeddings is not None:
            upos_embeddings = self._get_embeddings(instances,
                                                   max_length, 'upos')
            all_embeddings.append(upos_embeddings)

        if self.xpos_embeddings is not None:
            xpos_embeddings = self._get_embeddings(instances,
                                                   max_length, 'xpos')
            all_embeddings.append(xpos_embeddings)

        if self.char_rnn is not None:
            char_embeddings = self._run_char_rnn(instances, max_length)
            all_embeddings.append(char_embeddings)

        # each embedding tensor is (batch, num_tokens, embedding_size)
        embeddings = torch.cat(all_embeddings, dim=2)
        embeddings = self.dropout(embeddings)

        return embeddings

    def _get_embeddings(self, instances, max_length, word_or_tag):
        """
        Get the word or tag embeddings for all tokens in the instances.

        This function takes care of padding.

        :param word_or_tag: either 'word' or 'tag'
        :param max_length: length of the longest instance
        :return: a tensor with shape (batch, sequence, embedding size)
        """
        # padding is not supposed to be used in the end results
        index_matrix = torch.full((len(instances), max_length), 0,
                                  dtype=torch.long)
        for i, instance in enumerate(instances):
            if word_or_tag == 'word':
                indices = instance.get_all_embedding_ids()
            elif word_or_tag == 'upos':
                indices = instance.get_all_upos()
            else:
                indices = instance.get_all_xpos()

            index_matrix[i, :len(instance)] = torch.tensor(indices)

        if word_or_tag == 'word':
            embedding_matrix = self.word_embeddings
            dropout_rate = self.word_dropout_rate
            unknown_symbol = self.unknown_word
        else:
            dropout_rate = self.tag_dropout_rate
            if word_or_tag == 'upos':
                embedding_matrix = self.upos_embeddings
                unknown_symbol = self.unknown_upos
            else:
                embedding_matrix = self.xpos_embeddings
                unknown_symbol = self.unknown_xpos

        if self.training and dropout_rate:
            dropout_draw = torch.rand_like(index_matrix, dtype=torch.float)
            inds = dropout_draw < dropout_rate
            index_matrix[inds] = unknown_symbol

        if self.on_gpu:
            index_matrix = index_matrix.cuda()

        return embedding_matrix(index_matrix)

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
        char_indices = torch.full(shape, 0, dtype=torch.long)

        for i, instance in enumerate(instances):
            token_lengths[i, :len(instance)] = torch.tensor(token_lengths_[i])

            for j in range(len(instance)):
                # each j is a token
                chars = instance.get_characters(j)
                char_indices[i, j, :len(chars)] = torch.tensor(chars)

        # now we have a 3d matrix with char indices. let's reshape it to 2d,
        # stacking all tokens with no sentence separation
        new_shape = [batch_size * max_sentence_length, max_token_length]
        char_indices = char_indices.view(new_shape)
        lengths1d = token_lengths.view(-1)

        # now order by descending length and keep track of the originals
        sorted_lengths, sorted_inds = lengths1d.sort(descending=True)

        # we can't pass 0-length tensors to the LSTM (they're the padding)
        nonzero = sorted_lengths > 0
        sorted_lengths = sorted_lengths[nonzero]
        sorted_inds = sorted_inds[nonzero]

        sorted_token_inds = char_indices[sorted_inds]
        if self.on_gpu:
            sorted_token_inds = sorted_token_inds.cuda()

        # embedded is [batch * max_sentence_len, max_token_len, char_embedding]
        embedded = self.char_embeddings(sorted_token_inds)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths,
                                                   batch_first=True)
        outputs, (last_output, cell) = self.char_rnn(packed)

        # concatenate the last outputs of both directions
        last_output_bi = torch.cat([last_output[0], last_output[1]], dim=-1)
        num_words = batch_size * max_sentence_length
        shape = [num_words, 2 * self.char_rnn.hidden_size]
        char_representation = torch.zeros(shape)
        if self.on_gpu:
            char_representation = char_representation.cuda()
        char_representation[sorted_inds] = last_output_bi

        if self.training and self.word_dropout_rate:
            # sample a dropout mask and replace the dropped representations
            dropout_mask = torch.rand(num_words) < self.word_dropout_rate
            char_representation[dropout_mask] = self.char_dropout_replacement

        return char_representation.view([batch_size, max_sentence_length, -1])

    def forward(self, instances, parts):
        """
        :param instances: a list of DependencyInstance objects
        :param parts: a list of DependencyParts objects
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
        shared_states = self._packed_dropout(shared_states)

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

        # now go through each batch item
        for i in range(batch_size):
            length = lengths[i].item()
            states = parser_batch_states[i, :length]
            sent_parts = parts[i]

            self._compute_arc_scores(states, sent_parts)
            if self.model_type.consecutive_siblings:
                self._compute_consecutive_sibling_scores(states, sent_parts)

            if self.model_type.grandparents:
                self._compute_grandparent_scores(states, sent_parts)

            if self.model_type.grandsiblings:
                self._compute_grandsibling_scores(states, sent_parts)

        return self.scores
