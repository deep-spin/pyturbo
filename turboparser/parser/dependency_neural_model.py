import torch
import torch.nn as nn
from .dependency_parts import Arc, DependencyParts, NextSibling, Grandparent, \
    GrandSibling, LabeledArc
from .token_dictionary import TokenDictionary, UNKNOWN, PADDING
import numpy as np

#TODO: maybe this should be elsewhere?
# special pseudo-tokens to index embeddings
# root is not one of these since it is a token in the sentences
NULL_SIBLING = '_NULL_SIBLING_'
special_tokens = [NULL_SIBLING]


class DependencyNeuralModel(nn.Module):
    def __init__(self,
                 model_type,
                 token_dictionary,
                 dependency_dictionary,
                 word_embeddings,
                 char_embedding_size,
                 tag_embedding_size,
                 distance_embedding_size,
                 rnn_size,
                 mlp_size,
                 rnn_layers,
                 mlp_layers,
                 dropout,
                 word_dropout,
                 tag_dropout):
        """
        :param model_type: a ModelType object
        :param token_dictionary: TokenDictionary object
        :type token_dictionary: TokenDictionary
        :param dependency_dictionary: DependencyDictionary object
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
        self.mlp_size = mlp_size
        self.rnn_layers = rnn_layers
        self.mlp_layers = mlp_layers
        self.dropout_rate = dropout
        self.word_dropout_rate = word_dropout
        self.tag_dropout_rate = tag_dropout
        self.num_labels = len(dependency_dictionary.relation_alphabet)
        self.padding_word = token_dictionary.get_embedding_id(PADDING)
        self.padding_tag = token_dictionary.get_tag_id(PADDING)
        self.unknown_word = token_dictionary.get_embedding_id(UNKNOWN)
        self.unknown_tag = token_dictionary.get_tag_id(UNKNOWN)
        self.on_gpu = torch.cuda.is_available()

        # self.word_embeddings = nn.Embedding(token_dictionary.get_num_forms(),
        #                                     word_embedding_size)
        word_embeddings = torch.tensor(word_embeddings, dtype=torch.float32)
        self.word_embeddings = nn.Embedding.from_pretrained(word_embeddings,
                                                            freeze=False)

        if self.char_embedding_size:
            char_vocab = token_dictionary.get_num_characters()
            self.char_embeddings = nn.Embedding(char_vocab, char_embedding_size)

            self.char_rnn = nn.LSTM(
                input_size=char_embedding_size, hidden_size=char_embedding_size,
                bidirectional=True, batch_first=True)
            char_based_embedding_size = char_embedding_size
        else:
            self.char_embeddings = None
            self.char_rnn = None
            char_based_embedding_size = 0

        if self.tag_embedding_size:
            self.tag_embeddings = nn.Embedding(token_dictionary.get_num_tags(),
                                               tag_embedding_size)
        else:
            self.tag_embeddings = None

        if self.distance_embedding_size:
            self.distance_bins = np.array(
                list(range(10)) + list(range(10, 40, 5)) + [40])
            self.distance_embeddings = nn.Embedding(len(self.distance_bins) * 2,
                                                    distance_embedding_size)
        else:
            self.distance_bins = None
            self.distance_embeddings = None

        input_size = self.word_embedding_size + tag_embedding_size + \
            (2 * char_based_embedding_size)
        self.rnn = nn.LSTM(
            input_size=input_size,
            hidden_size=rnn_size,
            num_layers=rnn_layers,
            dropout=dropout,
            bidirectional=True,
            batch_first=True)
        self.rnn_hidden_size = 2 * rnn_size
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # first order
        self.head_mlp = self._create_mlp()
        self.modifier_mlp = self._create_mlp()
        self.arc_scorer = self._create_scorer()
        self.label_scorer = self._create_scorer(output_size=self.num_labels)

        if model_type.grandparents:
            self.gp_grandparent_mlp = self._create_mlp()
            self.gp_head_mlp = self._create_mlp()
            self.gp_modifier_mlp = self._create_mlp()
            self.grandparent_scorer = self._create_scorer()

        if model_type.consecutive_siblings:
            self.sib_head_mlp = self._create_mlp()
            self.sib_modifier_mlp = self._create_mlp()
            self.sib_sibling_mlp = self._create_mlp()
            self.sibling_scorer = self._create_scorer()

        if model_type.consecutive_siblings or model_type.grandsiblings \
                or model_type.trisiblings or model_type.arbitrary_siblings:
            self.null_sibling_tensor = self._create_parameter_tensor()

        if model_type.grandsiblings:
            self.gsib_head_mlp = self._create_mlp()
            self.gsib_modifier_mlp = self._create_mlp()
            self.gsib_sibling_mlp = self._create_mlp()
            self.gsib_grandparent_mlp = self._create_mlp()
            self.grandsibling_scorer = self._create_scorer()

        if self.distance_embedding_size:
            self.distance_projector = nn.Linear(
                distance_embedding_size,
                mlp_size,
                bias=True)
        else:
            self.distance_mlp = None

        # Clear out the gradients before the next batch.
        self.zero_grad()

    def _create_parameter_tensor(self, shape=None):
        """
        Create a tensor for representing some special token. It is included in
        the model parameters.

        If shape is None, it will have shape equal to hidden_size.
        """
        if shape is None:
            shape = self.rnn_hidden_size

        tensor = torch.randn(shape, requires_grad=True)
        if self.on_gpu:
            tensor = tensor.cuda()
        
        parameter = nn.Parameter(tensor)

        return parameter

    def _create_scorer(self, input_size=None, output_size=1):
        """
        Create the weights for scoring a given tensor representation to a
        single number.

        :param input_size: expected input size. If None, self.hidden_units
            is used.
        :return: an nn.Linear object
        """
        if input_size is None:
            input_size = self.mlp_size
        linear = nn.Linear(input_size, output_size, bias=False)
        scorer = nn.Sequential(self.dropout, linear)

        return scorer

    def _create_mlp(self, input_size=None, hidden_size=None, num_layers=None):
        """
        Create the weights for a fully connected subnetwork.

        The output has a linear activation; if num_layers > 1, hidden layers
        will use a non-linearity.

        The first layer will have a weight matrix (input x hidden), subsequent
        layers will be (hidden x hidden).

        :param input_size: if not given, will be assumed rnn_size * 2
        :param hidden_size: if not given, will be assumed mlp_size
        :param num_layers: number of hidden layers (including the last one). If
            not given, will be mlp_layers
        :return: an nn.Linear object, mapping an input with 2*hidden_units
            to hidden_units.
        """
        if input_size is None:
            input_size = self.rnn_size * 2
        if hidden_size is None:
            hidden_size = self.mlp_size
        if num_layers is None:
            num_layers = self.mlp_layers

        layers = []
        for i in range(num_layers):
            if i > 0:
                layers.append(self.relu)

            linear = nn.Linear(input_size, hidden_size)
            layers.extend([self.dropout, linear])
            input_size = hidden_size

        mlp = nn.Sequential(*layers)
        return mlp

    def save(self, file):
        torch.save(self.state_dict(), file)

    def load(self, file):
        if self.on_gpu:
            state_dict = torch.load(file)
        else:
            state_dict = torch.load(file, map_location='cpu')

        # kind of a hack to allow compatibility with previous versions
        own_state_dict = self.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           k in own_state_dict}
        own_state_dict.update(pretrained_dict)
        self.load_state_dict(own_state_dict)

    def _compute_arc_scores(self, states, parts, scores):
        """
        Compute the first order scores and store them in the appropriate
        position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to first order features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.head_mlp(states)
        modifier_tensors = self.modifier_mlp(states)
        offset, num_arcs = parts.get_offset(Arc)

        head_indices = []
        modifier_indices = []
        distance_indices = []

        for part in parts.iterate_over_type(Arc):
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            if self.distance_embedding_size:
                if part.modifier > part.head:
                    dist = part.modifier - part.head
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                else:
                    dist = part.head - part.modifier
                    dist = np.nonzero(dist >= self.distance_bins)[0][-1]
                    dist += len(self.distance_bins)

                distance_indices.append(dist)

        # now index all of them to process at once
        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        if self.distance_embedding_size:
            distance_indices = torch.tensor(distance_indices, dtype=torch.long)
            if self.on_gpu:
                distance_indices = distance_indices.cuda()
            
            distances = self.distance_embeddings(distance_indices)
            distance_projections = self.distance_projector(distances)
            distance_projections = distance_projections.view(
                -1, self.mlp_size)

        else:
            distance_projections = 0

        arc_states = self.tanh(heads + modifiers + distance_projections)
        arc_scores = self.arc_scorer(arc_states)
        scores[offset:offset + num_arcs] = arc_scores.view(-1)

        if not parts.has_type(LabeledArc):
            return

        offset_labeled, num_labeled = parts.get_offset(LabeledArc)
        label_scores = self.label_scorer(arc_states)
        indices = []

        # place the label scores in the correct position in the output
        for i, arc in enumerate(parts.iterate_over_type(Arc)):
            labels = parts.find_arc_labels(arc.head, arc.modifier)
            for label in labels:
                # first store all index pairs in a list to run only one
                # indexing operation
                indices.append((i, label))

        used_label_scores = label_scores[list(zip(*indices))]
        scores[offset_labeled:offset_labeled + num_labeled] = used_label_scores

    def _compute_grandparent_scores(self, states, parts, scores):
        """
        Compute the grandparent scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
        """
        head_tensors = self.gp_head_mlp(states)
        grandparent_tensors = self.gp_grandparent_mlp(states)
        modifier_tensors = self.gp_modifier_mlp(states)

        head_indices = []
        modifier_indices = []
        grandparent_indices = []

        for part in parts.iterate_over_type(Grandparent):
            # list all indices, then feed the corresponding tensors to the net
            head_indices.append(part.head)
            modifier_indices.append(part.modifier)
            grandparent_indices.append(part.grandparent)

        heads = head_tensors[head_indices]
        modifiers = modifier_tensors[modifier_indices]
        grandparents = grandparent_tensors[grandparent_indices]
        part_states = self.tanh(heads + modifiers + grandparents)
        part_scores = self.grandparent_scorer(part_states)

        offset, size = parts.get_offset(Grandparent)
        scores[offset:offset + size] = part_scores.view(-1)

    def _compute_consecutive_sibling_scores(self, states, parts, scores):
        """
        Compute the consecutive sibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
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

        for part in parts.iterate_over_type(NextSibling):
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

        offset, size = parts.get_offset(NextSibling)
        scores[offset:offset + size] = sibling_scores.view(-1)

    def _compute_grandsibling_scores(self, states, parts, scores):
        """
        Compute the consecutive grandsibling scores and store them in the
        appropriate position in the `scores` tensor.

        :param states: hidden states returned by the RNN; one for each word
        :param parts: a DependencyParts object containing the parts to be scored
        :type parts: DependencyParts
        :param scores: tensor for storing the scores for each part. The
            positions relative to the features are indexed by the parts
            object. It is modified in-place.
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

        for part in parts.iterate_over_type(GrandSibling):
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

        offset, size = parts.get_offset(GrandSibling)
        scores[offset:offset + size] = gsib_scores.view(-1)

    def get_word_representation(self, instances, max_length):
        """
        Get the full embedding representation of a word, including word type
        embeddings, char level and POS tag embeddings.
        """
        embeddings = self._get_embeddings(instances, max_length, 'word')
        if self.tag_embeddings is not None:
            tag_embeddings = self._get_embeddings(instances, max_length, 'tag')

            # each embedding tensor is (batch, num_tokens, embedding_size)
            embeddings = torch.cat([embeddings, tag_embeddings], dim=2)

        if self.char_rnn is not None:
            char_embeddings = self._run_char_rnn(instances, max_length)
            embeddings = torch.cat([embeddings, char_embeddings], dim=2)

        return embeddings

    def _get_embeddings(self, instances, max_length, word_or_tag):
        """
        Get the word or tag embeddings for all tokens in the instances.

        This function takes care of padding.

        :param word_or_tag: either 'word' or 'tag'
        :return: a tensor with shape (batch, sequence, embedding size)
        """
        # padding is not supposed to be used in the end results
        index_matrix = torch.full((len(instances), max_length), 0,
                                  dtype=torch.long)
        for i, instance in enumerate(instances):
            if word_or_tag == 'word':
                getter = instance.get_embedding_id
            else:
                getter = instance.get_tag

            indices = [getter(j) for j in range(len(instance))]
            index_matrix[i, :len(instance)] = torch.tensor(indices)

        if word_or_tag == 'word':
            embedding_matrix = self.word_embeddings
            dropout_rate = self.word_dropout_rate
        else:
            embedding_matrix = self.tag_embeddings
            dropout_rate = self.tag_dropout_rate

        if self.training and dropout_rate:
            dropout_draw = torch.rand_like(index_matrix, dtype=torch.float)
            inds = dropout_draw < dropout_rate
            unknown_symbol = self.unknown_word if word_or_tag == 'word' \
                else self.unknown_tag
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
        token_lengths_ = [[len(inst.get_characters(i))
                           for i in range(len(inst))]
                          for inst in instances]
        max_token_length = max(max(inst_lengths)
                               for inst_lengths in token_lengths_)

        shape = [len(instances), max_sentence_length]
        token_lengths = torch.zeros(shape, dtype=torch.long)

        shape = [len(instances), max_sentence_length, max_token_length]
        token_indices = torch.full(shape, 0, dtype=torch.long)

        for i, instance in enumerate(instances):
            token_lengths[i, :len(instance)] = torch.tensor(token_lengths_[i])

            for j in range(len(instance)):
                # each j is a token
                chars = instance.get_characters(j)
                token_indices[i, j, :len(chars)] = torch.tensor(chars)

        # now we have a 3d matrix with token indices. let's reshape it to 2d,
        # stacking all tokens with no sentence separation
        new_shape = [len(instances) * max_sentence_length, max_token_length]
        token_indices = token_indices.view(new_shape)
        lengths1d = token_lengths.view(-1)

        # now order by descending length and keep track of the originals
        sorted_lengths, sorted_inds = lengths1d.sort(descending=True)

        # we can't pass 0-length tensors to the LSTM
        nonzero = sorted_lengths > 0
        sorted_lengths = sorted_lengths[nonzero]
        sorted_inds = sorted_inds[nonzero]

        sorted_token_inds = token_indices[sorted_inds]
        if self.on_gpu:
            sorted_token_inds = sorted_token_inds.cuda()

        # embedded is [batch * max_sentence_len, max_token_len, char_embedding]
        embedded = self.char_embeddings(sorted_token_inds)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths,
                                                   batch_first=True)
        outputs, (last_output, cell) = self.char_rnn(packed)

        # concatenate the last outputs of both directions
        last_output_bi = torch.cat([last_output[0], last_output[1]], dim=-1)
        shape = [len(instances) * max_sentence_length,
                 2 * self.char_rnn.hidden_size]
        char_representation = torch.zeros(shape)
        if self.on_gpu:
            char_representation = char_representation.cuda()
        char_representation[sorted_inds] = last_output_bi

        return char_representation.view([len(instances), max_sentence_length,
                                         -1])

    def forward(self, instances, parts):
        """
        :param instances: a list of DependencyInstance objects
        :param parts: a list of DependencyParts objects
        :return: a score matrix with shape (num_instances, longest_length)
        """
        batch_size = len(instances)
        lengths = torch.tensor([len(instance) for instance in instances],
                               dtype=torch.long)
        # packed sequences must be sorted by decreasing length
        lengths, inds = lengths.sort(descending=True)
        if self.on_gpu:
            lengths = lengths.cuda()

        # instances = [instances[i] for i in inds]
        max_length = lengths[0].item()
        max_num_parts = max(len(p) for p in parts)
        batch_scores = torch.zeros(batch_size, max_num_parts)
        embeddings = self.get_word_representation(instances, max_length)
        sorted_embeddings = embeddings[inds]

        # pack to account for variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            sorted_embeddings, lengths, batch_first=True)

        # batch_states is (batch, num_tokens, hidden_size)
        batch_packed_states, _ = self.rnn(packed_embeddings)
        batch_states, _ = nn.utils.rnn.pad_packed_sequence(
            batch_packed_states, batch_first=True)

        # now go through each batch item
        for i in range(batch_size):
            # i points to positions in the original instances and parts
            # inds[i] points to the corresponding position in the RNN output
            rnn_ind = inds[i]

            # we will set the scores corresponding to the i-th item in the
            # batch, or the rnn_ind-th item in the instances

            length = lengths[i].item()
            states = batch_states[i, :length]
            scores = batch_scores[rnn_ind]
            sent_parts = parts[rnn_ind]

            self._compute_arc_scores(states, sent_parts, scores)

            if sent_parts.has_type(NextSibling):
                self._compute_consecutive_sibling_scores(states, sent_parts,
                                                         scores)

            if sent_parts.has_type(Grandparent):
                self._compute_grandparent_scores(states, sent_parts, scores)

            if sent_parts.has_type(GrandSibling):
                self._compute_grandsibling_scores(states, sent_parts, scores)

        return batch_scores
