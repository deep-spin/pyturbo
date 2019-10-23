import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from torch.distributions.gumbel import Gumbel
import numpy as np
import pickle
from joeynmt.embeddings import Embeddings as Seq2seqEmbeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.decoders import RecurrentDecoder
from joeynmt.search import greedy

from .token_dictionary import TokenDictionary, UNKNOWN
from .constants import Target, SPECIAL_SYMBOLS, PADDING, BOS, EOS, \
    ParsingObjective, structured_objectives
from ..classifier.lstm import CharLSTM, HighwayLSTM
from ..classifier.biaffine import DeepBiaffineScorer


gumbel = Gumbel(0, 1)


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


def get_padded_lemma_indices(instances, max_instance_length):
    """
    Create a tensor with lemma char indices.

    :param instances: list of instances
    :param max_instance_length: int, maximum number of tokens including root
    :return: tuple padded_lemmas, padded_lengths. Roots are not counted.
        padded_lemmas is a tensor shape (batch, num_words, num_chars)
        padded_lengths is (batch, num_words)
    """
    instances_lemmas = []
    lengths = []
    max_instance_length -= 1

    for instance in instances:
        # each item in lemma_characters is a numpy array with lemma chars
        # [1:] to skip root
        lemma_list = [torch.tensor(lemma)
                      for lemma in instance.lemma_characters[1:]]
        instance_lengths = torch.tensor([len(lemma) for lemma in lemma_list])

        # create empty tensors to match max instance length
        diff = max_instance_length - len(lemma_list)
        if diff:
            padding = diff * [torch.tensor([], dtype=torch.long)]
            lemma_list += padding

        instance_lemmas = rnn_utils.pad_sequence(lemma_list, batch_first=True)

        # we transpose num_tokens with token_length because it is easier to add
        # padding tokens than padding chars
        instances_lemmas.append(instance_lemmas.transpose(0, 1))
        lengths.append(instance_lengths)

    padded_transposed = rnn_utils.pad_sequence(instances_lemmas, True)
    padded_lemmas = padded_transposed.transpose(1, 2)
    padded_lengths = rnn_utils.pad_sequence(lengths, batch_first=True)

    return padded_lemmas, padded_lengths


def create_char_indices(instances, max_sentence_length):
    """
    Create a tensor with the character indices for all words in the instances.

    :param instances: a list of DependencyInstance objects
    :param max_sentence_length: int
    :return: a tuple (char_indices, token_lengths). Sentence length includes
        root.
        - char_indices is (batch, max_sentence_length, max_token_length).
        - token_lengths is (batch_size, max_sentence_length)
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

    return char_indices, token_lengths


class Lemmatizer(nn.Module):
    """
    Lemmatizer that uses a recurrent encoder-decoder framework.
    """
    def __init__(self, vocab_size, embedding_size, hidden_size, dropout_rate,
                 context_size, token_dictionary):
        """
        :param vocab_size: size of the char vocabulary
        :param embedding_size: size of the char embeddings
        :param hidden_size: hidden state of the encoder. Decoder is twice that.
        :param dropout_rate: dropout
        :param context_size: size of the context vector given as additional
            input
        :type token_dictionary: TokenDictionary
        """
        super(Lemmatizer, self).__init__()
        self.padding_idx = token_dictionary.get_character_id(PADDING)
        self.eos_idx = token_dictionary.get_character_id(EOS)
        self.bos_idx = token_dictionary.get_character_id(BOS)

        self.dropout = nn.Dropout(dropout_rate)
        self.embeddings = Seq2seqEmbeddings(
            embedding_size, vocab_size=vocab_size, padding_idx=self.padding_idx)
        self.encoder = RecurrentEncoder(
            hidden_size=hidden_size, emb_size=embedding_size,
            num_layers=2, dropout=dropout_rate, bidirectional=True)
        self.decoder = RecurrentDecoder(
            emb_size=embedding_size, hidden_size=2 * hidden_size,
            encoder=self.encoder, attention='luong', num_layers=2,
            vocab_size=vocab_size, dropout=dropout_rate, input_feeding=True)
        self.context_transform = nn.Linear(
            context_size, embedding_size, bias=False)

    def append_eos(self, chars, lengths):
        """
        Append an EOS token at the appropriate position after each sequence.

        The returned tensor will have the max length increased by one.

        :param chars: tensor (batch, max_length)
        :param lengths: tensor (batch)
        :return: tensor (batch, max_length + 1)
        """
        batch_size, max_length = chars.shape
        padding_column = self.padding_idx * torch.ones_like(chars[:, 0])
        extended = torch.cat([chars, padding_column.unsqueeze(1)], dim=1)

        # trick to get the last non-padding position
        extended[torch.arange(batch_size), lengths] = self.eos_idx

        return extended

    def prepend_bos(self, chars):
        """
        Prepend a BOS token at the beginning of the character sequences.

        The returned tensor will have the max length increased by one.

        :param chars: tensor (batch, max_length)
        :return: tensor (batch, max_length + 1)
        """
        bos_column = self.bos_idx * torch.ones_like(chars[:, 0])
        extended = torch.cat([bos_column.unsqueeze(1), chars], dim=1)

        return extended

    def forward(self, chars, context, token_lengths, gold_chars=None,
                gold_lengths=None):
        """
        :param chars: tensor (batch, max_sentence_length, max_token_length)
            with char ids
        :param context: tensor (batch, max_sentence_length, max_token_length.
            num_units) with contextual representation of each word
        :param token_lengths: tensor (batch, max_sentence_length) with length
            of each word
        :param gold_chars: only used in training. tensor
            (batch, max_sentence_length, max_lemma_length)
        :param gold_lengths: only used in training; length of each gold lemma.
            tensor (batch, max_sentence_length)
        :return:
            If training: tensor (batch, max_sentence_length, max_token_length,
            vocab_size) with logits for each character.
            At inference time: tensor (batch, max_sentence_length,
            unroll_steps + 1) with character indices and possibly EOS.
        """
        batch_size, max_sentence_length, max_token_length = chars.shape
        new_shape = [batch_size * max_sentence_length, max_token_length]
        chars = chars.reshape(new_shape)
        token_lengths1d = token_lengths.reshape(-1)

        # run only on non-padding tokens
        real_tokens = token_lengths1d > 0
        num_real_tokens = real_tokens.sum().item()
        token_lengths1d = token_lengths1d[real_tokens]

        # project contexts into (num_real_tokens, 1, num_units)
        projected_context = self.context_transform(self.dropout(context))
        projected_context = projected_context.view(
            batch_size * max_sentence_length, 1, -1)
        projected_context = projected_context[real_tokens]

        # (num_real_tokens, max_token_length, num_units)
        chars = chars[real_tokens]

        embedded_chars = self.embeddings(chars)

        # create a binary mask
        counts = torch.arange(max_token_length).view(1, 1, -1).to(chars.device)
        stacked_counts = counts.repeat(num_real_tokens, 1, 1)
        lengths3d = token_lengths1d.view(-1, 1, 1)
        mask = stacked_counts < lengths3d

        encoder_input = torch.cat([projected_context, embedded_chars], 1)
        encoder_output, encoder_state = self.encoder(
            encoder_input, token_lengths1d, mask)

        if gold_chars is None:
            # allow for short words with longer lemmas
            unroll_steps = max(5, int(1.5 * max_token_length))
            predictions, _ = greedy(
                mask, self.embeddings, self.bos_idx, unroll_steps,
                self.decoder, encoder_output, encoder_state)

            # predictions is a numpy array
            output = predictions.reshape([num_real_tokens, -1])

            real_tokens_np = real_tokens.cpu().numpy()
            shape = [batch_size * max_sentence_length, unroll_steps]
            padded_output = np.zeros(shape, np.int)
            padded_output[real_tokens_np] = output
            padded_output = padded_output.reshape(
                [batch_size, max_sentence_length, unroll_steps])
        else:
            gold_chars2d = gold_chars.reshape(
                batch_size * max_sentence_length, -1)
            gold_chars2d = gold_chars2d[real_tokens]
            gold_lengths1d = gold_lengths.view(-1)[real_tokens]
            gold_chars2d_eos = self.append_eos(gold_chars2d, gold_lengths1d)
            self.cached_gold_chars = gold_chars2d_eos
            self.cached_real_token_inds = real_tokens
            gold_chars2d = self.prepend_bos(gold_chars2d)
            embedded_target = self.embeddings(gold_chars2d)

            # unroll for the number of gold steps.
            unroll_steps = gold_chars2d.shape[-1]

            outputs = self.decoder(
                embedded_target, encoder_output, encoder_state,
                mask, unroll_steps)

            # outputs is a tuple (logits, state, att_distribution, att_sum)
            logits = outputs[0]

            # (batch, max_sentence_length, max_predicted_length, vocab_size)
            padded_output = torch.zeros(
                batch_size * max_sentence_length, unroll_steps,
                logits.shape[-1], device=chars.device)
            padded_output[real_tokens] = logits
            padded_output = padded_output.reshape(
                [batch_size, max_sentence_length, unroll_steps, -1])

        return padded_output


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
                 predict_lemma,
                 predict_tree,
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
        self.predict_lemma = predict_lemma
        self.predict_tree = predict_tree
        self.predict_tags = predict_upos or predict_xpos or \
            predict_morph or predict_lemma
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

        num_chars = token_dictionary.get_num_characters()
        if self.char_embedding_size:
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
        rnn_input_size += transform_size

        self.shared_rnn = HighwayLSTM(rnn_input_size, rnn_size, rnn_layers,
                                      dropout=self.dropout_rate)
        self.parser_rnn = HighwayLSTM(2 * rnn_size, rnn_size,
                                      dropout=self.dropout_rate)

        self.dropout_replacement = nn.Parameter(
            torch.randn(rnn_input_size) / np.sqrt(rnn_input_size))

        if self.predict_tags:
            self.tagger_rnn = HighwayLSTM(2 * rnn_size, rnn_size, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        # POS tags
        if predict_upos:
            self.upos_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=torch.relu)
            num_tags = token_dictionary.get_num_upos_tags()
            self.upos_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                   bias=True)
        if predict_xpos:
            self.xpos_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=torch.relu)
            num_tags = token_dictionary.get_num_xpos_tags()
            self.xpos_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                   bias=True)
        if predict_morph:
            self.morph_mlp = self._create_mlp(
                hidden_size=tag_mlp_size, num_layers=1,
                output_activation=torch.relu)
            num_tags = token_dictionary.get_num_morph_singletons()
            self.morph_scorer = self._create_scorer(tag_mlp_size, num_tags,
                                                    bias=True)
        if predict_lemma:
            self.lemmatizer = Lemmatizer(
                num_chars, char_embedding_size, char_hidden_size, dropout,
                2 * rnn_size, token_dictionary)

        if self.predict_tree:
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

            # Higher order layers
            if model_type.grandparents:
                self.gp_grandparent_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.gp_head_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.gp_modifier_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.gp_coeff = self._create_parameter_tensor(3, 1.)
                self.grandparent_scorer = self._create_scorer(self.ho_mlp_size)

            if model_type.consecutive_siblings:
                self.sib_head_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.sib_modifier_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.sib_sibling_mlp = self._create_mlp(
                    hidden_size=self.ho_mlp_size)
                self.sib_coeff = self._create_parameter_tensor(3, 1.)
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

    def _create_parameter_tensor(self, shape, value=None):
        """
        Create a tensor for representing some special token. It is included in
        the model parameters.
        """
        if value is None:
            tensor = torch.randn(shape) / np.sqrt(shape)
        else:
            tensor = torch.full(shape, value)
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
                layers.append(torch.relu)

            linear = nn.Linear(input_size, hidden_size)
            layers.extend([self.dropout, linear])
            input_size = hidden_size

        if output_activation is not None:
            layers.append(output_activation)

        mlp = nn.Sequential(*layers)
        return mlp

    def save(self, file):
        torch.save(self.state_dict(), file)

    def create_metadata(self):
        """
        Return a dictionary with metadata needed to reconstruct a serialized
        model.
        """
        vocab, dim = self.fixed_word_embeddings.weight.shape
        data = {'fixed_embedding_vocabulary': vocab,
                'fixed_embedding_size': dim}

        return data

    @classmethod
    def load(cls, torch_file, options, token_dictionary, metadata):
        fixed_embedding_vocab_size = metadata['fixed_embedding_vocabulary']
        fixed_embedding_size = metadata['fixed_embedding_size']
        trainable_embedding_size = options.embedding_size
        char_embedding_size = options.char_embedding_size
        tag_embedding_size = options.tag_embedding_size
        char_hidden_size = options.char_hidden_size
        transform_size = options.transform_size
        rnn_size = options.rnn_size
        arc_mlp_size = options.arc_mlp_size
        tag_mlp_size = options.tag_mlp_size
        label_mlp_size = options.label_mlp_size
        ho_mlp_size = options.ho_mlp_size
        rnn_layers = options.rnn_layers
        mlp_layers = options.mlp_layers
        dropout = options.dropout
        word_dropout = options.word_dropout
        predict_upos = options.upos
        predict_xpos = options.xpos
        predict_morph = options.morph
        predict_lemma = options.lemma
        predict_tree = options.parse
        model_type = options.model_type

        dummy_embeddings = np.empty([fixed_embedding_vocab_size,
                                     fixed_embedding_size], np.float32)
        model = DependencyNeuralModel(
            model_type, token_dictionary, dummy_embeddings,
            trainable_embedding_size,
            char_embedding_size,
            tag_embedding_size=tag_embedding_size,
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
            predict_morph=predict_morph, predict_lemma=predict_lemma,
            predict_tree=predict_tree)

        if model.on_gpu:
            state_dict = torch.load(torch_file)
        else:
            state_dict = torch.load(torch_file, map_location='cpu')

        # kind of a hack to allow compatibility with previous versions
        own_state_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if
                           k in own_state_dict}
        own_state_dict.update(pretrained_dict)
        model.load_state_dict(own_state_dict)

        return model

    def _compute_arc_scores(self, states, lengths, normalization):
        """
        Compute the first order scores and store them in the appropriate
        position in the `scores` tensor.

        The score matrices have shape (modifer, head), and do not have the row
        corresponding to the root as a modifier. Thus they have shape
        (num_words - 1, num_words).

        :param states: hidden states returned by the RNN; one for each word
        :param lengths: length of each sentence in the batch (including root)
        :param normalization: 'global' or 'local'
        """
        batch_size, max_sent_size, _ = states.size()

        # apply dropout separately to get different masks
        # head_scores is interpreted as (batch, modifier, head)
        head_scores = self.arc_scorer(self.dropout(states),
                                      self.dropout(states)).squeeze(3)
        s1 = self.dropout(states)
        s2 = self.dropout(states)
        label_scores = self.label_scorer(s1, s2)

        if normalization == ParsingObjective.LOCAL:
            # set arc scores from each word to itself as -inf
            # structured models don't need it as these arcs are never predicted
            diag = torch.eye(max_sent_size, device=states.device).bool()
            diag = diag.unsqueeze(0)
            head_scores.masked_fill_(diag, -np.inf)

            # set padding head scores to -inf
            # during training, label loss is computed with respect to the gold
            # arcs, so there's no need to set -inf scores to invalid positions
            # in label scores.
            padding_mask = create_padding_mask(lengths)
            head_scores = head_scores.masked_fill(padding_mask.unsqueeze(1),
                                                  -np.inf)

        if self.training and normalization == ParsingObjective.GLOBAL_MARGIN:
            dev = head_scores.device
            head_scores += gumbel.sample(head_scores.shape).to(dev)
            label_scores += gumbel.sample(label_scores.shape).to(dev)

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

        # we don't have H+M because those are already encoded in the arcs
        c = self.gp_coeff
        states_mg = c[0] * torch.tanh(modifiers + grandparents)
        states_hg = c[1] * torch.tanh(heads + grandparents)
        states_hmg = c[2] * torch.tanh(heads + modifiers + grandparents)
        final_states = states_mg + states_hg + states_hmg
        part_scores = self.grandparent_scorer(final_states)

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

        # we don't have H+M because those are already encoded in the arcs
        c = self.sib_coeff
        states_hs = c[0] * torch.tanh(heads + siblings)
        states_ms = c[1] * torch.tanh(modifiers + siblings)
        states_hms = c[2] * torch.tanh(heads + modifiers + siblings)
        final_states = states_hs + states_ms + states_hms

        sibling_scores = self.sibling_scorer(final_states)

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
        gsib_states = torch.tanh(heads + modifiers + siblings + grandparents)
        gsib_scores = self.grandsibling_scorer(gsib_states)

        self.scores[Target.GRANDSIBLINGS].append(gsib_scores.view(-1))

    def get_word_representations(self, instances, max_length, char_indices,
                                 token_lengths):
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
            char_embeddings = self.char_rnn(char_indices, token_lengths)
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
        elif type_ == 'trainableword':
            embedding_matrix = self.trainable_word_embeddings
        elif type_ == 'lemma':
            embedding_matrix = self.lemma_embeddings
        else:
            if type_ == 'upos':
                embedding_matrix = self.upos_embeddings
            else:
                embedding_matrix = self.xpos_embeddings

        if self.on_gpu:
            index_matrix = index_matrix.cuda()

        embeddings = embedding_matrix(index_matrix)
        return embeddings

    def _convert_arc_scores_to_parts(self, instances, parts):
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
            mask = torch.tensor(mask.astype(np.bool))
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
                    inst_parts.gold_parts[:inst_parts.num_arcs],
                    device=head_scores.device)

                offset = inst_parts.offsets[Target.RELATIONS]
                num_labeled = inst_parts.num_labeled_arcs
                gold_label_parts = torch.tensor(
                    inst_parts.gold_parts[offset:offset + num_labeled],
                    device=head_scores.device)
                head_scores1d = head_scores1d - gold_arc_parts
                label_scores1d = label_scores1d - gold_label_parts

            new_head_scores.append(head_scores1d)
            new_label_scores.append(label_scores1d)

        self.scores[Target.HEADS] = new_head_scores
        self.scores[Target.RELATIONS] = new_label_scores

    def forward(self, instances, parts, normalization=ParsingObjective.LOCAL):
        """
        :param instances: a list of DependencyInstance objects
        :param parts: a list of DependencyParts objects
        :param normalization: a ParsingObjective value indicating "local",
            "global-margin" or "global-prob". It only affects
            first order parts (arcs and labeled arcs).

            If "local", the losses for each word (as a modifier) is computed
            independently. The model will store a tensor with all arc scores
            (including padding) for efficient loss computation.

            If "global-margin", the loss is a hinge margin over the global
            structure.

            If "global-prob", the loss is the cross-entropy of the probability
            of the global structure.

            In the two latter cases, the model stores scores as a list of 1d
            arrays (without padding) that can easily be used with AD3 decoding
            functions.

        :return: a dictionary mapping each target to score tensors
        """
        self.scores = {}
        for type_ in parts[0].part_lists:
            self.scores[type_] = []

        batch_size = len(instances)
        lengths = torch.tensor([len(instance) for instance in instances],
                               dtype=torch.long)
        if self.on_gpu:
            lengths = lengths.cuda()

        # packed sequences must be sorted by decreasing length
        sorted_lengths, inds = lengths.sort(descending=True)

        # rev_inds are used to unsort the sorted sentences back
        _, rev_inds = inds.sort()
        if self.on_gpu:
            sorted_lengths = sorted_lengths.cuda()

        max_length = sorted_lengths[0].item()

        # compute char inds only once
        if self.char_rnn or self.predict_lemma:
            char_indices, token_lengths = create_char_indices(
                instances, max_length)
            char_indices = char_indices.to(lengths.device)
            token_lengths = token_lengths.to(lengths.device)
        else:
            char_indices, token_lengths = None, None

        embeddings = self.get_word_representations(
            instances, max_length, char_indices, token_lengths)
        sorted_embeddings = embeddings[inds]

        # pack to account for variable lengths
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            sorted_embeddings, sorted_lengths, batch_first=True)

        shared_states, _ = self.shared_rnn(packed_embeddings)

        if self.predict_tags or self.predict_lemma:
            if self.predict_tree:
                # If we don't create a copy, we get an error for variable
                # rewrite on gradient computation.
                padded, lens = rnn_utils.pad_packed_sequence(
                    shared_states, batch_first=True)
                packed = rnn_utils.pack_padded_sequence(padded, lens, True)
            else:
                packed = shared_states

            tagger_packed_states, _ = self.tagger_rnn(packed)
            tagger_batch_states, _ = nn.utils.rnn.pad_packed_sequence(
                tagger_packed_states, batch_first=True)
            tagger_batch_states = tagger_batch_states[rev_inds]

            # ignore root
            tagger_batch_states = tagger_batch_states[:, 1:]

            if self.predict_upos:
                hidden = self.upos_mlp(tagger_batch_states)
                self.scores[Target.UPOS] = self.upos_scorer(hidden)

            if self.predict_xpos:
                hidden = self.xpos_mlp(tagger_batch_states)
                self.scores[Target.XPOS] = self.xpos_scorer(hidden)

            if self.predict_morph:
                hidden = self.morph_mlp(tagger_batch_states)
                self.scores[Target.MORPH] = self.morph_scorer(hidden)

            if self.predict_lemma:
                if self.training:
                    lemmas, lemma_lengths = get_padded_lemma_indices(
                        instances, max_length)
                    lemmas = lemmas.to(lengths.device)
                    lemma_lengths = lemma_lengths.to(lengths.device)
                else:
                    lemmas, lemma_lengths = None, None

                # skip root
                logits = self.lemmatizer(
                    char_indices[:, 1:], tagger_batch_states,
                    token_lengths[:, 1:], lemmas, lemma_lengths)
                self.scores[Target.LEMMA] = logits

        if self.predict_tree:
            parser_packed_states, _ = self.parser_rnn(shared_states)

            # batch_states is (batch, num_tokens, hidden_size)
            parser_batch_states, _ = nn.utils.rnn.pad_packed_sequence(
                parser_packed_states, batch_first=True)

            # return to the original ordering
            parser_batch_states = parser_batch_states[rev_inds]

            self._compute_arc_scores(
                parser_batch_states, lengths, normalization)

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

            if normalization == ParsingObjective.GLOBAL_MARGIN:
                self._convert_arc_scores_to_parts(instances, parts)

        return self.scores
