import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils import rnn as rnn_utils
from torch.distributions.gumbel import Gumbel
import numpy as np
from transformers import BertModel, BertConfig
from joeynmt.embeddings import Embeddings as Seq2seqEmbeddings
from joeynmt.encoders import RecurrentEncoder
from joeynmt.decoders import RecurrentDecoder
from joeynmt.search import greedy

from .token_dictionary import TokenDictionary, UNKNOWN
from .constants import Target, SPECIAL_SYMBOLS, PADDING, BOS, EOS, \
    ParsingObjective
from ..classifier.lstm import CharLSTM, HighwayLSTM, LSTM
from ..classifier.biaffine import DeepBiaffineScorer


gumbel = Gumbel(0, 1)
encoder_dim = 768
max_encoder_length = 512


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
                 trainable_word_embedding_size=75,
                 lemma_embedding_size=0,
                 char_embedding_size=250,
                 char_hidden_size=400,
                 transform_size=125,
                 rnn_size=400,
                 shared_rnn_layers=2,
                 tag_embedding_size=0,
                 arc_mlp_size=400,
                 label_mlp_size=400,
                 ho_mlp_size=200,
                 dropout=0.5,
                 word_dropout=0.33,
                 predict_upos=True,
                 predict_xpos=True,
                 predict_morph=True,
                 predict_lemma=False,
                 predict_tree=True,
                 tag_mlp_size=0,
                 pretrained_name_or_config=None):
        """
        :param model_type: a ModelType object
        :param token_dictionary: TokenDictionary object
        :type token_dictionary: TokenDictionary
        :param fixed_word_embeddings: numpy or torch embedding matrix
            (kept fixed), or None
        :param word_dropout: probability of replacing a word with the unknown
            token
        :param pretrained_name_or_config: None, a string (with the pretrained
            BERT model to be used) or a BertConfig instance when loading a pre
            trained parser. If None, no BERT will be used.
        """
        super(DependencyNeuralModel, self).__init__()
        self.char_embedding_size = char_embedding_size
        self.char_hidden_size = char_hidden_size
        self.tag_embedding_size = tag_embedding_size
        self.transform_size = transform_size
        self.arc_mlp_size = arc_mlp_size
        self.tag_mlp_size = tag_mlp_size
        self.ho_mlp_size = ho_mlp_size
        self.label_mlp_size = label_mlp_size
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
        self.rnn_size = rnn_size

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

        total_encoded_dim = 0

        if trainable_word_embedding_size:
            num_words = token_dictionary.get_num_forms()
            self.trainable_word_embeddings = nn.Embedding(
                num_words, trainable_word_embedding_size)
            total_encoded_dim += trainable_word_embedding_size
        else:
            self.trainable_word_embeddings = None

        if lemma_embedding_size:
            num_lemmas = token_dictionary.get_num_lemmas()
            self.lemma_embeddings = nn.Embedding(
                num_lemmas, lemma_embedding_size)
            total_encoded_dim += lemma_embedding_size
        else:
            self.lemma_embeddings = None

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
                total_encoded_dim += tag_embedding_size
            self.morph_embeddings = nn.ModuleList()
            for feature_name in morph_alphabets:
                alphabet = morph_alphabets[feature_name]
                embeddings = nn.Embedding(len(alphabet), tag_embedding_size)
                self.morph_embeddings.append(embeddings)
            total_encoded_dim += tag_embedding_size
        else:
            self.upos_embeddings = None
            self.xpos_embeddings = None
            self.morph_embeddings = None

        num_chars = token_dictionary.get_num_characters()
        if self.char_embedding_size:
            self.char_rnn = CharLSTM(
                num_chars, char_embedding_size, char_hidden_size,
                dropout=dropout, bidirectional=False)

            if self.transform_size > 0:
                self.char_projection = nn.Linear(
                    char_hidden_size, transform_size, bias=False)
                total_encoded_dim += transform_size
            else:
                total_encoded_dim += char_hidden_size
        else:
            self.char_rnn = None

        if fixed_word_embeddings is None:
            self.fixed_word_embeddings = None
        else:
            fixed_word_embeddings = torch.tensor(fixed_word_embeddings,
                                                 dtype=torch.float)
            self.fixed_word_embeddings = nn.Embedding.from_pretrained(
                fixed_word_embeddings, freeze=True)
            if self.transform_size > 0:
                self.fixed_embedding_projection = nn.Linear(
                    fixed_word_embeddings.shape[1], transform_size, bias=False)
                total_encoded_dim += transform_size
            else:
                total_encoded_dim += fixed_word_embeddings.shape[1]

        if pretrained_name_or_config is None:
            self.encoder = None
        elif isinstance(pretrained_name_or_config, BertConfig):
            self.encoder = BertModel(pretrained_name_or_config)
            total_encoded_dim += encoder_dim
        else:
            self.encoder = BertModel.from_pretrained(
                pretrained_name_or_config, output_hidden_states=True)
            total_encoded_dim += encoder_dim

        self.dropout_replacement = nn.Parameter(
            torch.randn(total_encoded_dim) / np.sqrt(total_encoded_dim))
        self.dropout = nn.Dropout(dropout)
        self.total_encoded_dim = total_encoded_dim

        if shared_rnn_layers > 0 and rnn_size > 0:
            self.shared_rnn = HighwayLSTM(
                total_encoded_dim, rnn_size, shared_rnn_layers,
                self.dropout_rate)
            hidden_dim = 2 * self.rnn_size
        else:
            self.shared_rnn = None
            hidden_dim = total_encoded_dim

        # POS and morphology tags
        if self.predict_tags:
            if self.rnn_size > 0:
                self.tagger_rnn = LSTM(
                    hidden_dim, self.rnn_size, bidirectional=True)
                tagger_dim = 2 * self.rnn_size
            else:
                self.tagger_rnn = None
                tagger_dim = total_encoded_dim

            scorer_dim = tag_mlp_size if tag_mlp_size > 0 else tagger_dim

            if predict_upos:
                if tag_mlp_size > 0:
                    self.upos_mlp = self._create_mlp(
                        tagger_dim, tag_mlp_size, num_layers=1,
                        output_activation=nn.ReLU())
                num_tags = token_dictionary.get_num_upos_tags()
                self.upos_scorer = self._create_scorer(scorer_dim, num_tags,
                                                       bias=True)
            if predict_xpos:
                if tag_mlp_size > 0:
                    self.xpos_mlp = self._create_mlp(
                        tagger_dim, tag_mlp_size, num_layers=1,
                        output_activation=nn.ReLU())
                num_tags = token_dictionary.get_num_xpos_tags()
                self.xpos_scorer = self._create_scorer(scorer_dim, num_tags,
                                                       bias=True)
            if predict_morph:
                if tag_mlp_size > 0:
                    self.morph_mlp = self._create_mlp(
                        tagger_dim, tag_mlp_size, num_layers=1,
                        output_activation=nn.ReLU())
                num_tags = token_dictionary.get_num_morph_singletons()
                self.morph_scorer = self._create_scorer(scorer_dim, num_tags,
                                                        bias=True)
            if predict_lemma:
                self.lemmatizer = Lemmatizer(
                    num_chars, char_embedding_size, char_hidden_size, dropout,
                    tagger_dim, token_dictionary)

        if self.predict_tree:
            if self.rnn_size > 0:
                self.parser_rnn = LSTM(
                    hidden_dim, self.rnn_size, bidirectional=True)
                parser_dim = 2 * self.rnn_size
            else:
                self.parser_rnn = None
                parser_dim = hidden_dim

            # first order layers
            num_labels = token_dictionary.get_num_deprels()
            self.arc_scorer = DeepBiaffineScorer(
                parser_dim, parser_dim, arc_mlp_size, 1, dropout=dropout)
            self.label_scorer = DeepBiaffineScorer(
                parser_dim, parser_dim, label_mlp_size,
                num_labels, dropout=dropout)
            self.linearization_scorer = DeepBiaffineScorer(
                parser_dim, parser_dim, arc_mlp_size, 1, dropout=dropout)
            self.distance_scorer = DeepBiaffineScorer(
                parser_dim, parser_dim, arc_mlp_size, 1, dropout=dropout)

            # Higher order layers
            if model_type.grandparents:
                self.gp_grandparent_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gp_head_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gp_modifier_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gp_coeff = self._create_parameter_tensor([3], 1.)
                self.grandparent_scorer = self._create_scorer(self.ho_mlp_size)

            if model_type.consecutive_siblings:
                self.sib_head_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.sib_modifier_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.sib_sibling_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.sib_coeff = self._create_parameter_tensor([3], 1.)
                self.sibling_scorer = self._create_scorer(self.ho_mlp_size)

            if model_type.consecutive_siblings or model_type.grandsiblings \
                    or model_type.trisiblings or model_type.arbitrary_siblings:
                self.null_sibling_tensor = self._create_parameter_tensor(
                    parser_dim)

            if model_type.grandsiblings:
                self.gsib_head_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gsib_modifier_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gsib_sibling_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gsib_grandparent_mlp = self._create_mlp(
                    parser_dim, self.ho_mlp_size)
                self.gsib_coeff = self._create_parameter_tensor([3], 1.)
                self.grandsibling_scorer = self._create_scorer(self.ho_mlp_size)

        # Clear out the gradients before the next batch.
        self.zero_grad()

    def extra_repr(self) -> str:
        dim = self.dropout_replacement.shape[0]
        return '(dropout_replacement): Tensor(%d)' % dim

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

    def _create_mlp(self, input_size=None, hidden_size=None, num_layers=1,
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
            input_size = self.total_encoded_dim
        if hidden_size is None:
            hidden_size = self.arc_mlp_size

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

    def create_metadata(self) -> dict:
        """
        Return a dictionary with metadata needed to reconstruct a serialized
        model.
        """
        if self.fixed_word_embeddings is None:
            vocab, dim = 0, 0
        else:
            vocab, dim = self.fixed_word_embeddings.weight.shape
        if self.encoder is None:
            data = {}
        else:
            bert_config = self.encoder.config
            data = bert_config.to_dict()

        data['fixed_embedding_vocabulary'] = vocab
        data['fixed_embedding_size'] = dim

        return data

    @classmethod
    def load(cls, torch_file, options, token_dictionary, metadata):
        fixed_embedding_vocab_size = metadata['fixed_embedding_vocabulary']
        fixed_embedding_size = metadata['fixed_embedding_size']
        lemma_embedding_size = options.lemma_embedding_size
        char_embedding_size = options.char_embedding_size
        trainable_embedding_size = options.embedding_size
        tag_embedding_size = options.tag_embedding_size
        char_hidden_size = options.char_hidden_size
        transform_size = options.transform_size
        rnn_size = options.rnn_size
        shared_layers = options.rnn_layers
        arc_mlp_size = options.arc_mlp_size
        tag_mlp_size = options.tag_mlp_size
        label_mlp_size = options.label_mlp_size
        ho_mlp_size = options.ho_mlp_size
        dropout = options.dropout
        word_dropout = options.word_dropout
        predict_upos = options.upos
        predict_xpos = options.xpos
        predict_morph = options.morph
        predict_lemma = options.lemma
        predict_tree = options.parse
        model_type = options.model_type

        if fixed_embedding_vocab_size > 0:
            dummy_embeddings = np.empty([fixed_embedding_vocab_size,
                                         fixed_embedding_size], np.float32)
        else:
            dummy_embeddings = None

        if options.bert_model is None:
            config = None
        else:
            config = BertConfig.from_dict(metadata)

        model = DependencyNeuralModel(
            model_type, token_dictionary, dummy_embeddings,
            trainable_embedding_size,
            lemma_embedding_size,
            char_embedding_size,
            tag_embedding_size=tag_embedding_size,
            char_hidden_size=char_hidden_size,
            transform_size=transform_size,
            rnn_size=rnn_size,
            shared_rnn_layers=shared_layers,
            arc_mlp_size=arc_mlp_size,
            tag_mlp_size=tag_mlp_size,
            label_mlp_size=label_mlp_size,
            ho_mlp_size=ho_mlp_size,
            dropout=dropout,
            word_dropout=word_dropout,
            predict_upos=predict_upos, predict_xpos=predict_xpos,
            predict_morph=predict_morph, predict_lemma=predict_lemma,
            predict_tree=predict_tree, pretrained_name_or_config=config)

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

        c = self.gsib_coeff
        states_hsg = c[0] * torch.tanh(heads + siblings + grandparents)
        states_msg = c[1] * torch.tanh(modifiers + siblings + grandparents)
        states_hmsg = c[2] * torch.tanh(
            heads + modifiers + siblings + grandparents)
        gsib_states = states_hsg + states_msg + states_hmsg
        gsib_scores = self.grandsibling_scorer(gsib_states)

        self.scores[Target.GRANDSIBLINGS].append(gsib_scores.view(-1))

    def _get_bert_representations(self, instances, max_num_tokens):
        """
        Get BERT encoded representations for the instances

        :return: a tensor (batch, max_num_tokens, encoder_dim)
        """
        batch_size = len(instances)
        # word piece lengths
        wp_lengths = torch.tensor([len(inst.bert_ids) for inst in instances],
                                  dtype=torch.long)
        if self.on_gpu:
            wp_lengths = wp_lengths.cuda()
        max_length = wp_lengths.max()
        indices = torch.zeros([batch_size, max_length], dtype=torch.long,
                              device=wp_lengths.device)

        # this contains the indices of the first word piece of real tokens.
        # positions past the sequence size will have 0's and will be ignored
        # afterwards anyway (treated as padding)
        real_indices = torch.zeros([batch_size, max_num_tokens],
                                   dtype=torch.long, device=wp_lengths.device)
        for i, inst in enumerate(instances):
            inst_length = wp_lengths[i]
            indices[i, :inst_length] = torch.tensor(
                inst.bert_ids, device=indices.device)

            # instance length is not the same as wordpiece length!
            # start from 1, because 0 will point to CLS as the root symbol
            real_indices[i, 1:len(inst)] = torch.tensor(
                inst.bert_token_starts, device=indices.device) + 1

        ones = torch.ones_like(indices)
        mask = ones.cumsum(1) <= wp_lengths.unsqueeze(1)

        if max_length > max_encoder_length:
            # if there are more tokens than the encoder can handle, break them
            # in smaller sequences.
            quarter_max = max_encoder_length // 4
            partial_encoded = []

            # possibly not all samples in the batch have the same length, but
            # even more likely one single huge sentence has the batch for itself
            ind_splits = torch.split(indices, quarter_max, 1)
            mask_splits = torch.split(mask, quarter_max, 1)

            for i in range(0, len(ind_splits) - 2, 2):
                # run the inputs through the encoder with at least a quarter of
                # context before and after
                partial_inds = torch.cat(ind_splits[i:i + 4], 1)
                partial_mask = torch.cat(mask_splits[i:i + 4], 1)

                # partial_hidden is a tuple of embeddings and hidden layers
                _, _, partial_hidden = self.encoder(partial_inds, partial_mask)

                last_states = torch.stack(partial_hidden[-4:])
                if i == 0:
                    # include the first quarter
                    last_states = last_states[:, :, :3 * quarter_max]
                elif i + 4 >= len(ind_splits):
                    # include the last quarter
                    last_states = last_states[:, :, quarter_max:]
                else:
                    # only take the middle two quarters
                    last_states = last_states[:, :, quarter_max:-quarter_max]

                partial_encoded.append(last_states.mean(0))

            encoded = torch.cat(partial_encoded, 1)
        else:
            # hidden is a tuple of embeddings and hidden layers
            _, _, hidden = self.encoder(indices, mask)

            # TODO: use a better aggregation scheme
            last_states = torch.stack(hidden[-4:])
            encoded = last_states.mean(0)

        # get the first wordpiece for tokens that were split, and CLS for root
        r = torch.arange(batch_size, device=encoded.device).unsqueeze(1)
        encoded = encoded[r, real_indices]

        return encoded

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

        if self.trainable_word_embeddings is not None:
            trainable_embeddings = self._get_embeddings(instances, max_length,
                                                        'trainableword')
            all_embeddings.append(trainable_embeddings)

        if self.encoder is not None:
            bert_embeddings = self._get_bert_representations(instances,
                                                             max_length)
            all_embeddings.append(bert_embeddings)

        if self.fixed_word_embeddings is not None:
            word_embeddings = self._get_embeddings(
                instances, max_length, 'fixedword')
            if self.transform_size > 0:
                word_embeddings = self.fixed_embedding_projection(
                    word_embeddings)
            all_embeddings.append(word_embeddings)

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
            if self.transform_size > 0:
                dropped = self.dropout(char_embeddings)
                char_embeddings = self.char_projection(dropped)
            all_embeddings.append(char_embeddings)

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

        if self.shared_rnn is not None:
            packed_embeddings = rnn_utils.pack_padded_sequence(
                embeddings, lengths, batch_first=True, enforce_sorted=False)

            # get hidden states for all words, ignore final cell
            packed_states, _ = self.shared_rnn(packed_embeddings)

            # ignore lengths -- we already know them!
            hidden_states, _ = rnn_utils.pad_packed_sequence(
                packed_states, batch_first=True)
        else:
            hidden_states = embeddings

        if self.predict_tags or self.predict_lemma:
            if self.tagger_rnn is None:
                # ignore root
                tagger_states = hidden_states[:, 1:]
            else:
                dropped = self.dropout(hidden_states)
                packed_states = rnn_utils.pack_padded_sequence(
                    dropped, lengths, batch_first=True, enforce_sorted=False)
                tagger_packed_states, _ = self.tagger_rnn(packed_states)
                tagger_states, _ = rnn_utils.pad_packed_sequence(
                    tagger_packed_states, batch_first=True)
                tagger_states = tagger_states[:, 1:]

            if self.predict_upos:
                if self.tag_mlp_size > 0:
                    scorer_states = self.upos_mlp(tagger_states)
                else:
                    scorer_states = tagger_states
                self.scores[Target.UPOS] = self.upos_scorer(scorer_states)

            if self.predict_xpos:
                if self.tag_mlp_size > 0:
                    scorer_states = self.xpos_mlp(tagger_states)
                else:
                    scorer_states = tagger_states
                self.scores[Target.XPOS] = self.xpos_scorer(scorer_states)

            if self.predict_morph:
                if self.tag_mlp_size > 0:
                    scorer_states = self.morph_mlp(tagger_states)
                else:
                    scorer_states = tagger_states
                self.scores[Target.MORPH] = self.morph_scorer(scorer_states)

            # if self.predict_lemma:
            #     if self.training:
            #         lemmas, lemma_lengths = get_padded_lemma_indices(
            #             instances, max_length)
            #         lemmas = lemmas.to(lengths.device)
            #         lemma_lengths = lemma_lengths.to(lengths.device)
            #     else:
            #         lemmas, lemma_lengths = None, None
            #
            #     # skip root
            #     logits = self.lemmatizer(
            #         char_indices[:, 1:], tagger_batch_states,
            #         token_lengths[:, 1:], lemmas, lemma_lengths)
            #     self.scores[Target.LEMMA] = logits

        if self.predict_tree:
            if self.parser_rnn is None:
                parser_states = hidden_states
            else:
                dropped = self.dropout(hidden_states)
                packed_states = rnn_utils.pack_padded_sequence(
                    dropped, lengths, batch_first=True, enforce_sorted=False)

                parser_packed_states, _ = self.parser_rnn(packed_states)
                parser_states, _ = rnn_utils.pad_packed_sequence(
                    parser_packed_states, batch_first=True)

            self._compute_arc_scores(
                parser_states, lengths, normalization)

            # now go through each batch item
            for i in range(batch_size):
                length = lengths[i].item()
                states = parser_states[i, :length]
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
