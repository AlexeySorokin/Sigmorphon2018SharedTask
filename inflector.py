import os
import inspect
import ujson as json
from collections import defaultdict
from functools import partial

from neural.common import *
from neural.common_neural import *
from neural.cells import MultiConv1D, TemporalDropout, History, LocalAttention
from neural.neural_LM import load_lm
from aligner.mcmc_aligner import Aligner, output_alignment
# from pe import PairEncoder

import keras.backend as kb
if kb.backend() == "tensorflow":
    from common_tensorflow import *
else:
    raise NotImplementedError
import keras.regularizers as kreg
from keras.optimizers import adam
import keras.callbacks as kcall

import neural.common_neural as kcall

def alignment_to_symbols(alignment, reverse=False, append_eow=False):
    """
    Преобразует выравнивание в список символов входа и выхода

    :param alignment:
    :param reverse:
    :return:
    """
    source, target, buffer = [BOW, BOW], [BOW], []
    curr_upper = BOW
    if reverse:
        alignment = alignment[::-1]
    for i, (x, y) in enumerate(alignment):
        if x != "":
            curr_upper = x
            target.append(STEP)
            source += [x] * len(buffer)
            target += buffer
            source.append(x)
            buffer = []
        if y != "":
            if curr_upper is not None:
                target.append(y)
                source.append(curr_upper)
            else:
                buffer.append(y)
        else:
            curr_upper = None
    target.append(STEP)
    source += [EOW] * len(buffer)
    target += buffer
    if append_eow:
        source.append(EOW)
        target.append(EOW)
    return source, target


def make_alignment_indexes(alignment, reverse=False):
    """
    Возвращает массив source_indexes, где source_indexes[i] ---
    позиция элемента, порождающего i-ый символ target

    :param alignment:
    :param reverse:
    :return: source_indexes, list of ints
    """
    source, target = alignment_to_symbols(alignment, reverse=reverse, append_eow=True)
    source_indexes, pos = [0] * len(target), 0
    for i, a in enumerate(target):
        source_indexes[i] = pos
        if a == STEP:
            pos += 1
    return source_indexes, target

def collect_symbol_statistics(alignments):
    symbol_statistics = defaultdict(lambda: [set(), set()])
    for alignment in alignments:
        curr_upper = BOW
        for x, y in alignment:
            if x != "":
                curr_upper, index = x, 0
            else:
                index = 1
            if y == "":
                y = STEP
            symbol_statistics[curr_upper][index].add(y)
    symbol_statistics = {x: [list(y[0]), list(y[1])] for x, y in symbol_statistics.items()}
    return symbol_statistics


def make_auxiliary_target(alignment, mode):
    if mode == "morphemes":
        PREFIX, SUFFIX, INFIX, STEM = 0, 1, 2, 3
    else:
        alignment = [('^', '^')] + list(alignment) + [('$', '$')]
    if mode == "identity":
        answer = [alignment[1] != ""]
        for i, (x, y) in enumerate(alignment[1:-1], 1):
            if x != "":
                answer.append((x == y) and alignment[i+1][0] != "")
        answer.append(alignment[-2] != "")
    elif mode == "removal":
        answer = [False]
        for i, (x, y) in enumerate(alignment[1:-1], 1):
            if x != "":
                answer.append(y == "")
        answer.append(False)
    elif mode == "substitution":
        answer = [False]
        for i, (x, y) in enumerate(alignment[1:-1], 1):
            if x != "":
                answer.append(y != x and y != "")
        answer.append(False)
    elif mode == "morphemes":
        for i, (x, y) in enumerate(alignment):
            if x == y:
                prefix_end = i
                break
        else:
            prefix_end = len(alignment)
        for i, (x, y) in enumerate(alignment[::-1]):
            if x == y:
                suffix_start = len(alignment) - i
                break
        else:
            suffix_start = len(alignment)
        answer_indexes = [STEM if prefix_end == 0 else PREFIX]
        for i, (x, y) in enumerate(alignment[:prefix_end]):
            if x != "":
                answer_indexes.append(PREFIX)
        for i, (x, y) in enumerate(alignment[prefix_end:suffix_start]):
            if x != "":
                is_stem = (x == y) and (i == len(alignment)-1 or alignment[i+1][0] != "")
                answer_indexes.append(STEM if is_stem else INFIX)
        for i, (x, y) in enumerate(alignment[suffix_start:]):
            if x != "":
                answer_indexes.append(SUFFIX)
        answer_indexes.append(SUFFIX if suffix_start < len(alignment) else STEM)
        # answer = np.zeros(shape=(len(answer_indexes), 4), dtype=int)
        # answer[np.arange(len(answer)), answer_indexes] = 1
        answer = answer_indexes
    else:
        raise ValueError("Unknown function mode:", mode)
    return answer


def make_auxiliary_targets(alignment, modes, reverse=False):
    answer = [make_auxiliary_target(alignment, mode) for mode in modes]
    if reverse:
        answer = [elem[::-1] for elem in answer]
    return answer


def extend_history(histories, hyps, indexes, start=0, pos=None,
                   history_pos=0, value=None, func="append", **kwargs):
    """
    Updates histories with given indexes by a selected element
    of corresponding hypotheses. Schematically,
        histories[indexes[j]] = update_func(histories[indexes[j]], hyps[j][pos]),
        j = 0, ..., len(indexes) - 1
    :param histories:
    :param hyps:
    :param indexes:
    :param start:
    :param pos:
    :param history_pos:
    :param value:
    :param func:
    :param kwargs:
    :return:
    """
    if func != "copy":
        to_append = ([elem[pos] for elem in hyps] if (value is None) else ([value] * len(hyps)))
    else:
        to_append = [None] * len(hyps)
    if func == "append":
        func = lambda x, y: x + [y]
    elif func == "sum":
        func = lambda x, y: x + y
    elif func == "change":
        func = lambda x, y: y
    elif func == "append_truncated":
        func = lambda x, y: np.concatenate([x[1:], [y]])
    elif func == "set":
        i = kwargs["position"]
        func = lambda x, y: np.concatenate([x[:i], [y], x[i+1:]])
    elif func == "copy":
        func = lambda x, y: x
    elif not callable(func):
        raise ValueError("func must be 'append', 'sum' or a callable object")
    group_histories = [func(histories[elem[history_pos]], value)
                       for elem, value in zip(hyps, to_append)]
    for i, index in enumerate(indexes):
        histories[start+index] = group_histories[i]
    return


def load_inflector(infile, verbose=1):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback")
                    or key == "model_files" or key == "mask")}
    args['verbose'] = verbose
    # коллбэки
    args['callbacks'] = []
    for key, cls in zip(["early_stopping_callback", "reduce_LR_callback"],
                        [MultiEarlyStopping, ReduceLROnPlateau]):
        if key in json_data:
            args['callbacks'].append(cls(**json_data[key]))
    # создаём языковую модель
    args['slow'] = False
    inflector = Inflector(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        setattr(inflector, key, value)
    if inflector.use_lm and not hasattr(inflector, "recodings_for_lm_"):
        inflector.recodings_for_lm_ =\
            {i: inflector.lm_.toidx(x) for i, x in enumerate(inflector.symbols_)}
    elif inflector.use_lm:
        inflector.recodings_for_lm_ = {int(i): x for i, x in inflector.recodings_for_lm_.items()}
    # модель
    inflector.build()  # не работает сохранение модели, приходится сохранять только веса
    for i, (model, model_file) in enumerate(
            zip(inflector.models_, json_data['model_files'])):
        model.load_weights(model_file)
    return inflector


class Inflector:

    UNKNOWN_FEATURE = 0

    DEFAULT_ALIGNER_PARAMS = {"init_params": {"gap": -1, "initial_gap": -1}, "n_iter": 5,
                              "init": "lcs", "separate_endings": True, "verbose": 0}
    MAX_STEPS_NUMBER = 3

    def __init__(self, slow=False, aligner_params=None, use_full_tags=False,
                 models_number=1, buckets_number=10, batch_size=32,
                 nepochs=25, validation_split=0.2, reverse=False,
                 augmentation_weight = 0.1,
                 use_lm=False, lm_file=None,
                 use_input_attention=False, input_window=5,
                 output_history=1, use_output_embeddings=False, output_embeddings_size=16,
                 # separate_symbol_history=False, step_history=1,
                 # use_output_attention=False, history_embedding_size=32,
                 use_embeddings=False, embeddings_size=16,
                 use_symbol_statistics=False,
                 feature_embedding_layers=0, feature_embeddings_size=16,
                 conv_layers=0, conv_window=32, conv_filters=5,
                 rnn="lstm", encoder_rnn_layers=1, encoder_rnn_size=32,
                 attention_key_size=32, attention_value_size=32,
                 decoder_rnn_size=32, dense_output_size=32,
                 use_decoder_gate="", features_after_decoder=False,
                 conv_dropout=0.0, encoder_rnn_dropout=0.0, dropout=0.0,
                 history_dropout=0.0, decoder_dropout=0.0, lm_dropout=0.0,
                 regularizer=0.0, callbacks=None,
                 step_loss_weight=0.0, auxiliary_targets=None,
                 auxiliary_target_weights=None, auxiliary_dense_units=None,
                 encode_auxiliary_symbol_outputs=False,
                 min_prob=0.01, max_diff=2.0,
                 random_state=187, verbose=1):
        self.slow = slow
        self.use_full_tags = use_full_tags
        self.models_number = models_number
        self.buckets_number = buckets_number
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.reverse = reverse
        self.augmentation_weight = augmentation_weight
        self.use_lm = use_lm
        self.lm_file = lm_file
        self.use_input_attention = use_input_attention
        self.input_window = input_window
        # self.input_right_context = input_right_context
        # self.use_attention = use_attention
        self.output_history = output_history
        self.use_output_embeddings = use_output_embeddings
        self.output_embeddings_size = output_embeddings_size
        # self.separate_symbol_history = separate_symbol_history
        # self.step_history = step_history
        # self.use_output_attention = use_output_attention
        # self.history_embedding_size = history_embedding_size
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.use_symbol_statistics = use_symbol_statistics
        self.feature_embedding_layers = feature_embedding_layers
        self.feature_embeddings_size = feature_embeddings_size
        self.conv_layers = conv_layers
        self.conv_window = conv_window
        self.conv_filters = conv_filters
        # self.rnn = rnn
        self.encoder_rnn_layers = encoder_rnn_layers
        self.encoder_rnn_size = encoder_rnn_size
        self.attention_key_size = attention_key_size
        self.attention_value_size = attention_value_size
        self.decoder_rnn_size = decoder_rnn_size
        self.dense_output_size = dense_output_size
        self.use_decoder_gate = use_decoder_gate
        self.features_after_decoder = features_after_decoder
        # self.regularizer = regularizer
        self.conv_dropout = conv_dropout
        self.encoder_rnn_dropout = encoder_rnn_dropout
        self.dropout = dropout
        self.history_dropout = history_dropout
        self.decoder_dropout = decoder_dropout
        self.lm_dropout = lm_dropout
        self.regularizer = regularizer

        self.step_loss_weight = step_loss_weight
        self.auxiliary_targets = auxiliary_targets or []
        self.auxiliary_target_weights = auxiliary_target_weights
        self.auxiliary_dense_units = auxiliary_dense_units
        self.encode_auxiliary_symbol_outputs = encode_auxiliary_symbol_outputs
        # # декодинг
        self.min_prob = min_prob
        self.max_diff = max_diff
        # разное
        self.random_state = random_state
        self.verbose = verbose
        # выравнивания
        self._make_aligner(aligner_params)
        self._initialize(callbacks)
        # self._initialize_callbacks(model_file)

    def _initialize(self, callbacks=None):
        if self.lm_file is not None:
            self.lm_ = load_lm(self.lm_file)
        if isinstance(self.conv_window, int):
            self.conv_window = [self.conv_window]
        if isinstance(self.conv_filters, int):
            self.conv_filters = [self.conv_filters]
        # if isinstance(self.rnn, str):
        #     self.rnn = getattr(kl, self.rnn.upper())
        # if self.rnn not in [kl.GRU, kl.LSTM]:
        #     raise ValueError("Unknown recurrent network: {}".format(self.rnn))
        if self.auxiliary_targets_number:
            for attr in ["auxiliary_target_weights", "auxiliary_dense_units"]:
                value = getattr(self, attr)
                if isinstance(value, (int, float)):
                    setattr(self, attr, [value] * self.auxiliary_targets_number)
                elif len(value) != self.auxiliary_targets_number:
                    raise ValueError("Wrong number of elements in {}, "
                                     "there are {} auxiliary targets".format(
                                        value, self.auxiliary_targets_number))
        self.encode_auxiliary_symbol_outputs &= (self.auxiliary_targets_number > 0)
        callbacks = callbacks or dict()
        self.callbacks = [getattr(kcall, key)(**params) for key, params in callbacks.items()]

    def _make_aligner(self, aligner_params):
        """
        Создаём выравниватель

        :param aligner_params: dict or None,
            параметры выравнивателя
        :return: self
        """
        if aligner_params is None:
            aligner_params = self.DEFAULT_ALIGNER_PARAMS
        self.aligner = Aligner(**aligner_params)
        return self

    def load_model(self, infile):
        self.build()
        for i in range(self.models_number):
            self.models_[i].load_weights(self._make_model_file(infile, i+1))
        return self

    def to_json(self, outfile, model_file):
        info = dict()
        model_files = [self._make_model_file(model_file, i+1) for i in range(self.models_number)]
        for i in range(self.models_number):
            model_files[i] = os.path.abspath(model_files[i])
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(Inflector, attr, None), property) or
                    attr.isupper() or attr in ["callbacks", "models_", "aligner",
                                               "encoders_", "decoders_", "lm_", "func"]):
                info[attr] = val
            elif attr == "models_":
                info["model_files"] = model_files
                for model, curr_model_file in zip(self.models_, model_files):
                    model.save_weights(curr_model_file)
            elif attr == "callbacks":
                callback_info = dict()
                for callback in val:
                    if isinstance(callback, MultiEarlyStopping):
                        callback_info["MultiEarlyStopping"] =\
                            {key: getattr(callback, key) for key in ["patience", "monitor", "min_delta"]}
                    elif isinstance(callback, ReduceLROnPlateau):
                        callback_info["ReduceLROnPlateau"] = \
                            {key: getattr(callback, key) for key in ["patience", "monitor", "factor"]}
                info["callbacks"] = callback_info
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    def _make_model_file(self, name, i):
        pos = name.rfind(".")
        if pos != -1:
            # добавляем номер модели перед расширением
            return "{}-{}.{}".format(name[:pos], i, name[pos+1:])
        else:
            return "{}-{}".format(name, i)

    def set_language_model(self, model):
        self.lm_ = model

    @property
    def symbols_number(self):
        return len(self.symbols_)

    @property
    def labels_number_(self):
        return len(self.labels_)

    @property
    def feature_vector_size(self):
        return (self.labels_number_ + len(self.tags_)) if self.use_full_tags else self.labels_number_

    @property
    def inputs_number(self):
        return 4 + int(self.use_symbol_statistics) + int(self.use_lm != False)

    @property
    def auxiliary_targets_number(self):
        return len(self.auxiliary_targets)

    @property
    def lm_states_number(self):
        return 2 + 2 * int(self.lm_.has_attention)

    @property
    def lm_encoder_dim(self):
        return self.lm_.encoder_rnn_size

    @property
    def lm_state_dim(self):
        return self.lm_.dense_output_size if self.use_lm == "logit" else self.lm_.decoder_rnn_size

    @property
    def lm_decoder_dim(self):
        return self.lm_.decoder_rnn_size

    @property
    def lm_step_func(self):
        return self.lm_._logit_step_func_ if self.use_lm == "logit" else self.lm_._step_func_

    def _make_vocabulary(self, X):
        symbols = {a for first, second, _ in X for a in first+second}
        self.symbols_ = AUXILIARY + [STEP] + sorted(symbols)
        self.symbol_codes_ = {a: i for i, a in enumerate(self.symbols_)}
        return self

    def _make_features(self, descrs):
        """
        Extracts possible POS and feature values

        Parameters:
        -----------
            descrs: list of lists,
                descrs = [descr_1, ..., descr_m],
                descr = [pos, feat_1, ..., feat_k]

        Returns:
        -----------
            self
        """
        pos_labels, feature_labels, tags = set(), set(), set()
        for elem in descrs:
            pos_label = elem[0]
            curr_feature_labels = {pos_label + "_" + x for x in elem[1:]}
            pos_labels.add(pos_label)
            feature_labels.update(curr_feature_labels)
            tags.add(tuple(elem))
        self.labels_ = AUXILIARY + sorted(pos_labels) + sorted(feature_labels)
        self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        if self.use_full_tags:
            self.tags_ = sorted(tags)
            self.tag_codes_ = {x: i for i, x in enumerate(self.tags_)}
        return self

    def extract_features(self, descr):
        answer = np.zeros(shape=(self.feature_vector_size,), dtype=np.int32)
        features = [descr[0]] + ["{}_{}".format(descr[0], x) for x in descr[1:]]
        feature_codes = [self.label_codes_[x] for x in features if x in self.label_codes_]
        answer[feature_codes] = 1
        if self.use_full_tags:
            code = self.label_codes_.get(tuple(descr))
            if code is not None:
                answer[len(self.labels_) + code] = 1
        return answer

    def _make_bucket_data(self, lemmas, bucket_indexes):
        bucket_size = len(bucket_indexes)
        # bucket_length = max(len(x) + 3 for x in lemmas)
        bucket_length = max(len(lemmas[i]) + 3 for i in bucket_indexes)
        bucket_data = np.full(shape=(bucket_size, bucket_length),
                              fill_value=PAD, dtype=int)
        # заполняем закодированными символами
        bucket_data[:,0] = BEGIN
        for j, i in enumerate(bucket_indexes):
            lemma = lemmas[i]
            bucket_data[j,1:1+len(lemma)] = [self.symbol_codes_.get(x, UNKNOWN) for x in lemma]
            bucket_data[j,1+len(lemma)] = END
        return bucket_data

    def _extract_symbol_data(self, lemmas):
        answer = np.zeros(shape=lemmas.shape+(2*self.symbols_number,), dtype=int)
        for i, lemma in enumerate(lemmas):
            for j, x in enumerate(lemma):
                if x == END:
                    break
                x = self.symbols_[x]
                if x == UNKNOWN or x not in self.symbol_statistics_:
                    possible_symbols = [x], []
                else:
                    possible_symbols = self.symbol_statistics_[x]
                possible_symbols = [np.fromiter((self.symbol_codes_[y] for y in elem), dtype=int)
                                    for elem in possible_symbols]
                answer[i,j,possible_symbols[0]] = 1
                answer[i,j, self.symbols_number+np.array(possible_symbols[1])] = 1
        return answer

    def _make_shifted_output(self, bucket_length, bucket_size, targets=None):
        answer = np.full(shape=(bucket_size, bucket_length), fill_value=BEGIN, dtype=int)
        if targets is not None:
            answer[:,1:] = targets[:, :-1]
        # answer = np.concatenate([np.ones_like(targets[:,:1]) * BEGIN, targets[:,:-1]], axis=1)
        return answer

    def _make_steps_shifted_output(self, bucket_length, bucket_size, targets=None):
        """
        Constructs the history of step operations for each position

        bucket_length: int, length of sequences in targets
        bucket_size: int, number of objects in targets
        targets: np.array of ints or None
        :return:
        """
        answer = np.zeros(shape=(bucket_size, bucket_length,
                                 self.step_history), dtype=bool)
        if targets is not None:
            for i in range(1, bucket_length):
                d = i - self.step_history
                answer[:,i,max(-d,0):] = (targets[:,max(d, 0):i] == self.STEP_CODE)
        return answer

    def _make_symbols_shifted_output(self, bucket_length, bucket_size, targets=None):
        """
        Constructs the history of output symbols (excluding step actions)
        for each position in the target sequence

        :param bucket_length:
        :param bucket_size:
        :param targets:
        :return:
        """
        answer = np.zeros(shape=(bucket_size, bucket_length,
                                 self.output_history), dtype=np.int32)
        answer[:,0] = BEGIN
        if targets is not None:
            for i in range(1, bucket_length):
                # shift the history only if last action was not the step symbol
                answer[:,i] = np.where(targets[:,i-1,None] == STEP_CODE, answer[:,i-1],
                                       np.hstack([answer[:,i-1,1:], targets[:,i-1,None]]))
        return answer

    def _make_targets_with_copy(self, targets, lemmas, indexes):
        targets_with_copy = []
        for i, (curr_lemma, curr_indexes, curr_target) in\
                enumerate(zip(lemmas, indexes, targets)):
            curr_answer = curr_target[:]
            for i, (index, y) in enumerate(zip(curr_indexes, curr_target)):
                if y in ['BEGIN', 'STEP']:
                    continue
                if y == 'END' or index >= len(curr_lemma):
                    break
                # index-1 to deal with BEGIN prepending
                if y == curr_lemma[index-1] and curr_indexes[i-1] == index-1:
                    curr_answer[i] = 'COPY'
            targets_with_copy.append(curr_answer)
        return targets_with_copy

    def _make_lm_data(self, data, indexes):
        lm_states, _ = self.lm_.predict_states_batch(data, mode=self.use_lm)
        lm_states = np.concatenate([np.zeros_like(lm_states[:,:1]), lm_states], axis=1)
        right_pad_length = np.max(indexes)+1 - lm_states.shape[1]
        if right_pad_length > 0:
            right_pad = np.repeat(lm_states[:,-1:], right_pad_length, axis=1)
            lm_states = np.concatenate([lm_states, right_pad], axis=1)
        states = lm_states[np.arange(len(data))[:,None], indexes]
        return states

    def transform_training_data(self, lemmas, features, letter_positions,
                                targets, words, auxiliary_targets=None,
                                buckets_number=None, bucket_size=-1):
        """

        lemmas: list of strs,
            список лемм (без добавления начала и конца строки)
        letter_positions: list of lists of ints
            список индексов букв в выравнивании,
            letter_indexes[r][i] = j <=> lemmas[r][i] порождает targets[r][j]
        targets: list of strs,
            список порождаемых словоформ (с учётом начала и конца строки)

        """
        if buckets_number is None and bucket_size == -1:
            buckets_number = self.buckets_number
        lemmas_lengths = [len(lemma)+3 for lemma in lemmas]
        alignment_lengths = [len(target) for target in targets]
        self.max_length_shift_ = max(
            0, max(y - 2*x for x, y in zip(lemmas_lengths, alignment_lengths)))
        buckets_with_indexes = collect_buckets(
            lemmas_lengths, buckets_number, max_bucket_length=bucket_size)
        data_by_buckets = [self._make_bucket_data(lemmas, indexes)
                           for length, indexes in buckets_with_indexes]
        features_by_buckets = [
            np.array([self.extract_features(features[i]) for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        targets = [[self.symbol_codes_.get(x, UNKNOWN) for x in elem] for elem in targets]
        letter_positions_by_buckets = [
            make_table(letter_positions, indexes, fill_with_last=True)
            for length, indexes in buckets_with_indexes]
        if self.use_symbol_statistics:
            symbol_data_by_buckets = [self._extract_symbol_data(bucket) for bucket in data_by_buckets]
        if self.use_lm:
            lm_data_by_buckets = []
            for i, (_, indexes) in enumerate(buckets_with_indexes):
                curr_positions = letter_positions_by_buckets[i]
                curr_data = [[words[i], features[i]] for i in indexes]
                curr_positions_in_target = np.arange(curr_positions.shape[1])[None,:] - curr_positions
                curr_lm_data = self._make_lm_data(curr_data, curr_positions_in_target)
                lm_data_by_buckets.append(curr_lm_data)
        targets_by_buckets = [make_table(targets, indexes, fill_value=PAD)
                              for length, indexes in buckets_with_indexes]
        if auxiliary_targets is not None:
            fill_values = [elem == "identity" for elem in self.auxiliary_targets]
            auxiliary_targets_by_buckets = []
            for i, elem in enumerate(auxiliary_targets):
                auxiliary_targets_by_buckets.append(
                    [make_table(elem, indexes, fill_value=fill_values[i], length=length)
                     for length, indexes in buckets_with_indexes])
        else:
            auxiliary_targets_by_buckets = []
        alignment_length_by_buckets = [max(len(letter_positions[i]) for i in indexes)
                                       for _, indexes in buckets_with_indexes]
        for_bucket_data = [(L, indexes) for L, (_, indexes) in zip(
                                alignment_length_by_buckets, buckets_with_indexes)]
        history_by_buckets = [self._make_shifted_output(L, len(indexes), elem)
                              for elem, (L, indexes) in zip(targets_by_buckets, for_bucket_data)]
        answer = list(zip(data_by_buckets, features_by_buckets,
                          letter_positions_by_buckets, history_by_buckets))
        for i in range(len(answer)):
            if self.use_symbol_statistics:
                answer[i] += (symbol_data_by_buckets[i],)
            if self.use_lm:
                answer[i] += (lm_data_by_buckets[i],)
            answer[i] += (targets_by_buckets[i],)
            for elem in auxiliary_targets_by_buckets:
                answer[i] += (elem[i],)
        return answer, [elem[1] for elem in buckets_with_indexes]

    def _preprocess(self, data, alignments, to_fit=True, alignments_outfile=None,
                    buckets_number=None, bucket_size=-1):
        to_align = [(elem[0], elem[1]) for elem in data]
        if alignments is None:
            # вычисляем выравнивания
            alignments = self.aligner.align(to_align, to_fit=to_fit)
            if alignments_outfile is not None:
                output_alignment(to_align, alignments, alignments_outfile, sep="_")
        elif to_fit:
            self.aligner.align(to_align, to_fit=True, only_initial=True)
        indexes_with_targets = [make_alignment_indexes(alignment, reverse=self.reverse)
                                for alignment in alignments]
        if to_fit and self.use_symbol_statistics:
            self.symbol_statistics_ = collect_symbol_statistics(alignments)
        if len(self.auxiliary_targets) > 0:
            auxiliary_letter_targets = [
                make_auxiliary_targets(alignment, self.auxiliary_targets, reverse=self.reverse)
                for alignment in alignments]
            auxiliary_letter_targets = list(map(list, zip(*auxiliary_letter_targets)))
        else:
            auxiliary_letter_targets = None
        lemmas, words = [elem[0] for elem in data], [elem[1] for elem in data]
        features = [elem[2] for elem in data]
        letter_indexes = [elem[0] for elem in indexes_with_targets]
        targets = [elem[1] for elem in indexes_with_targets]
        if self.reverse:
            lemmas = [x[::-1] for x in lemmas]
        return self.transform_training_data(lemmas, features, letter_indexes,
                                            targets, words, auxiliary_letter_targets,
                                            buckets_number=buckets_number, bucket_size=bucket_size)

    def train(self, data, alignments=None, dev_data=None, dev_alignments=None,
              alignments_outfile=None, augmented_data=None,
              save_file=None, model_file=None):
        """

        Parameters:
        -----------
            data: list of tuples,
                data = [(lemma, word, problem_descr), ..]
                problem_descr = [pos, feat_1, ..., feat_r]
            alignment(optional): list of lists of pairs of strs or None,
                list of 0-1,1-0,0-0 alignments for each lemma-word pair in data,
                alignment = [[(i1, o1), ..., (ir, or)], ...]

        Returns:
        -----------
            self

        """
        # if self.use_pair_encoding:
        #     data_for_pair_encoding = [x for elem in data for x in elem[:2]]
        #     self.pair_encoder = PairEncoder().fit(data_for_pair_encoding)
        #     transformed_data = self.pair_encoder.transform(data_for_pair_encoding)
        #     data = [[transformed_data[2*i], transformed_data[2*i+1], feats]
        #             for i, (_, _, feats) in enumerate(data)]
        self._make_vocabulary(data)
        self._make_features([elem[2] for elem in data])
        if self.use_lm:
            self.recodings_for_lm_ = {i: self.lm_.toidx(x) for i, x in enumerate(self.symbols_)}
        if augmented_data is not None:
            weights = [1.0] * len(data) + [self.augmentation_weight] * len(augmented_data)
            data += augmented_data
        else:
            weights = [1.0] * len(data)
        weights = np.array(weights)
        data_by_buckets, buckets_indexes = self._preprocess(
            data, alignments, to_fit=True, alignments_outfile=alignments_outfile)
        if dev_data is not None:
            dev_data_by_buckets, dev_bucket_indexes =\
                self._preprocess(dev_data, dev_alignments, bucket_size=64, to_fit=False)
        else:
            dev_data_by_buckets, dev_bucket_indexes = None, None
        if not hasattr(self, "built_"):
            self.build()
            if save_file is not None:
                if model_file is None:
                    pos = save_file.rfind(".") if "." in save_file else len(save_file)
                    model_file = save_file[:pos] + "-model.hdf5"
                self.to_json(save_file, model_file)
        weights = [weights[elem] for elem in buckets_indexes]
        self._train_model(data_by_buckets, X_dev=dev_data_by_buckets, weights=weights, model_file=model_file)
        return self

    def _train_model(self, X, X_dev=None, weights=None, model_file=None):
        if weights is None:
            weights = [1.0] * len(X)
        weights = np.array(weights)
        train_indexes_by_buckets, dev_indexes_by_buckets = [], []
        weights_by_buckets, dev_weights_by_buckets = [], []
        np.random.seed(self.random_state)
        for curr_data, curr_weights in zip(X, weights):
            curr_indexes = list(range(len(curr_data[0])))
            np.random.shuffle(curr_indexes)
            if X_dev is None:
                # отделяем в каждой корзине данные для валидации
                train_bucket_size = int((1.0 - self.validation_split) * len(curr_indexes))
                train_indexes_by_buckets.append(curr_indexes[:train_bucket_size])
                dev_indexes_by_buckets.append(curr_indexes[train_bucket_size:])
                weights_by_buckets.append(curr_weights[curr_indexes[:train_bucket_size]])
                dev_weights_by_buckets.append(curr_weights[curr_indexes[train_bucket_size:]])
            else:
                train_indexes_by_buckets.append(curr_indexes)
                weights_by_buckets.append(curr_weights[curr_indexes])
        if X_dev is not None:
            for curr_data in X_dev:
                dev_indexes_by_buckets.append(list(range(len(curr_data[0]))))
                dev_weights_by_buckets.append(np.ones(shape=len(curr_data[0],)))
        train_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(train_indexes_by_buckets))))
        dev_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(dev_indexes_by_buckets))))
        if X_dev is None:
            X_dev = X
        for i, model in enumerate(self.models_):
            if model_file is not None:
                curr_model_file = self._make_model_file(model_file, i+1)
                monitor = "outputs_acc" if self.auxiliary_targets_number else "acc"
                save_callback = ModelMultiCheckpoint(
                    curr_model_file, save_weights_only=True, save_best_only=True, monitor=monitor)
                curr_callbacks = self.callbacks + [save_callback]
            else:
                curr_callbacks = self.callbacks
            curr_callbacks.append(BasicMetricsProgbarLogger(verbose=1))
            if "morphemes" in self.auxiliary_targets:
                auxiliary_symbols_number = [(self.auxiliary_targets.index("morphemes")+1, 4)]
            else:
                auxiliary_symbols_number = None
            train_gen = generate_data(X, train_indexes_by_buckets, train_batches_indexes,
                                      self.batch_size, self.symbols_number,
                                      inputs_number=self.inputs_number, weights=weights_by_buckets,
                                      auxiliary_symbols_number=auxiliary_symbols_number)
            val_gen = generate_data(X_dev, dev_indexes_by_buckets, dev_batches_indexes,
                                    self.batch_size, self.symbols_number, shuffle=False,
                                    inputs_number=self.inputs_number, weights=dev_weights_by_buckets,
                                    auxiliary_symbols_number=auxiliary_symbols_number)
            model.fit_generator(train_gen, len(train_batches_indexes), epochs=self.nepochs,
                                callbacks=curr_callbacks, verbose=0,
                                validation_data=val_gen, validation_steps=len(dev_batches_indexes))
            if model_file is not None:
                self.models_[i].load_weights(curr_model_file)
                ### ВЫДАЁТ ОШИБКУ NOT_JSON_SERIALIZABLE
                # keras.models.save_model(self.models_[i], self.model_files[i])
        return self

    def build(self):
        self.models_ = [None] * self.models_number
        self.encoders_ = [None] * self.models_number
        self.decoders_ = [None] * self.models_number
        for i in range(self.models_number):
            self.models_[i], self.encoders_[i], self.decoders_[i] = self._build_model()
        self.built_ = "train"
        if self.verbose > 0:
            print(self.models_[0].summary())
        return self

    def rebuild_test(self):
        for i, model in enumerate(self.models_):
            weights = model.get_weights()
            self.models_[i], self.encoders_[i], self.decoders_[i] = self._build_model(test=True)
            self.models_[i].set_weights(weights)
        if self.use_lm:
            self.lm_.rebuild()
        self.built_ = "test"
        return self

    def _build_word_network(self, inputs, feature_inputs, stats_inputs=None):
        if self.use_embeddings:
            embedding = kl.Embedding(self.symbols_number, self.embeddings_size, mask_zero=True)
        else:
            embedding = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                  output_shape=(None, self.symbols_number))
        outputs = embedding(inputs)
        # outputs = kl.Lambda(lambda x, y: x * y[..., None], arguments={"y": useful_symbols_mask})(outputs)
        if self.conv_layers > 0:
            outputs = MultiConv1D(outputs, self.conv_layers, self.conv_window,
                                  self.conv_filters, self.conv_dropout, activation="relu")
        if self.encoder_rnn_layers > 0:
            for i in range(self.encoder_rnn_layers):
                outputs = kl.Bidirectional(kl.LSTM(self.encoder_rnn_size,
                                                   dropout=self.encoder_rnn_dropout,
                                                   return_sequences=True))(outputs)
        if self.use_input_attention:
            h, r = (self.input_window // 2) + 1, (self.input_window - 1) // 2
            outputs = LocalAttention(outputs, self.attention_key_size, self.attention_value_size, h, r)
        if self.use_symbol_statistics:
            stats_inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(stats_inputs)
            outputs = kl.Concatenate()([outputs, stats_inputs])
        if self.auxiliary_targets_number > 0:
            auxiliary_probs, auxiliary_logits = [], []
            for i, target_name in enumerate(self.auxiliary_targets):
                args = [outputs, feature_inputs, self.auxiliary_dense_units[i]]
                if target_name == "morphemes":
                    args.append(4)
                probs, logits = self._build_auxiliary_objective_network(*args, name=target_name)
                auxiliary_probs.append(probs)
                auxiliary_logits.append(logits)
            if self.encode_auxiliary_symbol_outputs:
                outputs = kl.Concatenate()([outputs] + auxiliary_logits)
        else:
            auxiliary_probs = None
        return outputs, auxiliary_probs

    @property
    def symbol_outputs_dim(self):
        if self.encoder_rnn_layers > 0:
            return 2 * self.encoder_rnn_size
        else:
            return sum(self.conv_filters)

    def _build_alignment(self, inputs, indexes, dropout=0.0):
        if kb.backend() == "tensorflow":
            indexing_function = gather_indexes
        lambda_func = kl.Lambda(indexing_function, output_shape=lambda x:x[0])
        answer = lambda_func([inputs,  indexes])
        if dropout > 0.0:
            answer = kl.Dropout(dropout)(answer)
        return answer

    def _build_feature_network(self, inputs, k):
        if self.feature_embedding_layers:
            inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
            inputs = kl.Dense(self.feature_embeddings_size,
                              input_shape=(self.feature_vector_size,),
                              activation="relu", use_bias=False)(inputs)
            for _ in range(1, self.feature_embedding_layers):
                inputs = kl.Dense(self.feature_embeddings_size,
                                  input_shape=(self.feature_embeddings_size,),
                                  activation="relu", use_bias=False)(inputs)
        def tiling_func(x):
            x = kb.expand_dims(x, 1)
            return kb.tile(x, [1, k, 1])
        answer = kl.Lambda(tiling_func, output_shape=(lambda x: (None,) + x))(inputs)
        answer = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(answer)
        return answer

    def _build_history_network(self, inputs, only_last=False):
        if self.use_output_embeddings:
            inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
            outputs = kl.Embedding(self.symbols_number, self.output_embeddings_size)(inputs)
        else:
            outputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                output_shape=(None, self.symbols_number))(inputs)
        if self.history_dropout > 0.0:
            outputs = TemporalDropout(outputs, self.history_dropout)
        if self.output_history > 1 or only_last:
            outputs = History(outputs, self.output_history, flatten=True, only_last=only_last)
        return outputs

    def _build_decoder_network(self, inputs, source, lm_inputs=None):
        to_concatenate = [inputs[0]] + inputs[2:] if self.features_after_decoder else inputs
        concatenated_inputs = kl.Concatenate()(to_concatenate)
        decoder = kl.LSTM(self.decoder_rnn_size, dropout=self.decoder_dropout,
                          return_sequences=True, return_state=True)
        initial_states = [kb.zeros_like(concatenated_inputs[:,0,0]),
                          kb.zeros_like(concatenated_inputs[:,0,0])]
        for i, elem in enumerate(initial_states):
            initial_states[i] = kb.tile(elem[:,None], [1, self.decoder_rnn_size])
        decoder_outputs, final_h_states, final_c_states =\
            decoder(concatenated_inputs, initial_state=initial_states)
        if self.features_after_decoder:
            decoder_outputs = kl.Concatenate()([decoder_outputs, inputs[1]])
        if lm_inputs is not None and self.use_lm != "logit":
            if self.lm_dropout > 0.0:
                lm_inputs = TemporalDropout(lm_inputs, self.lm_dropout)
            decoder_outputs = kl.Concatenate()([decoder_outputs, lm_inputs])
        pre_outputs = kl.TimeDistributed(kl.Dense(
            self.dense_output_size, activation="relu", use_bias=False,
            input_shape=(self.decoder_rnn_size,)))(decoder_outputs)
        if lm_inputs is not None and self.use_lm == "logit":
            if self.lm_dropout > 0.0:
                lm_inputs = TemporalDropout(lm_inputs, self.lm_dropout)
            pre_outputs = kl.Concatenate()([pre_outputs, lm_inputs])
        output_name = "outputs" if not self.use_decoder_gate else "basic_outputs"
        outputs = kl.TimeDistributed(kl.Dense(
            self.symbols_number, activation="softmax", name=output_name,
            activity_regularizer=kreg.l2(self.regularizer),
            input_shape=(self.dense_output_size,)))(pre_outputs)
        if self.use_decoder_gate:
            concatenated_inputs = kl.Concatenate()(inputs)
            gate_inputs = kl.Concatenate()([concatenated_inputs, decoder_outputs])

            source_inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                      output_shape=(lambda x: x + (self.symbols_number,)))(source)
            if self.use_decoder_gate == "step":
                gate_outputs = kl.Dense(3, activation="softmax")(gate_inputs)
                step_inputs = kl.Lambda(lambda x: kb.ones_like(x) * STEP_CODE)(source)
                step_inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                        output_shape=(lambda x: x + (self.symbols_number,)))(step_inputs)
                output_layer = kl.Lambda(multigated_sum, arguments={"disable_first": True},
                                         output_shape=(lambda x: x[0]), name="outputs")
                outputs = output_layer([source_inputs, step_inputs, outputs, gate_outputs])
            else:
                gate_outputs = kl.Dense(1, activation="sigmoid")(gate_inputs)
                output_layer = kl.Lambda(gated_sum, arguments={"disable_first": False},
                                         output_shape=(lambda x:x[0]), name="outputs")
                outputs = output_layer([source_inputs, outputs, gate_outputs])
        return outputs, initial_states, [final_h_states, final_c_states]

    def _build_auxiliary_objective_network(self, inputs, feature_inputs, units_number,
                                           output_units=1, name=None):
        activation = "sigmoid" if output_units == 1 else "softmax"
        feature_inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(feature_inputs)
        feature_inputs = kl.Dense(self.feature_embeddings_size,
                                  activation="relu", use_bias=False)(feature_inputs)
        def tiling_func(x):
            return kb.tile(x[:,None], [1, kb.shape(inputs)[1], 1])
        feature_inputs = kl.Lambda(tiling_func, output_shape=(lambda x: (None,) + x))(feature_inputs)
        pre_answer = kl.Concatenate()([inputs, feature_inputs])
        pre_answer = kl.Dense(units_number, activation="relu")(pre_answer)
        pre_answer = kl.Dense(output_units)(pre_answer)
        answer = kl.Activation(activation, name=name)(pre_answer)
        return answer, pre_answer


    def _build_model(self, test=False):
        # входные данные
        symbol_inputs = kl.Input(name="symbol_inputs", shape=(None,), dtype='int32')
        feature_inputs = kl.Input(
            shape=(self.feature_vector_size,), name="feature_inputs", dtype='int32')
        letter_indexes_inputs = kl.Input(shape=(None,), dtype='int32')
        shifted_target_inputs = kl.Input(
            shape=(None,), name="shifted_target_inputs", dtype='int32')
        if self.use_lm:
            lm_inputs = kl.Input(shape=(None, self.lm_state_dim), name="lm_inputs", dtype="float32")
        else:
            lm_inputs = None
        if self.use_symbol_statistics:
            symbol_statistics_inputs = kl.Input(shape=(None, 2*self.symbols_number),
                                                name="symbol_stats_inputs", dtype="int32")
        else:
            symbol_statistics_inputs = None
        symbol_outputs, auxiliary_symbol_outputs =\
            self._build_word_network(symbol_inputs, feature_inputs, symbol_statistics_inputs)
        aligned_symbol_outputs = self._build_alignment(
            symbol_outputs, letter_indexes_inputs, dropout=self.dropout)
        aligned_inputs = self._build_alignment(symbol_inputs, letter_indexes_inputs)
        inputs = [symbol_inputs, feature_inputs,
                  letter_indexes_inputs, shifted_target_inputs]
        if self.use_lm:
            inputs.append(lm_inputs)
        if self.use_symbol_statistics:
            inputs.append(symbol_statistics_inputs)
        # буквы входного слова
        # to_concatenate = [aligned_symbol_outputs, tiled_feature_inputs]
        shifted_outputs = self._build_history_network(
            shifted_target_inputs, only_last=test and not self.slow)
        tiled_feature_inputs = self._build_feature_network(
            feature_inputs, kb.shape(shifted_outputs)[1])
        to_decoder = [aligned_symbol_outputs, tiled_feature_inputs, shifted_outputs]
        first_output, initial_decoder_states, final_decoder_states =\
            self._build_decoder_network(to_decoder, aligned_inputs, lm_inputs=lm_inputs)
        if self.step_loss_weight:
            loss = ClassCrossEntropy([STEP_CODE], [self.step_loss_weight])
        else:
            loss = "categorical_crossentropy"
        if self.auxiliary_targets_number > 0:
            loss, outputs, metrics = [loss], [first_output], ["outputs_accuracy"]
            for i, target_name in enumerate(self.auxiliary_targets):
                outputs.append(auxiliary_symbol_outputs[i])
                loss.append("binary_crossentropy" if target_name != "morphemes"
                            else "categorical_crossentropy")
            weights = [1.0] + self.auxiliary_target_weights
        else:
            outputs, metrics, weights = first_output, ["accuracy"], None
        model = Model(inputs, outputs)
        sample_weight_mode = "temporal" if self.auxiliary_targets_number > 0 else None
        model.compile(optimizer=adam(clipnorm=5.0), loss=loss, metrics=["accuracy"],
                      loss_weights=weights, sample_weight_mode=sample_weight_mode)
        to_encoder = ([symbol_inputs, feature_inputs] if self.encode_auxiliary_symbol_outputs
                      else [symbol_inputs])
        if self.use_symbol_statistics:
            to_encoder.append(symbol_statistics_inputs)
        encoder = kb.function(to_encoder + [kb.learning_phase()], [symbol_outputs])
        decoder_inputs = [aligned_symbol_outputs, feature_inputs,
                          shifted_target_inputs, aligned_inputs]
        if self.use_lm:
            decoder_inputs.append(lm_inputs)
        if self.slow:
            decoder = kb.Function(decoder_inputs + [kb.learning_phase()], [first_output])
        else:
            decoder = kb.Function(decoder_inputs + initial_decoder_states + [kb.learning_phase()],
                                  [first_output] + final_decoder_states)
        self.func = kb.Function(inputs + [kb.learning_phase()], [first_output, symbol_outputs] + decoder_inputs)
        return model, encoder, decoder

    def _prepare_for_prediction(self, words, features,
                                buckets_with_indexes, known_answers=None):
        encoded_words = np.array(
            [([BEGIN] + [self.symbol_codes_.get(
                x, UNKNOWN) for x in word] + [END]) for word in words])
        encoded_words_by_buckets = [make_table(encoded_words, indexes, fill_value=PAD)
                                    for length, indexes in buckets_with_indexes]
        # представления входных слов после lstm
        # encoder_outputs_by_buckets[i].shape = (self.models, lengths[i])
        features_by_buckets = [
            np.array([self.extract_features(features[i]) for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        raw_features_by_buckets =\
            [[features[i] for i in indexes] for _, indexes in buckets_with_indexes]
        encoder_args = [encoded_words_by_buckets, features_by_buckets]
        if self.use_symbol_statistics:
            symbol_stats_by_buckets = [self._extract_symbol_data(elem)
                                       for elem in encoded_words_by_buckets]
            encoder_args.append(symbol_stats_by_buckets)
        encoder_outputs_by_buckets = self._predict_encoder_outputs(*encoder_args)

        targets_by_buckets = [self._make_shifted_output(L, len(indexes))
                              for (L, indexes) in buckets_with_indexes]
        if known_answers is not None:
            encoded_answers = np.array([([BEGIN] +
                                         [self.symbol_codes_.get(x, UNKNOWN) for x in word] +
                                         [END]) for word in known_answers])
            encoded_answers_by_buckets = [
                make_table(encoded_answers, indexes, fill_value=PAD)
                for length, indexes in buckets_with_indexes]
        else:
            encoded_answers_by_buckets = [None] * len(buckets_with_indexes)
        inputs = [encoder_outputs_by_buckets, features_by_buckets,
                  targets_by_buckets, encoded_words_by_buckets, raw_features_by_buckets]
        return inputs, encoded_answers_by_buckets

    def predict(self, data, feat_column=2, known_answers=None,
                return_alignment_positions=False, return_log=False,
                log_base=10.0, verbose=0, **kwargs):
        """

        Parameters:
        -----------
            data: list of tuples,
                data = [(lemma, problem_descr), ..]
                problem_descr = {('pos', feat_1, ..., feat_d), (pos, val_1, ... val_d})
        Returns:
        ----------
            answer: a list of strs
        """
        if not getattr(self, "built_", "") == "test":
            self.rebuild_test()
        words, features = [elem[0] for elem in data], [elem[feat_column] for elem in data]
        # output_lengths = [30 for x in words]
        output_lengths = [len(x) + 3 for x in words]
        # output_lengths = [30 for x in words]
        buckets_with_indexes = collect_buckets(output_lengths, max_bucket_length=34)
        inputs, encoded_answers_by_buckets =\
            self._prepare_for_prediction(words, features, buckets_with_indexes, known_answers)
        answer = [[] for _ in words]
        for bucket_number, elem in enumerate(zip(*inputs)):
            if verbose > 0:
                print("Bucket {} of {} predicting".format(
                    bucket_number+1, len(buckets_with_indexes)))
            predict_func = self._predict_on_batch_slow if self.slow else self._predict_on_batch
            bucket_words, bucket_probs = predict_func(
                *elem, known_answers=encoded_answers_by_buckets[bucket_number], **kwargs)
            _, bucket_indexes = buckets_with_indexes[bucket_number]
            for i, curr_words, curr_probs in zip(bucket_indexes, bucket_words, bucket_probs):
                for curr_symbols, curr_prob in zip(curr_words, curr_probs):
                    answer[i].append(self.decode_answer(
                        curr_symbols, curr_prob,
                        return_alignment_positions=return_alignment_positions,
                        return_log=return_log, log_base=log_base))
                scores = np.array([10**(-elem[2]) for elem in answer[i]])
                scores /= np.sum(scores)
                for j, score in enumerate(scores):
                    answer[i][j][2] = score
        return answer

    def _predict_encoder_outputs(self, data, features, symbol_stats=None):
        answer = []
        for i, bucket in enumerate(data):
            if self.encode_auxiliary_symbol_outputs:
                to_encoder = [bucket, features[i]]
            else:
                to_encoder = [bucket]
            if self.use_symbol_statistics:
                to_encoder.append(symbol_stats[i])
            curr_answers = np.array([encoder(to_encoder + [0])[0] for encoder in self.encoders_])
            self._dump = [bucket, features[i], curr_answers]
            answer.append(curr_answers)
        return answer


    def _predict_on_batch(self, symbols, features, target_history, source,
                          source_features, beam_width=1, beam_growth=None,
                          prune_start=0, known_answers=None):
        """

        symbols: np.array of float, shape=(models_number, m, L, H)
            RNN-закодированные символы входной последовательности;
            m -- размер корзины,
            L -- максимально возможная длина ответа,
            H -- размерность рекурсивной нейронной сети.
        :param features:
        :param target_history:
        :return:
        """
        beam_growth = beam_growth or beam_width
        _, m, max_length, H = symbols.shape
        L = 2 * max_length + self.max_length_shift_ + 3
        target_history = np.concatenate(
            [target_history, np.zeros(shape=(m, max(L - target_history.shape[1], 0)) + target_history.shape[2:])], axis=1)
        if known_answers is not None:
            known_answers = np.concatenate(
                [known_answers, np.zeros(shape=(m, max(L - known_answers.shape[1], 0)) + known_answers.shape[2:], dtype=int)], axis=1)
        M = m * beam_width
        # positions[j] --- текущая позиция в symbols[j]
        positions = np.zeros(shape=(M,), dtype=int)
        positions_in_answers = np.zeros(shape=(M,), dtype=int)
        if known_answers is not None:
            current_steps_number = np.zeros(shape=(M,), dtype=int)
            known_answers = np.repeat(known_answers, beam_width, axis=0)
            allowed_steps_number = L - np.count_nonzero(known_answers, axis=1) - 1
        words, probs = [[] for _ in range(M)], [[] for _ in range(M)]
        partial_scores = np.zeros(shape=(M,), dtype=float)
        is_active, active_count = np.zeros(dtype=bool, shape=(M,)), m
        is_completed = np.zeros(dtype=bool, shape=(M,))# закончено ли порождение для текущего слова
        for i in range(0, M, beam_width):
            is_active[i] = True
        # сколько осталось предсказать элементов в каждой группе
        group_beam_widths = [beam_width] * m
        # symbol_inputs = np.zeros(shape=(self.models_number, M, L, H), dtype=float)
        # размножаем признаки и shifted_targets
        features = np.repeat(features, beam_width, axis=0)
        target_history = np.repeat(target_history, beam_width, axis=0)
        source = np.repeat(source, beam_width, axis=0)
        h_states = np.zeros(shape=(M, self.models_number, self.decoder_rnn_size), dtype="float32")
        c_states = np.zeros(shape=(M, self.models_number, self.decoder_rnn_size), dtype="float32")
        if self.use_lm:
            lm_states = np.zeros(shape=(M, self.lm_state_dim), dtype="float32")
            # lm_encoder_states = np.zeros(shape=(2, M, self.lm_encoder_dim), dtype="float32")
            lm_decoder_states = np.zeros(shape=(M, 2, self.lm_decoder_dim), dtype="float32")
            encoded_features = [self.lm_._make_feature_vector(x) for x in source_features]
            encoded_features = np.repeat(encoded_features, beam_width, axis=0)
        for i in range(L):
            # текущие символы
            curr_symbols = symbols[:, np.arange(M) // beam_width, positions,:][:, is_active]
            active_source = source[np.arange(M), positions][is_active]
            args = [curr_symbols[:, :, None], features[is_active],
                    target_history[is_active,max(i-self.output_history,0):i+1],
                    active_source[:, None]]
            if self.use_lm:
                args.append(lm_states[is_active,None])
            args += [h_states[is_active, :], c_states[is_active, :]]
            self._dump.append([np.where(is_active)[0]])
            curr_outputs, new_h_states, new_c_states = self._predict_current_output(*args)
            # active_source = source[np.arange(M), positions][is_active]
            curr_outputs[active_source == END, STEP_CODE] = 0.0
            curr_outputs[active_source == PAD, STEP_CODE] = 0.0
            if i == L-1:
                curr_outputs[:,np.arange(self.symbols_number) != END] = 0.0
            # если текущий символ в
            if known_answers is not None:
                curr_known_answers = known_answers[np.arange(M),positions_in_answers]
                are_steps_allowed = np.min([i - positions_in_answers < allowed_steps_number,
                                            source[np.arange(M), positions] != END,
                                            source[np.arange(M), positions] != PAD,
                                            np.maximum(current_steps_number < self.MAX_STEPS_NUMBER,
                                                       curr_known_answers == END)],
                                           axis=0)
                self._zero_impossible_probs(
                    curr_outputs, curr_known_answers[is_active], are_steps_allowed[is_active])
            hypotheses_by_groups = [[] for _ in range(m)]
            if beam_width == 1:
                curr_output_symbols = np.argmax(curr_outputs, axis=1)
                # for j, curr_probs, index in zip(
                #         np.nonzero(is_active)[0], curr_outputs[:,-1], curr_output_symbols):
                for r, (j, curr_probs, index) in enumerate(zip(
                        np.nonzero(is_active)[0], curr_outputs, curr_output_symbols)):
                    new_score = partial_scores[j] - np.log10(curr_probs[index])
                    hyp = (j, index, new_score, -np.log10(curr_probs[index]),
                           new_h_states[r], new_c_states[r])
                    hypotheses_by_groups[j] = [hyp]
            else:
                curr_best_scores = [np.inf] * m
                for r, (j, curr_probs) in enumerate(zip(np.nonzero(is_active)[0], curr_outputs)):
                    group_index = j // beam_width
                    prev_partial_score = partial_scores[j]
                    curr_probs = -np.log10(curr_probs)   # переходим к логарифмической шкале
                    if np.isinf(curr_best_scores[group_index]):
                        curr_best_scores[group_index] = prev_partial_score + np.min(curr_probs)
                    if i >= prune_start:
                        min_log_prob = curr_best_scores[group_index] - prev_partial_score + self.max_diff
                        min_log_prob = min(-np.log10(self.min_prob), min_log_prob)
                    else:
                        min_log_prob = 5
                    if known_answers is None:
                        possible_indexes = np.where(curr_probs <= min_log_prob)[0]
                    else:
                        possible_indexes = np.where(curr_probs < np.inf)[0]
                    if len(possible_indexes) == 0:
                        possible_indexes = [np.argmin(curr_probs)]
                    for index in possible_indexes:
                        new_score = prev_partial_score + curr_probs[index]
                        hyp = (j, index, new_score, curr_probs[index], new_h_states[r], new_c_states[r])
                        hypotheses_by_groups[group_index].append(hyp)
                    hypotheses_by_groups[group_index] =\
                        sorted(hypotheses_by_groups[group_index], key=lambda x:x[2])[:beam_growth]
            symbol_outputs_indexes = []
            for j, (curr_hypotheses, group_beam_width) in\
                    enumerate(zip(hypotheses_by_groups, group_beam_widths)):
                if group_beam_width == 0:
                    continue
                curr_hypotheses = sorted(
                    map(list, curr_hypotheses), key=(lambda x:x[2]))[:group_beam_width]
                group_start = j * beam_width
                free_indexes = np.where(np.logical_not(
                    is_completed[group_start:group_start+beam_width]))[0]
                free_indexes = free_indexes[:len(curr_hypotheses)] + group_start
                is_active[group_start:group_start+beam_width] = False
                extend_history(words, curr_hypotheses, free_indexes, pos=1)
                extend_history(probs, curr_hypotheses, free_indexes, pos=3)
                extend_history(partial_scores, curr_hypotheses, free_indexes, pos=2, func="change")
                extend_history(positions, curr_hypotheses, free_indexes, pos=1,
                               func=(lambda x, y: x+int(y == STEP_CODE)))
                extend_history(positions_in_answers, curr_hypotheses, free_indexes,
                               pos=1, func=(lambda x, y: x + int(y != STEP_CODE)))
                if self.use_lm:
                    for r, hyp in zip(free_indexes, curr_hypotheses):
                        symbol_outputs_indexes.append((hyp[0], r, hyp[1]))
                if known_answers is not None:
                    extend_history(current_steps_number, curr_hypotheses, free_indexes,
                                   pos=1, func=(lambda x, y: (0 if y != STEP_CODE else x+1)))
                extend_history(h_states, curr_hypotheses, free_indexes, pos=4, func="change")
                extend_history(c_states, curr_hypotheses, free_indexes, pos=5, func="change")
                if i < L-1:
                    extend_history(target_history, curr_hypotheses, free_indexes, pos=1,
                                   func="set", h=self.output_history, position=i+1)
                for r, index in enumerate(free_indexes):
                    if r > len(curr_hypotheses):
                        break
                    is_active[index] = (curr_hypotheses[r][1] != END)
                    is_completed[index] = not(is_active[index])
            # нужно пересчитать состояния языковой модели
            if self.use_lm and len(symbol_outputs_indexes) > 0:
                indexes_to_update_states = []
                new_lm_states = np.zeros(shape=(M, self.lm_state_dim), dtype="float32")
                # lm_encoder_states = np.zeros(shape=(2, M, self.lm_encoder_dim), dtype="float32")
                new_lm_decoder_states = np.zeros(shape=(M, 2, self.lm_decoder_dim), dtype="float32")
                for old_index, index, letter in symbol_outputs_indexes:
                    new_lm_states[index] = lm_states[old_index]
                    new_lm_decoder_states[index] = lm_decoder_states[old_index]
                    if letter != STEP_CODE:
                        indexes_to_update_states.append(index)
                if len(indexes_to_update_states) > 0:
                    curr_features = [encoded_features[index] for index in indexes_to_update_states]
                    lm_histories = []
                    for index in indexes_to_update_states:
                        word = [s for s in words[index] if s != STEP_CODE]
                        encoded_word = [self.recodings_for_lm_[x] for x in word[-self.lm_.history:]]
                        encoded_word = [0] * (self.lm_.history - len(encoded_word)) + encoded_word
                        lm_histories.append(encoded_word)
                    to_step = [np.array(lm_histories), np.array(curr_features),
                               new_lm_decoder_states[indexes_to_update_states,0],
                               new_lm_decoder_states[indexes_to_update_states,1]]
                    self._dump[-1].append((indexes_to_update_states, np.array(lm_histories)))
                    from_step = self.lm_step_func(to_step + [0])
                    for j, (index, state, c_state, h_state) in\
                            enumerate(zip(indexes_to_update_states, *from_step)):
                        new_lm_states[index] = state
                        new_lm_decoder_states[index] = [c_state, h_state]
                else:
                    self._dump[-1].append(([], []))
                lm_states, lm_decoder_states = new_lm_states, new_lm_decoder_states

            if not any(is_active):
                break
        # здесь нужно переделать words, probs в список
        words_by_groups, probs_by_groups = [], []
        for group_start in range(0, M, beam_width):
            # приводим к списку, чтобы иметь возможность сортировать
            active_indexes_for_group = list(np.where(is_completed[group_start:group_start+beam_width])[0])
            group_scores = partial_scores[group_start:group_start+beam_width]
            active_indexes_for_group.sort(key=(lambda i: group_scores[i]))
            words_by_groups.append([words[group_start+i] for i in active_indexes_for_group])
            probs_by_groups.append([probs[group_start+i] for i in active_indexes_for_group])
        return words_by_groups, probs_by_groups

    def _predict_on_batch_slow(self, symbols, features, target_history, source,
                               source_features, beam_width=1, beam_growth=None,
                               prune_start=0, known_answers=None):
        """

        symbols: np.array of float, shape=(models_number, m, L, H)
            RNN-закодированные символы входной последовательности;
            m -- размер корзины,
            L -- максимально возможная длина ответа,
            H -- размерность рекурсивной нейронной сети.
        :param features:
        :param target_history:
        :return:
        """
        beam_growth = beam_growth or beam_width
        _, m, max_length, H = symbols.shape
        L = 2 * max_length + self.max_length_shift_ + 3
        target_history = np.concatenate(
            [target_history, np.zeros(shape=(m, max(L - target_history.shape[1], 0)) + target_history.shape[2:])], axis=1)
        M = m * beam_width
        # positions[j] --- текущая позиция в symbols[j]
        positions = np.zeros(shape=(M,), dtype=int)
        positions_in_answers = np.zeros(shape=(M,), dtype=int)
        if known_answers is not None:
            current_steps_number = np.zeros(shape=(M,), dtype=int)
            known_answers = np.repeat(known_answers, beam_width, axis=0)
            allowed_steps_number = L - np.count_nonzero(known_answers, axis=1) - 1
        words, probs = [[] for _ in range(M)], [[] for _ in range(M)]
        partial_scores = np.zeros(shape=(M,), dtype=float)
        is_active, active_count = np.zeros(dtype=bool, shape=(M,)), m
        is_completed = np.zeros(dtype=bool, shape=(M,))# закончено ли порождение для текущего слова
        for i in range(0, M, beam_width):
            is_active[i] = True
        # сколько осталось предсказать элементов в каждой группе
        group_beam_widths = [beam_width] * m
        # symbol_inputs = np.zeros(shape=(self.models_number, M, L, H), dtype=float)
        # размножаем признаки и shifted_targets
        features = np.repeat(features, beam_width, axis=0)
        target_history = np.repeat(target_history, beam_width, axis=0)
        source = np.repeat(source, beam_width, axis=0)
        source_for_decoder = np.zeros(shape=(M, L)+source.shape[2:], dtype=source.dtype)
        shape = (M, self.models_number, L, H)
        symbols_for_decoder = np.zeros(shape=shape, dtype="float32")
        symbols_for_decoder = symbols_for_decoder.reshape((M, self.models_number) + symbols_for_decoder.shape[2:])
        # h_states = np.zeros(shape=(M, self.models_number, self.decoder_rnn_size), dtype="float32")
        # c_states = np.zeros(shape=(M, self.models_number, self.decoder_rnn_size), dtype="float32")
        if self.use_lm:
            lm_states = np.zeros(shape=(M, self.lm_state_dim), dtype="float32")
            # lm_encoder_states = np.zeros(shape=(2, M, self.lm_encoder_dim), dtype="float32")
            lm_decoder_states = np.zeros(shape=(M, 2, self.lm_decoder_dim), dtype="float32")
            encoded_features = [self.lm_._make_feature_vector(x) for x in source_features]
            encoded_features = np.repeat(encoded_features, beam_width, axis=0)
        for i in range(L):
            # текущие символы
            source_for_decoder[:,i] = source[np.arange(M), positions]
            for j in range(self.models_number):
                symbols_for_decoder[np.arange(M),j,i] = symbols[j][np.arange(M) // beam_width, positions]
            curr_symbols = symbols_for_decoder[is_active, :, :i+1,:]
            active_source = source_for_decoder[is_active, :i+1]
            args = [curr_symbols, features[is_active], target_history[is_active,:i+1], active_source]
            if self.use_lm:
                args.append(lm_states[is_active, :i+1])
            curr_outputs = self._predict_current_output_slow(*args)
            # active_source = source[np.arange(M), positions][is_active]
            curr_outputs[active_source[:,-1] == END, STEP_CODE] = 0.0
            curr_outputs[active_source[:,-1] == PAD, STEP_CODE] = 0.0
            if i == L-1:
                curr_outputs[:,np.arange(self.symbols_number) != END] = 0.0
            # если текущий символ в
            if known_answers is not None:
                curr_known_answers = known_answers[np.arange(M),positions_in_answers]
                are_steps_allowed = np.min([i - positions_in_answers < allowed_steps_number,
                                            source[np.arange(M), positions] != END,
                                            source[np.arange(M), positions] != PAD,
                                            np.maximum(current_steps_number < self.MAX_STEPS_NUMBER,
                                                       curr_known_answers == END)],
                                           axis=0)
                self._zero_impossible_probs(
                    curr_outputs, curr_known_answers[is_active], are_steps_allowed[is_active])
            hypotheses_by_groups = [[] for _ in range(m)]
            if beam_width == 1:
                curr_output_symbols = np.argmax(curr_outputs, axis=1)
                # for j, curr_probs, index in zip(
                #         np.nonzero(is_active)[0], curr_outputs[:,-1], curr_output_symbols):
                for r, (j, curr_probs, index) in enumerate(zip(
                        np.nonzero(is_active)[0], curr_outputs, curr_output_symbols)):
                    new_score = partial_scores[j] - np.log10(curr_probs[index])
                    hyp = (j, index, new_score, -np.log10(curr_probs[index]))
                    hypotheses_by_groups[j] = [hyp]
            else:
                curr_best_scores = [np.inf] * m
                for r, (j, curr_probs) in enumerate(zip(np.nonzero(is_active)[0], curr_outputs)):
                    group_index = j // beam_width
                    prev_partial_score = partial_scores[j]
                    curr_probs = -np.log10(curr_probs)   # переходим к логарифмической шкале
                    if np.isinf(curr_best_scores[group_index]):
                        curr_best_scores[group_index] = prev_partial_score + np.min(curr_probs)
                    if i >= prune_start:
                        min_log_prob = curr_best_scores[group_index] - prev_partial_score + self.max_diff
                        min_log_prob = min(-np.log10(self.min_prob), min_log_prob)
                    else:
                        min_log_prob = 5
                    if known_answers is None:
                        possible_indexes = np.where(curr_probs <= min_log_prob)[0]
                    else:
                        possible_indexes = np.where(curr_probs < np.inf)[0]
                    if len(possible_indexes) == 0:
                        possible_indexes = [np.argmin(curr_probs)]
                    for index in possible_indexes:
                        new_score = prev_partial_score + curr_probs[index]
                        hyp = (j, index, new_score, curr_probs[index])
                        hypotheses_by_groups[group_index].append(hyp)
                    hypotheses_by_groups[group_index] =\
                        sorted(hypotheses_by_groups[group_index], key=lambda x:x[2])[:beam_growth]
            symbol_outputs_indexes = []
            for j, (curr_hypotheses, group_beam_width) in\
                    enumerate(zip(hypotheses_by_groups, group_beam_widths)):
                if group_beam_width == 0:
                    continue
                curr_hypotheses = sorted(
                    map(list, curr_hypotheses), key=(lambda x:x[2]))[:group_beam_width]
                group_start = j * beam_width
                free_indexes = np.where(np.logical_not(
                    is_completed[group_start:group_start+beam_width]))[0]
                free_indexes = free_indexes[:len(curr_hypotheses)] + group_start
                is_active[group_start:group_start+beam_width] = False
                extend_history(words, curr_hypotheses, free_indexes, pos=1)
                extend_history(probs, curr_hypotheses, free_indexes, pos=3)
                extend_history(partial_scores, curr_hypotheses, free_indexes, pos=2, func="change")
                extend_history(positions, curr_hypotheses, free_indexes, pos=1,
                               func=(lambda x, y: x+int(y == STEP_CODE)))
                extend_history(positions_in_answers, curr_hypotheses, free_indexes,
                               pos=1, func=(lambda x, y: x + int(y != STEP_CODE)))
                extend_history(source_for_decoder, curr_hypotheses, free_indexes, func="copy")
                extend_history(symbols_for_decoder, curr_hypotheses, free_indexes, func="copy")
                if self.use_lm:
                    for r, hyp in zip(free_indexes, curr_hypotheses):
                        symbol_outputs_indexes.append((hyp[0], r, hyp[1]))
                if known_answers is not None:
                    extend_history(current_steps_number, curr_hypotheses, free_indexes,
                                   pos=1, func=(lambda x, y: (0 if y != STEP_CODE else x+1)))
                if i < L-1:
                    extend_history(target_history, curr_hypotheses, free_indexes, pos=1,
                                   func="set", h=self.output_history, position=i+1)
                for r, index in enumerate(free_indexes):
                    if r > len(curr_hypotheses):
                        break
                    is_active[index] = (curr_hypotheses[r][1] != END)
                    is_completed[index] = not(is_active[index])
            # нужно пересчитать состояния языковой модели
            if self.use_lm and len(symbol_outputs_indexes) > 0:
                indexes_to_update_states = []
                new_lm_states = np.zeros(shape=(M, self.lm_state_dim), dtype="float32")
                # lm_encoder_states = np.zeros(shape=(2, M, self.lm_encoder_dim), dtype="float32")
                new_lm_decoder_states = np.zeros(shape=(M, 2, self.lm_decoder_dim), dtype="float32")
                for old_index, index, letter in symbol_outputs_indexes:
                    new_lm_states[index] = lm_states[old_index]
                    new_lm_decoder_states[index] = lm_decoder_states[old_index]
                    if letter != STEP_CODE:
                        indexes_to_update_states.append(index)
                if len(indexes_to_update_states) > 0:
                    curr_features = [encoded_features[index] for index in indexes_to_update_states]
                    lm_histories = []
                    for index in indexes_to_update_states:
                        word = words[index][-self.lm_.history:]
                        encoded_word = [self.recodings_for_lm_[x] for x in word]
                        encoded_word = [0] * (self.lm_.history - len(encoded_word)) + encoded_word
                        lm_histories.append(encoded_word)
                    to_step = [np.array(lm_histories), np.array(curr_features),
                               new_lm_decoder_states[indexes_to_update_states,0],
                               new_lm_decoder_states[indexes_to_update_states,1]]
                    from_step = self.lm_step_func(to_step + [0])
                    for j, (index, state, c_state, h_state) in\
                            enumerate(zip(indexes_to_update_states, *from_step)):
                        new_lm_states[index] = state
                        new_lm_decoder_states[index] = [c_state, h_state]
                lm_states, lm_decoder_states = new_lm_states, new_lm_decoder_states
            if not any(is_active):
                break
        # здесь нужно переделать words, probs в список
        words_by_groups, probs_by_groups = [], []
        for group_start in range(0, M, beam_width):
            # приводим к списку, чтобы иметь возможность сортировать
            active_indexes_for_group = list(np.where(is_completed[group_start:group_start+beam_width])[0])
            group_scores = partial_scores[group_start:group_start+beam_width]
            active_indexes_for_group.sort(key=(lambda i: group_scores[i]))
            words_by_groups.append([words[group_start+i] for i in active_indexes_for_group])
            probs_by_groups.append([probs[group_start+i] for i in active_indexes_for_group])
        return words_by_groups, probs_by_groups

    def _predict_current_output(self, *args):
        answer = [[], [], []]
        if not hasattr(self, "_dump"):
            self._dump = []
        for i, decoder in enumerate(self.decoders_):
            curr_args = [args[0][i]] + list(args[1:-2]) + [args[-2][:,i], args[-1][:,i]]
            curr_answer = decoder(curr_args + [0])
            for elem, to_append in zip(answer, curr_answer):
                elem.append(to_append)
            self._dump[-1].extend((curr_args, curr_answer))
        answer[0] = np.mean(answer[0], axis=0)[:,0]
        answer[1] = np.transpose(answer[1], axes=(1, 0, 2))
        answer[2] = np.transpose(answer[2], axes=(1, 0, 2))
        return answer

    def _predict_current_output_slow(self, *args):
        answer = []
        # if not hasattr(self, "_dump"):
        #     self._dump = []
        for i, decoder in enumerate(self.decoders_):
            curr_args = [args[0][:,i]] + list(args[1:])
            curr_answer = decoder(curr_args + [0])[-1]
            answer.append(curr_answer[:,-1])
            # self._dump.append((curr_args, curr_answer))
        answer = np.mean(answer, axis=0)
        return answer

    def _zero_impossible_probs(self, curr_output, known_answers, are_steps_allowed):
        # если знаем символ, разрешаем только его и STEP_CODE
        M = curr_output.shape[0]
        mask = np.ones(shape=(M, self.symbols_number), dtype=bool)
        mask[np.arange(M), known_answers] = False
        mask[are_steps_allowed, STEP_CODE] = False
        # curr_output[:,-1][mask] = 0.0
        curr_output[mask] = 0.0
        return curr_output
        # for r, j in enumerate(np.nonzero(is_active)[0]):
        #     mask = np.ones(shape=(self.symbols_number,), dtype=bool)
        #     pos_in_answer = positions_in_answers[j]
        #     allowed_indexes = [known_answers[j][pos_in_answer]]
        #     if (known_answers[j][pos_in_answer] == END or
        #             (current_steps_number[j] <= self.MAX_STEPS_NUMBER and
        #              words[j].count(STEP_CODE) <= allowed_steps_number[j])):
        #         allowed_indexes.append(STEP_CODE)
        #     mask[allowed_indexes] = False
        #     curr_outputs[r,-1,mask] = 0.0

    def decode_answer(self, symbol_indexes, probs, return_alignment_positions=False,
                      return_log=True, log_base=10.0, are_log_probs=True):
        curr_pos, curr_prob_log, probs_to_return = 0, 0.0, []
        word, alignment_positions = "", []
        if symbol_indexes[0] != BEGIN:
            symbol_indexes = [BEGIN] + symbol_indexes
            probs = [0.0] + probs
        if symbol_indexes[-1] != END:
            symbol_indexes.append(END)
            probs.append(0.0)
        for index, prob in zip(symbol_indexes, probs):
            if not are_log_probs:
                prob = -np.log10(prob)
            curr_prob_log += prob
            if index == STEP_CODE:
                curr_pos += 1
                continue
            alignment_positions.append(curr_pos)
            probs_to_return.append(curr_prob_log)
            curr_prob_log = 0.0
            # разобраться с UNKNOWN!
            if self.symbols_[index] not in AUXILIARY:
                word += self.symbols_[index]
        answer = [word, probs_to_return, np.sum(probs_to_return)]
        if return_alignment_positions:
            answer += (alignment_positions,)
        if not return_log:
            answer[1] = [10.0**(-x) for x in probs_to_return]
        else:
            answer[1] = [x / np.log10(log_base) for x in probs_to_return]
        return answer

    def evaluate_lm(self, data, processed_data):
        lm_states, X_test = self.lm_.predict_states_batch(data, mode=self.use_lm)
        X_, indexes = self.lm_.transform(data, buckets_number=1)
        # X = X[0]
        _, features = processed_data[:2]
        X = np.zeros_like(X_[0][0], dtype=int)
        for i, (word, _) in enumerate(data):
            X[i, 0] = BEGIN
            X[i, 1:1+len(word)] = [self.symbol_codes_[x] for x in word]
            X[i, len(word) + 1] = END
        # lm_states = self.lm_._logit_func_([X[0], X[1], 0])[0]
        self.lm_.rebuild()
        # for bucket, curr_indexes in zip(X, indexes):
        m = len(data)
        h_states = np.zeros(shape=(m, self.lm_.decoder_rnn_size), dtype=float)
        c_states = np.zeros(shape=(m, self.lm_.decoder_rnn_size), dtype=float)
        # for i in range(X[0].shape[1]):
        for i in range(X.shape[1]):
            curr_data_ = X_[0][0][:, max(i - 4, 0): i + 1]
            curr_data = X[:, max(i - 4, 0): i + 1]
            curr_data = np.array([[self.recodings_for_lm_[index] for index in elem] for elem in curr_data])
            if i < 4:
                curr_data = np.concatenate([np.zeros(shape=(m, 4 - i)), curr_data], axis=1)
                curr_data_ = np.concatenate([np.zeros(shape=(m, 4 - i)), curr_data_], axis=1)
            recoded_features = X_[0][1]
            for_step = [curr_data_, recoded_features, h_states, c_states, 0]
            from_step = self.lm_step_func(for_step)
            curr_answer, h_states, c_states = from_step
            diff = np.max(np.abs(lm_states[:,i] - curr_answer[:,0]))
            if diff > 0:
                print(i, "{:.2f}".format(diff))
        return

    def check_histories(self, i, indexes, bucket_data, history_indexes, histories):
        return

    def evaluate(self, data, alignment_data):
        M = len(data)
        self.aligner.align(alignment_data, to_fit=True)
        data_by_buckets, indexes_by_buckets = self._preprocess(data, alignments=None, bucket_size=20, to_fit=False)
        for bucket_data, bucket_indexes in zip(data_by_buckets, indexes_by_buckets):
            # self.rebuild_test()
            if self.use_lm:
                self.evaluate_lm([data[i][1:] for i in bucket_indexes], bucket_data)
                # continue
            m = len(bucket_indexes)
            bucket_data = list(bucket_data)
            target_index = -1 - self.auxiliary_targets_number
            data_to_evaluate, answer = bucket_data[:target_index], bucket_data[target_index]
            encoded_data = self.encoders_[0](data_to_evaluate[:1] + [0])[0]
            aligned_encoded_data = encoded_data[np.arange(m)[:,None], data_to_evaluate[2]]
            aligned_source = data_to_evaluate[0][np.arange(m)[:,None], data_to_evaluate[2]]
            shifted_targets = self._make_shifted_output(answer.shape[1], m, targets=answer)
            to_decoder = [aligned_encoded_data, data_to_evaluate[1], shifted_targets, aligned_source]
            states = [np.zeros(shape=(m, self.decoder_rnn_size), dtype=float),
                      np.zeros(shape=(m, self.decoder_rnn_size), dtype=float)]
            decoder_answers = []
            # predictions = self.models_[0].predict(data_to_evaluate)
            predictions = self.func(data_to_evaluate + [0])
            # if isinstance(predictions, list):
            #     predictions = predictions[0]
            labels = np.argmax(predictions[0], axis=-1)
            for i in range(min(labels.shape[1], answer.shape[1])):
                diff = (labels[:,:i+1] != answer[:, :i+1]).astype("int")
                if max(diff[:,-1]):
                    print(i, ",".join(str(x) for x in np.max(diff, axis=-1)))
            print("")
            # continue
        # for i in range(25):
        #     # curr_features = np.repeat(np.array(data_to_evaluate[1])[:,None], i+1, axis=1)
        #     curr_to_decoder = [aligned_encoded_data[:,:i+1], data_to_evaluate[1],
        #                        shifted_targets[:, :i+1], aligned_source[:, :i+1]]
        #     curr_predictions = self.decoders_[0](curr_to_decoder + [0])
        #     decoder_answers.append(curr_predictions)
        #     labels = np.argmax(curr_predictions[0], axis=-1)
        #     diff = (labels != answer[:, :i+1]).astype("int")
        #     x = 2
        #
        #     print(",".join(str(x) for x in np.max(diff, axis=1)))
        #
        #     for j, (elem, curr_elem) in enumerate(zip(predictions[2:], curr_to_decoder)):
        #         if j != 1:
        #             elem = elem[:,:i+1]
        #         diff = (elem != curr_elem).astype("int")
        #         axis = tuple(np.arange(1, diff.ndim))
        #         print(",".join(str(x) for x in np.max(diff, axis=axis)))
        #     print("")
        # return
        # # print(labels[:,-1])
        # # print(answer[:,i])
        # print("")
            curr_data = [data[index] for index in bucket_indexes]
            self.predict(curr_data)
            m = min(data_to_evaluate[0].shape[1], self._dump[0].shape[1])
            print(np.max(np.abs(data_to_evaluate[0][:,:m] - self._dump[0][:,:m])))
            print(np.max(np.abs(data_to_evaluate[1] - self._dump[1])))
            m = min(encoded_data.shape[1], self._dump[2].shape[2])
            diff = encoded_data[:,:m] - self._dump[2][0,:,:m]
            diff *= (data_to_evaluate[0][:,:m,None] > 0).astype("int")
            print(np.max(np.abs(diff)))
            for i in range(25):
                indexes, args, curr_predictions, lm_histories = self._dump[i+3]
                labels = np.argmax(curr_predictions[0][:,-1], axis=-1)
                diff = (labels != answer[indexes, i]).astype("int")
                print(i)
                print(",".join(str(x) for x in diff))
                for j, (elem, curr_elem) in enumerate(zip(predictions[2:], args)):
                    elem = elem[indexes]
                    if j in [0, 3, 4]:
                        elem, curr_elem = elem[:,i], curr_elem[:,0]
                    elif j == 2:
                        elem = elem[:,max(i-2,0):i+1]
                    elif j != 1:
                        elem = elem[:,:i+1]
                    if j not in [0, 4]:
                        diff = (elem != curr_elem).astype("int")
                    else:
                        diff = np.abs(elem - curr_elem)
                    axis = tuple(np.arange(1, diff.ndim))
                    print(",".join(str(x) for x in np.max(diff, axis=axis)))
                self.check_histories(i, indexes, bucket_data, *lm_histories)
                print("")
            # positions, histories = [0] * m, np.array([[1] for _ in range(m)])
        # for i in range(1):
        #     curr_to_decoder = [encoded_data[np.arange(m),positions,None], data_to_evaluate[1],
        #                        histories[:, :i+1], data_to_evaluate[0][:, :i+1]]
        #     curr_predictions = self.decoders_[0](curr_to_decoder + states + [0])
        #     for x, y in zip(decoder_answers[i], curr_predictions):
        #         if x.ndim > 2:
        #             x = x[:, i:i+1]
        #         print(np.max(np.abs(x[:, i:i+1] - y)))
        # predictions = self.decoders_[0](to_decoder + states + [0])[0]
        # shifted_targets = self._model_funcs[0](data_to_evaluate + [0])[:1]
        # self.rebuild_test()
        # test_shifted_targets = self._make_shifted_output(answer.shape[1], len(data), targets=answer)
        # print(shifted_targets.shape, test_shifted_targets.shape)
        # diff = np.max(np.abs(shifted_targets[:,0]-test_shifted_targets[:,:1]))
        # for i, elem in enumerate(diff):
        #     print(i, "{:.8f}".format(elem))
        # labels = np.argmax(predictions, axis=-1)
        # print(np.sum(labels != answer).astype("int"))
        return

    # def test_prediction(self, data, alignments=None):
    #     # as for now, we cannot load aligner, therefore have to train it
    #     data_by_buckets, buckets_indexes = self._preprocess(data, alignments, to_fit=False)
    #     predictions, answer = [None] * len(data), [None] * len(data)
    #     for curr_indexes, elem in zip(buckets_indexes, data_by_buckets):
    #         curr_predictions = self.models_[0].predict(list(elem[:-1]))
    #         for index, outputs, corr_outputs in zip(curr_indexes, curr_predictions, elem[-1]):
    #             predictions[index] = outputs
    #             curr_answer = []
    #             for j, probs in enumerate(outputs):
    #                 letter_code = corr_outputs[j]
    #                 if letter_code == END:
    #                     break
    #                 curr_answer.append((self.symbols_[letter_code], probs[letter_code]))
    #             answer[index] = curr_answer
    #     return predictions, answer

def predict_missed_answers(test_data, answers, inflector, **kwargs):
    indexes = [i for i, x in enumerate(test_data)
               if x[1] not in [elem[0] for elem in answers[i]]]
    data_with_missed_answers = [test_data[i] for i in indexes]
    known_answers = [elem[1] for elem in data_with_missed_answers]
    scores = inflector.predict(data_with_missed_answers, known_answers=known_answers, **kwargs)
    answer = [(index, elem[0]) for index, elem in zip(indexes, scores)]
    return answer


