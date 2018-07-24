"""
Эксперименты с языковыми моделями
"""
import sys
import os
import getopt
from collections import defaultdict
import bisect
import copy
import itertools
import inspect
import json
# import ujson as json

import numpy as np
np.set_printoptions(precision=3)

import keras
import keras.optimizers as ko
import keras.backend as kb
import keras.layers as kl
from keras.models import Model
import keras.callbacks as kcall
from keras.callbacks import EarlyStopping, ModelCheckpoint

from .common import *
from .common import generate_data
from .vocabulary import Vocabulary, vocabulary_from_json
from .cells import History, AttentionCell, attention_func

def to_one_hot(indices, num_classes):
    """
    Theano implementation for numpy arrays

    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))

def read_input(infile, label_field=None, max_num=-1):
    answer = []
    feats_column = 2 if label_field is None else 1
    with open(infile, "r", encoding="utf8") as fin:
        for i, line in enumerate(fin):
            if i == max_num:
                break
            line = line.strip()
            if line == "":
                continue
            splitted = line.split()
            curr_elem = [splitted[2]]
            feats = splitted[feats_column] if len(splitted) > feats_column else ""
            feats = [x.split("=") for x in feats.split(",")]
            feats = {x[0]: x[1] for x in feats}
            if label_field is not None:
                label = feats.pop(label_field, None)
            else:
                label = splitted[1] if len(splitted) > 1 else None
            if label is not None:
                curr_elem.append(label)
                curr_elem.append(feats)
            answer.append(curr_elem)
    return answer


def make_bucket_indexes(lengths, buckets_number=None,
                        bucket_size=None, join_buckets=True):
    if buckets_number is None and bucket_size is None:
        raise ValueError("Either buckets_number or bucket_size should be given")
    indexes = np.argsort(lengths)
    lengths = sorted(lengths)
    m = len(lengths)
    if buckets_number is not None:
        level_indexes = [m * (i+1) // buckets_number for i in range(buckets_number)]
    else:
        level_indexes = [min(start+bucket_size, m) for start in range(0, m, bucket_size)]
    if join_buckets:
        new_level_indexes = []
        for i, index in enumerate(level_indexes[:-1]):
            if lengths[index-1] < lengths[level_indexes[i+1]-1]:
                new_level_indexes.append(index)
        level_indexes = new_level_indexes + [m]
    bucket_indexes =  [indexes[start:end] for start, end in
                       zip([0] + level_indexes[:-1], level_indexes)]
    bucket_lengths = [lengths[i-1] for i in level_indexes]
    return bucket_indexes, bucket_lengths

class NeuralLM:

    def __init__(self, reverse=False, min_symbol_count=1,
                 batch_size=32, nepochs=20, validation_split=0.2,
                 use_label=False, use_feats=False, use_full_tags=False,
                 history=1, use_attention=False, attention_activation="concatenate",
                 use_attention_bias=False,
                 use_embeddings=False, embeddings_size=16,
                 feature_embedding_layers=False, feature_embeddings_size=32,
                 rnn="lstm", encoder_rnn_size=64,
                 decoder_rnn_size=64, dense_output_size=32,
                 embeddings_dropout=0.0, encoder_dropout=0.0, decoder_dropout=0.0,
                 random_state=187, verbose=1, callbacks=None,
                 # use_custom_callback=False
                 ):
        self.reverse = reverse
        self.min_symbol_count = min_symbol_count
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.use_label = use_label
        self.use_feats = use_feats
        self.use_full_tags = use_full_tags
        self.history = history
        self.use_attention = use_attention
        self.attention_activation = attention_activation
        self.use_attention_bias = use_attention_bias
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.feature_embedding_layers = feature_embedding_layers
        self.feature_embeddings_size = feature_embeddings_size
        self.rnn = rnn
        self.encoder_rnn_size = encoder_rnn_size
        self.decoder_rnn_size = decoder_rnn_size
        self.dense_output_size = dense_output_size
        # self.dropout = dropout
        self.embeddings_dropout = embeddings_dropout
        self.encoder_dropout = encoder_dropout
        self.decoder_dropout = decoder_dropout
        self.random_state = random_state
        self.verbose = verbose
        # if self.use_custom_callback:
        #     self.callbacks.append(CustomCallback())
        self.initialize(callbacks)

    def initialize(self, callbacks=None):
        if isinstance(self.rnn, str):
            self.rnn = getattr(kl, self.rnn.upper())
        if self.rnn not in [kl.GRU, kl.LSTM]:
            raise ValueError("Unknown recurrent network: {}".format(self.rnn))
        callbacks = callbacks or dict()
        self.callbacks = [getattr(kcall, key)(**params) for key, params in callbacks.items()]

    def to_json(self, outfile, model_file):
        info = dict()
        # model_file = os.path.abspath(model_file)
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(NeuralLM, attr, None), property) or
                    isinstance(val, Vocabulary) or attr.isupper() or
                    attr in ["callbacks", "model_", "step_func_"]):
                info[attr] = val
            elif isinstance(val, Vocabulary):
                info[attr] = val.jsonize()
            elif attr == "model_":
                info["dump_file"] = model_file
                self.model_.save_weights(model_file)
            elif attr == "callbacks":
                callback_info = dict()
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        callback_info["EarlyStopping"] = \
                            {key: getattr(callback, key) for key in ["patience", "monitor", "min_delta"]}
                info["callbacks"] = callback_info
        with open(outfile, "w", encoding="utf8") as fout:
            json.dump(info, fout)

    def make_features(self, X):
        """
        :param X: list of lists,
            создаёт словарь на корпусе X = [x_1, ..., x_m]
            x_i = [w_i, (c_i, (feats_i)) ], где
                w: str, строка из корпуса
                c: str(optional), класс строки (например, часть речи)
                feats: dict(optional), feats = {f_1: v_1, ..., f_k: v_k},
                    набор пар признак-значение (например, значения грамматических категорий)
        :return: self, обученная модель
        """
        # первый проход: определяем длины строк, извлекаем словарь символов и признаков
        labels, features, tags = set(), set(), set()
        for elem in X:
            if len(elem) > 1:
                if isinstance(elem[1], list):
                    label, feats, tag = elem[1][0], elem[1][1:], tuple(elem[1])
                else:
                    label = elem[1]
                    if len(elem) > 1:
                        feats, tag = elem[2], (label,) + tuple(elem[2])
                    else:
                        feats, tag = None, elem[1]
                labels.add(label)
                if self.use_feats and feats is not None:
                    if isinstance(feats, dict):
                        for feature, value in feats.items():
                            features.add("_".join([label, feature, value]))
                    else:
                        features.update({"{}_{}".format(label, elem) for elem in feats})
                tags.add(tuple(tag))
        # создаём словари нужного размера
        self.labels_ = AUXILIARY + sorted(labels) + sorted(features)
        self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        if self.use_full_tags:
            self.tags_ = sorted(tags)
            self.tag_codes_ = {x: i for i, x in enumerate(self.tags_)}
        return self

    @property
    def symbols_number_(self):
        return self.vocabulary_.symbols_number_

    @property
    def labels_number_(self):
        if self.use_label:
            return len(self.labels_) + len(self.tags_) * int(self.use_full_tags)
        else:
            return 0

    def _make_word_vector(self, word, bucket_length=None, symbols_has_features=False):
        """
        :param word:
        :param pad:
        :return:
        """
        m = len(word)
        if bucket_length is None:
            bucket_length = m + 2
        answer = np.full(shape=(bucket_length,), fill_value=PAD, dtype="int32")
        answer[0], answer[m+1] = BEGIN, END
        for i, x in enumerate(word, 1):
            answer[i] = self.vocabulary_.toidx(x)
        return answer

    def _make_feature_vector(self, label, feats=None):
        if isinstance(label, list):
            label, feats = label[0], label[1:]
        answer = np.zeros(shape=(self.labels_number_,))
        label_code = self.label_codes_.get(label, UNKNOWN)
        answer[label_code] = 1
        if label_code != UNKNOWN:
            if isinstance(feats, dict):
                feats = ["{}_{}_{}".format(label, *elem) for elem in feats]
            else:
                feats = ["{}_{}".format(label, elem) for elem in feats]
            for feature in feats:
                feature_code = self.label_codes_.get(feature)
                if self.use_feats and feature_code is not None:
                    answer[feature_code] = 1
        return answer

    def _get_symbol_features_codes(self, symbol, feats):
        symbol_code = self.vocabulary_.get_feature_code(symbol)
        answer = [symbol_code]
        if symbol_code == UNKNOWN:
            return answer
        for feature, value in feats:
            if feature != "token":
                feature_repr = "{}_{}_{}".format(symbol, feature, value)
                symbol_code = self.vocabulary_.get_feature_code(feature_repr)
            else:
                symbol_code = self.vocabulary_.get_token_code(value)
            if symbol_code is not None:
                answer.append(symbol_code)
        return answer

    def _make_bucket_data(self, lemmas, bucket_length, bucket_indexes):
        bucket_size = len(bucket_indexes)
        bucket_data = np.full(shape=(bucket_size, bucket_length), fill_value=PAD, dtype=int)
        # заполняем закодированными символами
        bucket_data[:,0] = BEGIN
        for j, i in enumerate(bucket_indexes):
            lemma = lemmas[i]
            bucket_data[j,1:1+len(lemma)] = [self.vocabulary_.toidx(x) for x in lemma]
            bucket_data[j,1+len(lemma)] = END
        return bucket_data

    def transform(self, X, return_indexes=True, buckets_number=10, max_bucket_length=-1):
        lengths = [len(x[0])+2 for x in X]
        buckets_with_indexes = collect_buckets(lengths, buckets_number=buckets_number,
                                               max_bucket_length=max_bucket_length)
        data = [elem[0] for elem in X]
        data_by_buckets = [[self._make_bucket_data(data, length, indexes)]
                           for length, indexes in buckets_with_indexes]
        if self.use_label:
            features = np.array([self._make_feature_vector(*elem[1:]) for elem in X])
            features_by_buckets = [features[indexes] for _, indexes in buckets_with_indexes]
            for i, elem in enumerate(features_by_buckets):
                data_by_buckets[i].append(elem)
        for i, elem in enumerate(data_by_buckets):
            curr_answer = np.concatenate([elem[0][:,1:], PAD*np.ones_like(elem[0][:,-1:])], axis=1)
            elem.append(curr_answer)
        if return_indexes:
            return data_by_buckets, [elem[1] for elem in buckets_with_indexes]
        else:
            return data_by_buckets

    def train(self, X, X_dev=None, model_file=None, save_file=None):
        np.random.seed(self.random_state)  # initialize the random number generator
        self.vocabulary_ = Vocabulary(self.min_symbol_count).train([elem[0] for elem in X])
        if self.use_label:
            self.make_features(X)
        else:
            self.labels_, self.label_codes_ = None, None
        X_train, indexes_by_buckets = self.transform(X)
        if X_dev is not None:
            X_dev, dev_indexes_by_buckets = self.transform(X_dev, max_bucket_length=256)
        else:
            X_dev, dev_indexes_by_buckets = None, None
        self.build()
        if save_file is not None and model_file is not None:
            self.to_json(save_file, model_file)
        self.train_model(X_train, X_dev, model_file=model_file)
        return self

    def build(self, test=False):
        symbol_inputs = kl.Input(shape=(None,), dtype='int32')
        symbol_embeddings = self._build_symbol_layer(symbol_inputs)
        memory, initial_encoder_states, final_encoder_states =\
            self._build_history(symbol_embeddings, only_last=test)
        if self.labels_ is not None:
            feature_inputs = kl.Input(shape=(self.labels_number_,))
            inputs = [symbol_inputs, feature_inputs]
            tiled_feature_embeddings =\
                self._build_feature_network(feature_inputs, kb.shape(memory)[1])
            to_decoder = kl.Concatenate()([memory, tiled_feature_embeddings])
        else:
            inputs, to_decoder = [symbol_inputs], memory
        # lstm_outputs = kl.LSTM(self.rnn_size, return_sequences=True, dropout=self.dropout)(to_decoder)
        outputs, initial_decoder_states, final_decoder_states = self._build_output_network(to_decoder)
        compile_args = {"optimizer": ko.nadam(clipnorm=5.0), "loss": "categorical_crossentropy"}
        self.model_ = Model(inputs, outputs)
        self.model_.compile(**compile_args)
        if self.verbose > 0:
            print(self.model_.summary())
        step_func_inputs = inputs + initial_decoder_states + initial_encoder_states
        step_func_outputs = [outputs] + final_decoder_states + final_encoder_states
        self._step_func = kb.Function(step_func_inputs, step_func_outputs)
        return self

    def _build_symbol_layer(self, symbol_inputs):
        if self.use_embeddings:
            answer = kl.Embedding(self.symbols_number_, self.embeddings_size)(symbol_inputs)
            if self.embeddings_dropout > 0.0:
                answer = kl.Dropout(self.embeddings_dropout)(answer)
        else:
            answer = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number_},
                               output_shape=(None, self.symbols_number_))(symbol_inputs)
        return answer

    def _build_history(self, inputs, only_last=False):
        if not self.use_attention:
            memory = History(inputs, self.history, flatten=True, only_last=only_last)
            return memory, [], []
        initial_states = [kb.zeros_like(inputs[:, 0, 0]), kb.zeros_like(inputs[:, 0, 0])]
        for i, elem in enumerate(initial_states):
            initial_states[i] = kb.tile(elem[:, None], [1, self.encoder_rnn_size])
        encoder = kl.LSTM(self.encoder_rnn_size, return_sequences=True, return_state=True)
        lstm_outputs, final_c_states, final_h_states = encoder(inputs, initial_state=initial_states)
        attention_params = {"left": self.history, "input_dim": self.encoder_rnn_size,
                            "merge": self.attention_activation, "use_bias": self.use_attention_bias}
        memory = attention_func(lstm_outputs, only_last=only_last, **attention_params)
        return memory, initial_states, [final_c_states, final_h_states]

    def _build_feature_network(self, inputs, k):
        if self.feature_embedding_layers:
            inputs = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(inputs)
            inputs = kl.Dense(self.feature_embeddings_size,
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

    def _build_output_network(self, inputs):
        initial_states = [kb.zeros_like(inputs[:, 0, 0]), kb.zeros_like(inputs[:, 0, 0])]
        for i, elem in enumerate(initial_states):
            initial_states[i] = kb.tile(elem[:, None], [1, self.encoder_rnn_size])
        decoder = kl.LSTM(self.decoder_rnn_size, return_sequences=True, return_state=True)
        lstm_outputs, final_c_states, final_h_states = decoder(inputs, initial_state=initial_states)
        pre_outputs = kl.Dense(self.dense_output_size, activation="relu")(lstm_outputs)
        outputs = kl.TimeDistributed(
            kl.Dense(self.symbols_number_, activation="softmax"))(pre_outputs)
        return outputs, initial_states, [final_c_states, final_h_states]

    def train_model(self, X, X_dev=None, model_file=None):
        train_indexes_by_buckets, dev_indexes_by_buckets = [], []
        for curr_data in X:
            curr_indexes = list(range(len(curr_data[0])))
            np.random.shuffle(curr_indexes)
            if X_dev is None:
                # отделяем в каждой корзине данные для валидации
                train_bucket_size = int((1.0 - self.validation_split) * len(curr_indexes))
                train_indexes_by_buckets.append(curr_indexes[:train_bucket_size])
                dev_indexes_by_buckets.append(curr_indexes[train_bucket_size:])
            else:
                train_indexes_by_buckets.append(curr_indexes)
        if model_file is not None:
            callback = ModelCheckpoint(model_file, save_weights_only=True,
                                       save_best_only=True, verbose=0)
            if self.callbacks is not None:
                self.callbacks.append(callback)
            else:
                self.callbacks = [callback]
        if X_dev is not None:
            for curr_data in X_dev:
                dev_indexes_by_buckets.append(list(range(len(curr_data[0]))))
        train_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(train_indexes_by_buckets))))
        dev_batches_indexes = list(chain.from_iterable(
            (((i, j) for j in range(0, len(bucket), self.batch_size))
             for i, bucket in enumerate(dev_indexes_by_buckets))))
        if X_dev is None:
            X_dev = X
        train_gen = generate_data(X, train_indexes_by_buckets, train_batches_indexes,
                                  self.batch_size, self.symbols_number_)
        val_gen = generate_data(X_dev, dev_indexes_by_buckets, dev_batches_indexes,
                                self.batch_size, self.symbols_number_, shuffle=False)
        self.model_.fit_generator(
            train_gen, len(train_batches_indexes), epochs=self.nepochs,
            callbacks=self.callbacks, verbose=1, validation_data=val_gen,
            validation_steps=len(dev_batches_indexes))
        if model_file is not None:
            self.model_.load_weights(model_file)
        return self

    def _score_batch(self, bucket, answer, lengths, batch_size=1):
        """
        :Arguments
         batch: list of np.arrays, [data, (features)]
            data: shape=(batch_size, length)
            features(optional): shape=(batch_size, self.feature_vector_size)
        :return:
        """
        # elem[0] because elem = [word, (pos, (feats))]
        bucket_size, length = bucket[0].shape[:2]
        padding = np.full(answer[:,:1].shape, PAD, answer.dtype)
        shifted_data = np.hstack((answer[:,1:], padding))
        # evaluate принимает только данные того же формата, что и выход модели
        answers = to_one_hot(shifted_data, self.output_symbols_number)
        # total = self.model.evaluate(bucket, answers, batch_size=batch_size)
        # last two scores are probabilities of word end and final padding symbol
        scores = self.model_.predict(bucket, batch_size=batch_size)
        # answers_, scores_ = kb.constant(answers), kb.constant(scores)
        # losses = kb.eval(kb.categorical_crossentropy(answers_, scores_))
        scores = np.clip(scores, EPS, 1.0 - EPS)
        losses = -np.sum(answers * np.log(scores), axis=-1)
        total = np.sum(losses, axis=1) # / np.log(2.0)
        letter_scores = scores[np.arange(bucket_size)[:,np.newaxis],
                               np.arange(length)[np.newaxis,:], shifted_data]
        letter_scores = [elem[:length] for elem, length in zip(letter_scores, lengths)]
        return letter_scores, total

    def score(self, x, **args):
        return self.predict([x], batch_size=1, **args)

    def predict(self, X, batch_size=32, return_letter_scores=False,
                return_log_probs=False, return_exp_total=False):
        """

        answer = [answer_1, ..., answer_m]
        answer_i =
            (letter_score_i, total_score_i), если return_letter_scores = True,
            total_score, если return_letter_scores = False
        letter_score_i = [p_i0, ..., p_i(l_i-1), p_i(END)]
        p_ij --- вероятность j-ого элемента в X[i]
            (логарифм вероятности, если return_log_probs=True)


        Вычисляет логарифмические вероятности для всех слов в X,
        а также вероятности отдельных символов
        """
        fields_number = 2 if self.labels_ is not None else 1
        answer_index = -1 if self.symbols_has_features else 0
        X_test, indexes = self.transform(X, bucket_size=batch_size, join_buckets=False)
        answer = [None] * len(X)
        lengths = np.array([len(x[0]) + 1 for x in X])
        for j, curr_indexes in enumerate(indexes, 1):
            # print("Lm bucket {} scoring".format(j))
            curr_indexes.sort()
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            y_curr = np.array([X_test[j][answer_index] for j in curr_indexes])
            letter_scores, total_scores = self._score_batch(
                X_curr, y_curr, lengths[curr_indexes], batch_size=batch_size)
            if return_log_probs:
                # letter_scores = -np.log2(letter_scores)
                letter_scores = [-np.log(letter_score) for letter_score in letter_scores]
            if return_exp_total:
                total_scores = np.exp(total_scores) # 2.0 ** total_scores
            for i, letter_score, total_score in zip(curr_indexes, letter_scores, total_scores):
                answer[i] = (letter_score, total_score) if return_letter_scores else total_score
        return answer

    def predict_proba(self, X, batch_size=256):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, pad=False, return_indexes=True)
        answer = [None] * len(X)
        start_probs = np.zeros(shape=(1, self.output_symbols_number), dtype=float)
        start_probs[0, BEGIN] = 1.0
        for curr_indexes in indexes:
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            curr_probs = self.model_.predict(X_curr, batch_size=batch_size)
            for i, probs in zip(curr_indexes, curr_probs):
                answer[i] = np.vstack((start_probs, probs[:len(X[i][0])+1]))
        return answer

    def predict_attention(self, X, batch_size=32):
        fields_number = 2 if self.labels_ is not None else 1
        X_test, indexes = self.transform(X, pad=False, return_indexes=True)
        answer = [None] * len(X)
        for curr_indexes in indexes:
            X_curr = [np.array([X_test[j][k] for j in curr_indexes]) for k in range(fields_number)]
            # нужно добавить фазу обучения (используется dropout)
            curr_attention = self._attention_func_(X_curr + [0])
            for i, elem in zip(curr_indexes, curr_attention[0]):
                answer[i] = elem
        return answer

    def perplexity(self, X, bucket_size=None, log2=False):
        X_test, indexes = self.transform(X, bucket_size=bucket_size, join_buckets=False)
        use_last = not(self.symbols_has_features)
        eval_gen = generate_data(X_test, indexes, self.output_symbols_number,
                                 use_last=use_last, shift_answer=True, shuffle=False)
        loss = self.model_.evaluate_generator(eval_gen, len(indexes))
        if log2:
            loss /= np.log(2.0)
        return loss

    def get_embeddings_weights(self):
        if not self.use_embeddings:
            return None
        try:
            if self.symbols_has_features:
                layer = self.model_.get_layer("distributed_embeddings").layer
            else:
                layer = self.model_.get_layer("layer_embeddings")
        except ValueError:
            return None
        weights = layer.get_weights()
        return weights[0]


def load_lm(infile):
    with open(infile, "r", encoding="utf8") as fin:
        json_data = json.load(fin)
    args = {key: value for key, value in json_data.items()
            if not (key.endswith("_") or key.endswith("callback") or key.endswith("dump_file"))}
    callbacks = []
    early_stopping_callback_data = json_data.get("early_stopping_callback")
    if early_stopping_callback_data is not None:
        callbacks.append(EarlyStopping(**early_stopping_callback_data))
    # reduce_LR_callback_data = json_data.get("reduce_LR_callback")
    # if reduce_LR_callback_data is not None:
    #     callbacks.append(ReduceLROnPlateau(**reduce_LR_callback_data))
    model_checkpoint_callback_data = json_data.get("model_checkpoint_callback")
    if model_checkpoint_callback_data is not None:
        callbacks.append(ModelCheckpoint(**model_checkpoint_callback_data))
    args['callbacks'] = callbacks
    # создаём языковую модель
    lm = NeuralLM(**args)
    # обучаемые параметры
    args = {key: value for key, value in json_data.items() if key[-1] == "_"}
    for key, value in args.items():
        if key == "vocabulary_":
            setattr(lm, key, vocabulary_from_json(value, lm.symbols_has_features))
        else:
            setattr(lm, key, value)
    # модель
    lm.build()  # не работает сохранение модели, приходится сохранять только веса
    lm.model_.load_weights(json_data['dump_file'])
    return lm


