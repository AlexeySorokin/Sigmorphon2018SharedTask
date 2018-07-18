import os
import inspect
import ujson as json

from common import *
from common_neural import *
from cells import MultiConv1D, TemporalDropout
from mcmc_aligner import Aligner, output_alignment

import keras.backend as kb
if kb.backend() == "tensorflow":
    from common_tensorflow import *
else:
    import theano
import keras.regularizers as kreg
from keras.optimizers import adam

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


class Inflector:

    AUXILIARY = ['PAD', BOW, EOW, 'UNKNOWN']
    UNKNOWN_FEATURE = 0

    DEFAULT_ALIGNER_PARAMS = {"init_params": {"gap": 1, "initial_gap": 0}, "n_iter": 5,
                              "init": "lcs", "separate_endings": True}
    MAX_STEPS_NUMBER = 3

    def __init__(self, aligner_params=None, use_full_tags=False,
                 models_number=1, buckets_number=10, batch_size=32,
                 nepochs=25, validation_split=0.2, reverse=False,
                 # input_history=5, use_attention=False, input_right_context=0,
                 # output_history=1, separate_symbol_history=False, step_history=1,
                 # use_output_attention=False, history_embedding_size=32,
                 use_embeddings=False, embeddings_size=16,
                 use_feature_embeddings=False, feature_embeddings_size=16,
                 conv_layers=0, conv_window=32, conv_filters=5,
                 rnn="lstm", encoder_rnn_layers=1, encoder_rnn_size=32,
                 decoder_rnn_size=32, dense_output_size=32,
                 use_decoder_gate=False,
                 # regularizer="l2"
                 conv_dropout=0.0, encoder_rnn_dropout=0.0, dropout=0.0,
                 history_dropout=0.0, decoder_dropout=0.0, regularizer=0.0,
                 callbacks=None,
                 # use_lm=False, min_prob=0.01, max_diff=2.0,
                 random_state=187, verbose=1):
        self.use_full_tags = use_full_tags
        self.models_number = models_number
        self.buckets_number = buckets_number
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.reverse = reverse
        # self.input_history = input_history
        # self.input_right_context = input_right_context
        # self.use_attention = use_attention
        # self.output_history = output_history
        # self.separate_symbol_history = separate_symbol_history
        # self.step_history = step_history
        # self.use_output_attention = use_output_attention
        # self.history_embedding_size = history_embedding_size
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.use_feature_embeddings = use_feature_embeddings
        self.feature_embeddings_size = feature_embeddings_size
        self.conv_layers = conv_layers
        self.conv_window = conv_window
        self.conv_filters = conv_filters
        self.rnn = rnn
        self.encoder_rnn_layers = encoder_rnn_layers
        self.encoder_rnn_size = encoder_rnn_size
        self.decoder_rnn_size = decoder_rnn_size
        self.dense_output_size = dense_output_size
        self.use_decoder_gate = use_decoder_gate
        # self.regularizer = regularizer
        self.conv_dropout = conv_dropout
        self.encoder_rnn_dropout = encoder_rnn_dropout
        self.dropout = dropout
        self.history_dropout = history_dropout
        self.decoder_dropout = decoder_dropout
        self.regularizer = regularizer
        self.callbacks = callbacks or []
        # # декодинг
        # self.use_lm = use_lm
        # self.min_prob = min_prob
        # self.max_diff = max_diff
        # разное
        self.random_state = random_state
        self.verbose = verbose
        # выравнивания
        self._make_aligner(aligner_params)
        self._initialize()
        # self._initialize_callbacks(model_file)

    def _initialize(self):
        if isinstance(self.conv_window, int):
            self.conv_window = [self.conv_window]
        if isinstance(self.conv_filters, int):
            self.conv_filters = [self.conv_filters]
        if isinstance(self.rnn, str):
            self.rnn = getattr(kl, self.rnn.upper())
        if self.rnn not in [kl.GRU, kl.LSTM]:
            self.rnn = None

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

    def to_json(self, outfile, model_file=None):
        info = dict()
        if model_file is None:
            pos = outfile.rfind(".")
            model_file = outfile[:pos] + ("-model.hdf5" if pos != -1 else "-model")
        model_files = [self._make_model_file(model_file, i+1) for i in range(self.models_number)]
        for i in range(self.models_number):
            model_files[i] = os.path.abspath(model_files[i])
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(Inflector, attr, None), property) or
                    attr.isupper() or attr in ["callbacks", "models_"]):
                info[attr] = val
            elif attr == "models_":
                info["model_files"] = model_files
                for model, curr_model_file in zip(self.models_, model_files):
                    model.save_weights(curr_model_file)
            elif attr == "callbacks":
                raise NotImplementedError
                # for callback in val:
                #     if isinstance(callback, EarlyStopping):
                #         info["early_stopping_callback"] = {"patience": callback.patience,
                #                                            "monitor": callback.monitor}
                #     elif isinstance(callback, ReduceLROnPlateau):
                #         curr_val = dict()
                #         for key in ["patience", "monitor", "factor"]:
                #             curr_val[key] = getattr(callback, key)
                #         info["reduce_LR_callback"] = curr_val
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

    def _make_vocabulary(self, X):
        symbols = {a for first, second, _ in X for a in first+second}
        self.symbols_ = self.AUXILIARY + [STEP] + sorted(symbols)
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
        self.labels_ = self.AUXILIARY + sorted(pos_labels) + sorted(feature_labels)
        self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        if self.use_full_tags:
            self.tags_ = sorted(tags)
            self.tag_codes_ = {x: i for i, x in enumerate(self.tags_)}
        return self

    def extract_features(self, descr):
        answer = np.zeros(shape=(self.feature_vector_size,), dtype=np.uint8)
        features = [descr[0]] + ["{}_{}".format(descr[0], x) for x in descr[1:]]
        feature_codes = [self.label_codes_[x] for x in features if x in self.label_codes_]
        answer[feature_codes] = 1
        if self.use_full_tags:
            code = self.label_codes_.get(tuple(descr))
            if code is not None:
                answer[len(self.labels_) + code] = 1
        return answer

    def _make_bucket_data(self, lemmas, bucket_length, bucket_indexes):
        bucket_size = len(bucket_indexes)
        bucket_data = np.full(shape=(bucket_size, bucket_length),
                              fill_value=PAD, dtype=int)
        # заполняем закодированными символами
        bucket_data[:,0] = BEGIN
        for j, i in enumerate(bucket_indexes):
            lemma = lemmas[i]
            bucket_data[j,1:1+len(lemma)] = [self.symbol_codes_[x] for x in lemma]
            bucket_data[j,1+len(lemma)] = END
        return bucket_data

    def _make_shifted_output(self, bucket_length, bucket_size, targets=None):
        answer = np.full(shape=(bucket_size, bucket_length), fill_value=BEGIN, dtype=int)
        if targets is not None:
            answer[:,1:] = targets[:, :-1]
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
                                 self.output_history), dtype=np.uint8)
        answer[:,0] = BEGIN
        if targets is not None:
            for i in range(1, bucket_length):
                # shift the history only if last action was not the step symbol
                answer[:,i] = np.where(targets[:,i-1,None] == self.STEP_CODE, answer[:,i-1],
                                       np.hstack([answer[:,i-1,1:], targets[:,i-1,None]]))
        return answer

    def transform_training_data(self, lemmas, features, letter_positions, targets):
        """

        lemmas: list of strs,
            список лемм (без добавления начала и конца строки)
        letter_positions: list of lists of ints
            список индексов букв в выравнивании,
            letter_indexes[r][i] = j <=> lemmas[r][i] порождает targets[r][j]
        targets: list of strs,
            список порождаемых словоформ (с учётом начала и конца строки)

        """
        alignment_lengths = [max(len(lemma)+2, len(target))
                             for lemma, target in zip(lemmas, targets)]
        self.max_length_shift_ = max(0, max([len(target) - 2 * len(lemma)
                                             for lemma, target in zip(lemmas, targets)]))
        buckets_with_indexes = collect_buckets(alignment_lengths, self.buckets_number)
        data_by_buckets = [self._make_bucket_data(lemmas, length, indexes)
                           for length, indexes in buckets_with_indexes]
        features_by_buckets = [
            np.array([self.extract_features(features[i]) for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        targets = np.array([[self.symbol_codes_[x] for x in elem] for elem in targets])
        letter_positions_by_buckets = [
            make_table(letter_positions, length, indexes, fill_with_last=True)
            for length, indexes in buckets_with_indexes]
        targets_by_buckets = [make_table(targets, length, indexes, fill_value=PAD)
                              for length, indexes in buckets_with_indexes]
        def _make_bucket_data(func):
            return [func(L, len(indexes), table) for table, (L, indexes)
                    in zip(targets_by_buckets, buckets_with_indexes)]
        history_by_buckets = _make_bucket_data(self._make_shifted_output)
        answer = list(zip(data_by_buckets, features_by_buckets,
                          letter_positions_by_buckets, history_by_buckets))
        for i in range(len(answer)):
            answer[i] += (targets_by_buckets[i],)
        return answer, [elem[1] for elem in buckets_with_indexes]

    def _preprocess(self, data, alignments, to_fit=True, alignments_outfile=None):
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
        lemmas = [elem[0] for elem in data]
        features = [elem[2] for elem in data]
        letter_indexes = [elem[0] for elem in indexes_with_targets]
        targets = [elem[1] for elem in indexes_with_targets]
        if self.reverse:
            lemmas = [x[::-1] for x in lemmas]
        return self.transform_training_data(lemmas, features, letter_indexes, targets)

    def train(self, data, alignments=None, dev_data=None, dev_alignments=None,
              alignments_outfile=None, model_file=None, save_file=None):
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
        self._make_vocabulary(data)
        self._make_features([elem[2] for elem in data])

        data_by_buckets, buckets_indexes = self._preprocess(
            data, alignments, to_fit=True, alignments_outfile=alignments_outfile)
        if dev_data is not None:
            dev_data_by_buckets, dev_bucket_indexes =\
                self._preprocess(dev_data, dev_alignments, to_fit=False)
        else:
            dev_data_by_buckets, dev_bucket_indexes = None, None
        self.build()
        # if save_file is not None:
        #     self.to_json(save_file, model_file)
        self._train_model(data_by_buckets, X_dev=dev_data_by_buckets, model_file=model_file)
        return self

    def _train_model(self, X, X_dev=None, model_file=None):
        train_indexes_by_buckets, dev_indexes_by_buckets = [], []
        np.random.seed(self.random_state)
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
        for i, model in enumerate(self.models_):
            if model_file is not None:
                curr_model_file = self._make_model_file(model_file, i+1)
                save_callback = ModelCheckpoint(
                    curr_model_file, save_weights_only=True, save_best_only=True)
                curr_callbacks = self.callbacks + [save_callback]
            else:
                curr_callbacks = self.callbacks
            train_gen = generate_data(X, train_indexes_by_buckets, train_batches_indexes,
                                      self.batch_size, self.symbols_number)
            val_gen = generate_data(X_dev, dev_indexes_by_buckets, dev_batches_indexes,
                                    self.batch_size, self.symbols_number, shuffle=False)
            model.fit_generator(train_gen, len(train_batches_indexes),
                                epochs=self.nepochs, callbacks=curr_callbacks,
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
        if self.verbose > 0:
            print(self.models_[0].summary())
        return self

    def _build_word_network(self, inputs):
        if self.use_embeddings:
            embedding = kl.Embedding(self.symbols_number, self.embeddings_size)
        else:
            embedding = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                  output_shape=(None, self.symbols_number))
        outputs = embedding(inputs)
        if self.conv_layers > 0:
            outputs = MultiConv1D(outputs, self.conv_layers, self.conv_window,
                                  self.conv_filters, self.conv_dropout, activation="relu")
        if self.encoder_rnn_layers > 0:
            for i in range(self.encoder_rnn_layers):
                outputs = kl.Bidirectional(kl.LSTM(self.encoder_rnn_size,
                                                   dropout=self.encoder_rnn_dropout,
                                                   return_sequences=True))(outputs)
        return outputs

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
        if self.use_feature_embeddings:
            inputs = kl.Dense(self.feature_embeddings_size,
                              input_shape=(self.feature_vector_size,),
                              activation="relu", use_bias=False)(inputs)
        def tiling_func(x):
            x = kb.expand_dims(x, 1)
            return kb.tile(x, [1, k, 1])
        answer = kl.Lambda(tiling_func, output_shape=(lambda x: (None,) + x))(inputs)
        answer = kl.Lambda(kb.cast, arguments={"dtype": "float32"})(answer)
        return answer

    def _build_history_network(self, inputs):
        outputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                            output_shape=(None, self.symbols_number))(inputs)
        if self.history_dropout > 0.0:
            outputs = TemporalDropout(outputs, self.history_dropout)
        return outputs

    def _build_decoder_network(self, inputs, source):
        inputs = kl.Concatenate()(inputs)
        decoder_outputs = kl.LSTM(self.decoder_rnn_size, return_sequences=True,
                                 dropout=self.decoder_dropout)(inputs)
        pre_outputs = kl.TimeDistributed(kl.Dense(
            self.dense_output_size, activation="relu", use_bias=False,
            input_shape=(self.decoder_rnn_size,)))(decoder_outputs)
        outputs = kl.TimeDistributed(kl.Dense(
            self.symbols_number, activation="softmax", name="outputs",
            activity_regularizer=kreg.l2(self.regularizer),
            input_shape=(self.dense_output_size,)))(pre_outputs)
        if self.use_decoder_gate:
            gate_inputs = kl.Concatenate()([inputs, decoder_outputs])
            gate_outputs = kl.Dense(1, activation="sigmoid")(gate_inputs)
            source_inputs = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                      output_shape=(lambda x: x + (self.symbols_number,)))(source)
            outputs = kl.Lambda(gated_sum, output_shape=(lambda x:x[0]))([source_inputs, outputs, gate_outputs])
        return outputs

    def _build_model(self):
        # входные данные
        symbol_inputs = kl.Input(name="symbol_inputs", shape=(None,), dtype='uint8')
        feature_inputs = kl.Input(
            shape=(self.feature_vector_size,), name="feature_inputs", dtype='uint8')
        symbol_outputs = self._build_word_network(symbol_inputs)
        letter_indexes_inputs = kl.Input(shape=(None,), dtype='int32')
        aligned_symbol_outputs = self._build_alignment(
            symbol_outputs, letter_indexes_inputs, dropout=self.dropout)
        aligned_inputs = self._build_alignment(symbol_inputs, letter_indexes_inputs)
        shifted_target_inputs = kl.Input(
            shape=(None,), name="shifted_target_inputs", dtype='uint8')
        inputs = [symbol_inputs, feature_inputs,
                  letter_indexes_inputs, shifted_target_inputs]
        # буквы входного слова
        tiled_feature_inputs = self._build_feature_network(
            feature_inputs, kb.shape(aligned_symbol_outputs)[1])
        # to_concatenate = [aligned_symbol_outputs, tiled_feature_inputs]
        shifted_outputs = self._build_history_network(shifted_target_inputs)
        to_decoder = [aligned_symbol_outputs, tiled_feature_inputs, shifted_outputs]
        outputs = self._build_decoder_network(to_decoder, aligned_inputs)
        model = Model(inputs, outputs)
        model.compile(optimizer=adam(clipnorm=5.0),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        encoder = kb.function([symbol_inputs], [symbol_outputs])
        decoder_inputs = [aligned_symbol_outputs, feature_inputs, shifted_target_inputs]
        decoder = kb.function(decoder_inputs + [kb.learning_phase()], [outputs])
        return model, encoder, decoder

    def _prepare_for_prediction(self, words, features,
                                buckets_with_indexes, known_answers=None):
        encoded_words = np.array(
            [([BEGIN] + [self.symbol_codes_.get(
                x, UNKNOWN) for x in word] + [END]) for word in words])
        encoded_words_by_buckets = [make_table(encoded_words, length, indexes, fill_value=PAD)
                                    for length, indexes in buckets_with_indexes]
        # представления входных слов после lstm
        # encoder_outputs_by_buckets[i].shape = (self.models, lengths[i])
        encoder_outputs_by_buckets =\
            self._predict_encoder_outputs(encoded_words_by_buckets)
        features_by_buckets = [
            np.array([self.extract_features(features[i]) for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        shifted_targets_by_buckets = [self._make_shifted_output(L, len(indexes))
                                      for (L, indexes) in buckets_with_indexes]
        if known_answers is not None:
            encoded_answers = np.array([([BEGIN] +
                                         [self.symbol_codes_.get(x, UNKNOWN) for x in word] +
                                         [END]) for word in known_answers])
            encoded_answers_by_buckets = [
                make_table(encoded_answers, length, indexes, fill_value=PAD)
                for length, indexes in buckets_with_indexes]
        else:
            encoded_answers_by_buckets = [None] * len(buckets_with_indexes)
        # predictions = [(words_1, probs_1), ..., (words_m, probs_m)]
        inputs = [encoder_outputs_by_buckets, features_by_buckets,
                  shifted_targets_by_buckets]
        return inputs, encoded_words_by_buckets, encoded_answers_by_buckets

    def predict(self, data, known_answers=None, return_alignment_positions=False,
                return_log=False, beam_width=1, verbose=0):
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
        words, features = [elem[0] for elem in data], [elem[2] for elem in data]
        output_lengths = [(2 * len(x) + self.max_length_shift_ + 3) for x in words]
        buckets_with_indexes = collect_buckets(output_lengths, max_bucket_size=64)
        inputs, encoded_words_by_buckets, encoded_answers_by_buckets =\
            self._prepare_for_prediction(words, features, buckets_with_indexes, known_answers)
        answer = [[] for _ in words]
        for bucket_number, elem in enumerate(zip(*inputs)):
            if verbose > 0:
                print("Bucket {} of {} predicting".format(
                    bucket_number+1, len(buckets_with_indexes)))
            bucket_words, bucket_probs = self._predict_on_batch(
                elem[0], elem[1], elem[2], beam_width=beam_width,
                known_answers=encoded_answers_by_buckets[bucket_number],
                source=encoded_words_by_buckets[bucket_number])
            _, bucket_indexes = buckets_with_indexes[bucket_number]
            for i, curr_words, curr_probs in zip(bucket_indexes, bucket_words, bucket_probs):
                for curr_symbols, curr_prob in zip(curr_words, curr_probs):
                    answer[i].append(self.decode_answer(
                        curr_symbols, curr_prob,
                        return_alignment_positions=return_alignment_positions,
                        return_log=return_log))
                scores = np.array([10**(-elem[2]) for elem in answer[i]])
                scores /= np.sum(scores)
                for j, score in enumerate(scores):
                    answer[i][j][2] = score
        return answer

    def _predict_encoder_outputs(self, data):
        answer = []
        for i, bucket in enumerate(data):
            # нельзя брать средний выход энкодера!!!!
            curr_answers = np.array([encoder([bucket])[0] for encoder in self.encoders_])
            # answer.append(np.mean(curr_answers, axis=0))
            answer.append(curr_answers)
        return answer


    def _predict_on_batch(self, symbols, features, target_history,
                          steps_history=None, beam_width=1, known_answers=None,
                          source=None):
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
        ## ПОМЕНЯТЬ ДЛЯ self.separate_steps
        _, m, L, H = symbols.shape
        M = m * beam_width
        # positions[j] --- текущая позиция в symbols[j]
        positions = np.zeros(shape=(M,), dtype=int)
        if known_answers is not None:
            positions_in_answers = np.zeros(shape=(M,), dtype=int)
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
        symbol_inputs = np.zeros(shape=(self.models_number, M, L, H), dtype=float)
        # размножаем признаки и shifted_targets
        features = np.repeat(features, beam_width, axis=0)
        target_history = np.repeat(target_history, beam_width, axis=0)
        if steps_history is not None:
            steps_history = np.repeat(steps_history, beam_width, axis=0)
        if source is not None:
            source = np.repeat(source, beam_width, axis=0)
        for i in range(L):
            # текущие символы
            curr_symbols = symbols[:, np.arange(M) // beam_width, positions,:]
            symbol_inputs[:,:,i] = curr_symbols
            i_slice = i if self.new else np.arange(i+1)
            args = [symbol_inputs[:,is_active,i_slice], features[is_active],
                    target_history[is_active,i_slice]]
            if self.separate_symbol_history:
                args.append(steps_history[is_active,i_slice])
            if self.new:
                args += [h_states[is_active], c_states[is_active]]
                curr_outputs, new_h_states, new_c_states = self._predict_current_cell_output(*args)
            else:
                # curr_outputs = self._predict_current_output(*args)
                curr_outputs = self._predict_current_output(*args)[:,-1]
            active_source = source[np.arange(M), positions][is_active]
            # curr_outputs[active_source == END, -1, STEP_CODE] = 0.0
            # curr_outputs[active_source == PAD, -1, STEP_CODE] = 0.0
            curr_outputs[active_source == END, STEP_CODE] = 0.0
            curr_outputs[active_source == PAD, STEP_CODE] = 0.0
            if i == L-1:
                # curr_outputs[:,-1,np.arange(self.symbols_number) != END] = 0.0
                curr_outputs[:,np.arange(self.symbols_number) != END] = 0.0
            # если текущий символ в
            if known_answers is not None:
                curr_known_answers = known_answers[np.arange(M),positions_in_answers]
                are_steps_allowed = np.min(
                    [i - positions_in_answers < allowed_steps_number,
                     source[np.arange(M), positions] != END,
                     source[np.arange(M), positions] != PAD,
                     np.maximum(current_steps_number < self.MAX_STEPS_NUMBER,
                                curr_known_answers == END)], axis=0)
                self._zero_impossible_probs(
                    curr_outputs, curr_known_answers[is_active], are_steps_allowed[is_active])
            hypotheses_by_groups = [[] for _ in range(m)]
            if beam_width == 1:
                # curr_output_symbols = np.argmax(curr_outputs[:,-1], axis=1)
                curr_output_symbols = np.argmax(curr_outputs, axis=1)
                # for j, curr_probs, index in zip(
                #         np.nonzero(is_active)[0], curr_outputs[:,-1], curr_output_symbols):
                for r, (j, curr_probs, index) in enumerate(zip(
                        np.nonzero(is_active)[0], curr_outputs, curr_output_symbols)):
                    new_score = partial_scores[j] - np.log10(curr_probs[index])
                    hyp = (j, index, new_score, -np.log10(curr_probs[index]))
                    if self.new:
                        hyp += (new_h_states[r], new_c_states[r])
                    hypotheses_by_groups[j] = [hyp]
            else:
                curr_best_scores = [np.inf] * m
                # for j, curr_probs in zip(np.nonzero(is_active)[0], curr_outputs[:,-1]):
                for r, (j, curr_probs) in enumerate(zip(np.nonzero(is_active)[0], curr_outputs)):
                    group_index = j // beam_width
                    prev_partial_score = partial_scores[j]
                    # переходим к логарифмической шкале
                    curr_probs = -np.log10(curr_probs)
                    if np.isinf(curr_best_scores[group_index]):
                        curr_best_scores[group_index] = prev_partial_score + np.min(curr_probs)
                    min_log_prob = curr_best_scores[group_index] - prev_partial_score + self.max_diff
                    min_log_prob = min(-np.log10(self.min_prob), min_log_prob)
                    if known_answers is None:
                        possible_indexes = np.where(curr_probs <= min_log_prob)[0]
                    else:
                        possible_indexes = np.where(curr_probs < np.inf)[0]
                    if len(possible_indexes) == 0:
                        possible_indexes = [np.argmin(curr_probs)]
                    for index in possible_indexes:
                        new_score = prev_partial_score + curr_probs[index]
                        hyp = (j, index, new_score, curr_probs[index])
                        if self.new:
                            hyp += (new_h_states[r], new_c_states[r])
                        hypotheses_by_groups[group_index].append(hyp)
            for j, (curr_hypotheses, group_beam_width) in\
                    enumerate(zip(hypotheses_by_groups, group_beam_widths)):
                if group_beam_width == 0:
                    continue
                curr_hypotheses = sorted(
                    curr_hypotheses, key=(lambda x:x[2]))[:group_beam_width]
                group_start = j * beam_width
                free_indexes = np.where(np.logical_not(
                    is_completed[group_start:group_start+beam_width]))[0]
                free_indexes = free_indexes[:len(curr_hypotheses)] + group_start
                is_active[group_start:group_start+beam_width] = False
                extend_history(words, curr_hypotheses, free_indexes, pos=1)
                extend_history(probs, curr_hypotheses, free_indexes, pos=3)
                extend_history(partial_scores, curr_hypotheses, free_indexes,
                               pos=2, func=lambda x, y: y)
                extend_history(positions, curr_hypotheses, free_indexes,
                               pos=1, func=(lambda x, y: x+int(y == STEP_CODE)))
                if known_answers is not None:
                    extend_history(positions_in_answers, curr_hypotheses, free_indexes,
                                   pos=1, func=(lambda x, y: x+int(y != STEP_CODE)))
                    extend_history(current_steps_number, curr_hypotheses, free_indexes,
                                   pos=1, func=(lambda x, y: x+int(y == STEP_CODE)))
                if self.new:
                    extend_history(h_states, curr_hypotheses, free_indexes, pos=4, func=(lambda x, y: y))
                    extend_history(c_states, curr_hypotheses, free_indexes, pos=5, func=(lambda x, y: y))
                # здесь нужно достроить гипотезы и разместить их в нужные места
                # group_words = [words[elem[0]] + [elem[1]] for elem in curr_hypotheses]
                # group_probs = [probs[elem[0]] + [elem[3]] for elem in curr_hypotheses]
                # group_partial_scores = [elem[2] for elem in curr_hypotheses]
                # group_positions = [positions[elem[0]] for elem in curr_hypotheses]
                # if known_answers is not None:
                #     group_positions_in_answers =\
                #         [positions_in_answers[elem[0]] for elem in curr_hypotheses]
                #     group_steps_number = [current_steps_number[elem[0]] for elem in curr_hypotheses]
                if i < L-1:
                    # group_shifted_targets, group_shifted_steps = [], []
                    extend_history(target_history, curr_hypotheses, free_indexes,
                                   pos=1, func="append_shifted", end=i, record_steps=False,
                                   separate_symbol_history=self.separate_symbol_history)
                    if self.separate_symbol_history:
                        extend_history(steps_history, curr_hypotheses, free_indexes,
                                       pos=1, func="append_shifted", end=i, record_steps=True)
                    # for r, elem in enumerate(curr_hypotheses):
                    #     curr_target_history = np.copy(target_history[elem[0]])
                    #     if elem[1] == STEP_CODE and self.separate_symbol_history:
                    #         curr_target_history[i+1] = curr_target_history[i]
                    #     else:
                    #         curr_target_history[i+1] =\
                    #             np.concatenate((curr_target_history[i,1:], [elem[1]]))
                    #     group_shifted_targets.append(curr_target_history)
                    #     if self.separate_symbol_history:
                    #         curr_step_history = np.copy(steps_history[elem[0]])
                    #         curr_step_history[i+1][:-1] = curr_step_history[i][1:]
                    #         curr_step_history[i+1,-1] = int(elem[1] == STEP_CODE)
                    #         group_shifted_steps.append(curr_step_history)
                for r, index in enumerate(free_indexes):
                    if r > len(curr_hypotheses):
                        break
                    # words[index], probs[index] = group_words[r], group_probs[r]
                    # partial_scores[index] = group_partial_scores[r]
                    # positions[index] = group_positions[r] + int(last_symbol == STEP_CODE)
                    # if i < L - 1:
                    #     target_history[index] = group_shifted_targets[r]
                    #     if self.separate_symbol_history:
                    #         steps_history[index] = group_shifted_steps[r]
                    is_active[index] = (curr_hypotheses[r][1] != END)
                    is_completed[index] = not(is_active[index])
                    # if known_answers is not None:
                    #     positions_in_answers[index] =\
                    #         group_positions_in_answers[r] + int(last_symbol != STEP_CODE)
                    #     current_steps_number[index] =\
                    #         group_steps_number[r] + 1 if last_symbol == STEP_CODE else 0
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
        answer = []
        for i, decoder in enumerate(self.decoders_):
            answer.append(decoder([args[0][i]] + list(args[1:]) + [0])[0])
        result = np.mean(answer, axis=0)
        return result

    def _predict_current_cell_output(self, *args):
        answer = [[], [], []]
        for i, decoder in enumerate(self.decoders_):
            curr_args = [args[0][i]] + list(args[1:-2]) + [args[-2][:,i], args[-1][:,i]]
            # моделируем вычисление декодера на одном из элементов
            curr_answer = decoder.predict(curr_args)
            for elem, to_append in zip(answer, curr_answer):
                elem.append(to_append)
        answer[0] = np.mean(answer[0], axis=0)
        answer[1] = np.transpose(answer[1], axes=(1,0,2))
        answer[2] = np.transpose(answer[2], axes=(1,0,2))
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
                      return_log=True, are_log_probs=True):
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
            if self.symbols_[index] not in self.AUXILIARY:
                word += self.symbols_[index]
        answer = [word, probs_to_return, np.sum(probs_to_return)]
        if return_alignment_positions:
            answer += (alignment_positions,)
        if not return_log:
            answer[1] = [10.0**(-x) for x in probs_to_return]
        return answer

    def test_prediction(self, data, alignments=None):
        # as for now, we cannot load aligner, therefore have to train it
        data_by_buckets, buckets_indexes = self._preprocess(data, alignments, to_fit=False)
        predictions, answer = [None] * len(data), [None] * len(data)
        for curr_indexes, elem in zip(buckets_indexes, data_by_buckets):
            curr_predictions = self.models_[0].predict(list(elem[:-1]))
            for index, outputs, corr_outputs in zip(curr_indexes, curr_predictions, elem[-1]):
                predictions[index] = outputs
                curr_answer = []
                for j, probs in enumerate(outputs):
                    letter_code = corr_outputs[j]
                    if letter_code == END:
                        break
                    curr_answer.append((self.symbols_[letter_code], probs[letter_code]))
                answer[index] = curr_answer
        return predictions, answer