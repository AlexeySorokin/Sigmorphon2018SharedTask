

class AGInflector:

    AUXILIARY = ['PAD', BOW, EOW, 'UNKNOWN']
    PAD, BEGIN, END, UNKNOWN, STEP_CODE = 0, 1, 2, 3, 4
    UNKNOWN_FEATURE = 0

    DEFAULT_ALIGNER_PARAMS = {"init_params": {"gap": 1, "initial_gap": 0}, "n_iter": 5,
                              "init": "lcs", "separate_endings": True}
    MAX_STEPS_NUMBER = 3

    def __init__(self, aligner_params=None, add_no_value=False, use_entire_features=False,
                 models_number=1, buckets_number=10, batch_size=32,
                 nepochs=25, validation_split=0.2, reverse=False,
                 input_history=5, use_attention=False, input_right_context=0,
                 output_history=1, separate_symbol_history=False, step_history=1,
                 use_output_attention=False, history_embedding_size=32,
                 use_embeddings=False, embeddings_size=8,
                 use_feature_embeddings=False, feature_embeddings_size=8,
                 conv_layers=0, conv_size=32, rnn="lstm", rnn_size=32,
                 output_rnn_size=32, dense_output_size=32,
                 regularizer="l2", dropout=0.2, callbacks=None,
                 use_lm=False, min_prob=0.01, max_diff=2.0,
                 random_state=187, verbose=1):
        self.add_no_value = add_no_value
        self.use_entire_features = use_entire_features
        self.models_number = models_number
        self.buckets_number = buckets_number
        self.batch_size = batch_size
        self.nepochs = nepochs
        self.validation_split = validation_split
        self.reverse = reverse
        self.input_history = input_history
        self.input_right_context = input_right_context
        self.use_attention = use_attention
        self.output_history = output_history
        self.separate_symbol_history = separate_symbol_history
        self.step_history = step_history
        self.use_output_attention = use_output_attention
        self.history_embedding_size = history_embedding_size
        self.use_embeddings = use_embeddings
        self.embeddings_size = embeddings_size
        self.use_feature_embeddings = use_feature_embeddings
        self.feature_embeddings_size = feature_embeddings_size
        self.conv_layers = conv_layers
        self.conv_size = conv_size
        self.rnn = rnn
        self.rnn_size = rnn_size
        self.output_rnn_size = output_rnn_size
        self.dense_output_size = dense_output_size
        self.regularizer = regularizer
        self.dropout = dropout
        self.callbacks = callbacks
        # декодинг
        self.use_lm = use_lm
        self.min_prob = min_prob
        self.max_diff = max_diff
        # разное
        self.random_state = random_state
        self.verbose = verbose
        # выравнивания
        self._make_aligner(aligner_params)
        # self._initialize_callbacks(model_file)
        self.new = True

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
            if self.new:
                print(self.decoders_[i].layers)
                sys.exit()
        return self

    def to_json(self, outfile, model_file=None):
        info = dict()
        if model_file is None:
            pos = outfile.rfind(".")
            model_file = outfile[:pos] + ("-model.hdf5" if pos != -1 else "-model")
        model_files = [self._make_model_file(model_file, i+1) for i in range(models_number)]
        for i in range(self.models_number):
            model_files[i] = os.path.abspath(model_files[i])
        for (attr, val) in inspect.getmembers(self):
            if not (attr.startswith("__") or inspect.ismethod(val) or
                    isinstance(getattr(AGInflector, attr, None), property) or
                    attr.isupper() or attr in ["callbacks", "models_"]):
                info[attr] = val
            elif attr == "models_":
                info["model_files"] = model_files
                for model, curr_model_file in zip(self.models_, model_files):
                    model.save_weights(curr_model_file)
            elif attr == "callbacks":
                for callback in val:
                    if isinstance(callback, EarlyStopping):
                        info["early_stopping_callback"] = {"patience": callback.patience,
                                                           "monitor": callback.monitor}
                    elif isinstance(callback, ReduceLROnPlateau):
                        curr_val = dict()
                        for key in ["patience", "monitor", "factor"]:
                            curr_val[key] = getattr(callback, key)
                        info["reduce_LR_callback"] = curr_val
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
    def labels_number(self):
        return len(self.labels_)

    @property
    def feature_vector_size(self):
        answer = self.labels_number + self.feature_offsets_[-1]
        if self.use_entire_features:
            answer += len(self.features_)
        return answer


    def _make_vocabulary(self, X):
        symbols = {a for first, _, second in X for a in first+second}
        self.symbols_ = self.AUXILIARY + [STEP] + sorted(symbols)
        self.symbol_codes_ = {a: i for i, a in enumerate(self.symbols_)}
        return self

    def _make_features(self, descrs):
        """
        Extracts possible POS values, feature names and values
        from the tuples of feature names and feature values

        Parameters:
        -----------
            descrs: list of pairs of tuples,
                descrs = [descr_1, ..., descr_m],
                descr = [("pos", cat_1,..., cat_k), (pos, val_1, ..., val_k)]

        Returns:
        -----------
            self
        """
        pos_labels = set()
        feature_values = defaultdict(set)
        self.features_ = {"UNKNOWN_FEATURE": self.UNKNOWN_FEATURE}
        for cats, values in descrs:
            pos_label = values[0]
            pos_labels.add(pos_label)
            for cat, value in zip(cats[1:], values[1:]):
                cat = "_".join([cat, pos_label])
                feature_values[cat].add(value)
            if (cats, values) not in self.features_:
                self.features_[(cats, values)] = len(self.features_)
        self.labels_ = self.AUXILIARY + sorted(pos_labels)
        self.label_codes_ = {x: i for i, x in enumerate(self.labels_)}
        self.feature_values_, self.feature_codes_ = [], dict()
        for i, (feat, values) in enumerate(sorted(feature_values.items())):
            if self.add_no_value:
                values.add("NO_VALUE")
            self.feature_values_.append({value: j for j, value in enumerate(values)})
            self.feature_codes_[feat] = i
        self.feature_offsets_ = np.concatenate(
            ([0], np.cumsum([len(x) for x in self.feature_values_], dtype=np.int32)))
        self.feature_offsets_ = [int(x) for x in self.feature_offsets_]
        return self

    def extract_features(self, descr):
        answer = np.zeros(shape=(self.feature_vector_size,), dtype=np.uint8)
        cats, values = descr
        pos_label = values[0]
        label_code = self.label_codes_.get(pos_label)
        free_features = set(feature for feature in self.feature_codes_
                            if feature.split("_")[1] == pos_label)
        if label_code is not None:
            answer[label_code] = 1
            for feature, value in zip(cats[1:], values[1:]):
                feature = feature + "_" + pos_label
                free_features.discard(feature)
                feature_code = self.feature_codes_.get(feature)
                if feature_code is not None:
                    value_code = self.feature_values_[feature_code].get(value)
                    if value_code is not None:
                        value_code += self.feature_offsets_[feature_code]
                        answer[value_code + self.labels_number] = 1
        if self.add_no_value:
            for feature in free_features:
                feature_code = self.feature_codes_[feature]
                value_code = self.feature_values_[feature_code]["NO_VALUE"]
                value_code += self.feature_offsets_[feature_code]
                answer[value_code + self.labels_number] = 1
        if self.use_entire_features:
            values_encoding_size = self.labels_number + self.feature_offsets_[-1]
            answer[values_encoding_size+
                   self.features_.get(descr, self.UNKNOWN_FEATURE)] = 1
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
        answer = np.full(shape=(bucket_size, bucket_length, self.output_history),
                         fill_value=BEGIN, dtype=int)
        if targets is not None:
            for i in range(1, bucket_length):
                d = i - self.output_history
                answer[:,i,max(-d,0):] = targets[:,max(d, 0):i]
        # answers = np.empty(shape=(bucket_size, bucket_length, self.output_history), dtype=int)
        # answers[:,0] = BEGIN
        # if targets is not None:
        #     answers[:,1:] = targets[:,:-1]
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

    def transform_training_data(self, lemmas, features, letter_positions, targets,
                                save_bucket_lengths=False):
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
        if save_bucket_lengths:
            self.bucket_lengths_ = make_bucket_lengths(alignment_lengths, self.buckets_number)
        buckets_with_indexes = collect_buckets(alignment_lengths, self.bucket_lengths_)
        data_by_buckets = [self._make_bucket_data(lemmas, length, indexes)
                           for length, indexes in buckets_with_indexes]
        features_by_buckets = [
            np.array([self.extract_features(features[i])
                      for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        targets = np.array([[self.symbol_codes_[x] for x in elem] for elem in targets])
        letter_positions_by_buckets = [
            _make_table(letter_positions, length, indexes, fill_with_last=True)
            for length, indexes in buckets_with_indexes]
        targets_by_buckets = [_make_table(targets, length, indexes, fill_value=PAD)
                              for length, indexes in buckets_with_indexes]
        answer = list(zip(data_by_buckets, features_by_buckets,
                          letter_positions_by_buckets))
        def _make_bucket_data(func):
            return [func(L, len(indexes), table) for table, (L, indexes)
                    in zip(targets_by_buckets, buckets_with_indexes)]
        if self.separate_symbol_history:
            steps_by_buckets = _make_bucket_data(self._make_steps_shifted_output)
            history_by_buckets = _make_bucket_data(self._make_symbols_shifted_output)
            for i in range(len(history_by_buckets)):
                answer[i] += (history_by_buckets[i], steps_by_buckets[i])
        else:
            history_by_buckets = _make_bucket_data(self._make_shifted_output)
            for i in range(len(history_by_buckets)):
                answer[i] += (history_by_buckets[i],)
        for i in range(len(history_by_buckets)):
            answer[i] += (targets_by_buckets[i],)
        return answer, [elem[1] for elem in buckets_with_indexes]

    def _preprocess(self, data, alignments, to_fit=True, alignments_outfile=None):
        to_align = [(elem[0], elem[2]) for elem in data]
        if alignments is None:
            # вычисляем выравнивания
            alignments = self.aligner.align(to_align, to_fit=to_fit)
            if alignments_outfile is not None:
                output_data(to_align, alignments, alignments_outfile, sep="_")
        elif to_fit:
            self.aligner.align(to_align, to_fit=True, only_initial=True)
        indexes_with_targets = [make_alignment_indexes(
            alignment, reverse=self.reverse) for alignment in alignments]
        lemmas = [elem[0] for elem in data]
        features = [elem[1] for elem in data]
        letter_indexes = [elem[0] for elem in indexes_with_targets]
        targets = [elem[1] for elem in indexes_with_targets]
        return self.transform_training_data(
            lemmas, features, letter_indexes, targets, save_bucket_lengths=to_fit)

    def train(self, data, alignments=None, dev_data=None, dev_alignments=None,
              alignments_outfile=None, model_file=None, save_file=None):
        """

        Parameters:
        -----------
            data: list of tuples,
                data = [(lemma, problem_descr, word), ..]
                problem_descr = {"pos": pos, feat_1: val_1, ..., feat_d: val_d}
            alignment(optional): list of lists of pairs of strs or None,
                list of 0-1,1-0,0-0 alignments for each lemma-word pair in data,
                alignment = [[(i1, o1), ..., (ir, or)], ...]

        Returns:
        -----------
            self

        """
        self._make_vocabulary(data)
        self._make_features([elem[1] for elem in data])
        data_by_buckets, buckets_indexes = self._preprocess(
            data, alignments, to_fit=True, alignments_outfile=alignments_outfile)
        if dev_data is not None:
            dev_data_by_buckets, dev_bucket_indexes =\
                self._preprocess(dev_data, dev_alignments, to_fit=False)
        else:
            dev_data_by_buckets, dev_bucket_indexes = None, None
        self.build()
        if save_file is not None:
            self.to_json(save_file, model_file)
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

    def _build_model(self):
        RNN = kl.GRU if self.rnn.lower() == "gru" else kl.LSTM
        # входные данные
        symbol_inputs = kl.Input(name="symbol_inputs", shape=(None,), dtype='uint8')
        feature_inputs = kl.Input(
            shape=(self.feature_vector_size,), name="feature_inputs", dtype='uint8')
        letter_indexes_inputs = kl.Input(shape=(None,), dtype='uint8')
        shifted_target_inputs = kl.Input(
            shape=(None, self.output_history), name="shifted_target_inputs", dtype='uint8')
        inputs = [symbol_inputs, feature_inputs,
                  letter_indexes_inputs, shifted_target_inputs]
        if self.separate_symbol_history:
            step_target_inputs = kl.Input(shape=(None, self.step_history),
                                          name="step_target_inputs", dtype='uint8')
            inputs.append(step_target_inputs)
        # буквы входного слова
        if self.use_embeddings:
            embedding = kl.Embedding(self.symbols_number, self.embeddings_size,
                                     embeddings_regularizer=self.regularizer)
        else:
            embedding = kl.Lambda(kb.one_hot, arguments={"num_classes": self.symbols_number},
                                  output_shape=(None, self.symbols_number))
        embedded_inputs = embedding(symbol_inputs)
        lstm_outputs = kl.Bidirectional(RNN(self.rnn_size, return_sequences=True),
                                        name="lstm_outputs")(embedded_inputs)
        lstm_outputs_size = 2 * self.rnn_size
        # сохраняем кодировщик
        # здесь надо задать форму матрицы индексов
        def indexing_function(x):
            A, B = x[0], x[1]
            C = A[kb.arange(A.shape[0]).reshape((-1,1)), B]
            return C
        lambda_func = kl.Lambda(indexing_function,
                                input_shape=[(None, lstm_outputs_size), (None,)],
                                output_shape=(None, lstm_outputs_size),
                                name="aligned_lstm_outputs")
        aligned_lstm_outputs = lambda_func([lstm_outputs,  letter_indexes_inputs])
        if self.dropout > 0.0:
            to_decoder = kl.Dropout(self.dropout)(aligned_lstm_outputs)
        else:
            to_decoder = aligned_lstm_outputs
        # признаки
        if self.use_feature_embeddings:
            feature_embeddings = kl.Dense(
                self.feature_embeddings_size, input_shape=(self.feature_vector_size,),
                activation="relu", use_bias=False)(feature_inputs)
            feature_inputs_length = self.feature_embeddings_size
        else:
            feature_embeddings = feature_inputs
            feature_inputs_length = self.feature_vector_size
        feature_embeddings = kl.Lambda(
            repeat_, arguments={"k": aligned_lstm_outputs.shape[1]},
            output_shape=(None, feature_inputs_length))(feature_embeddings)
        to_concatenate = [to_decoder, feature_embeddings]
        if not self.use_output_attention:
            # строим функцию, готовящую истории
            shifted_outputs = kl.Lambda(
                kb.one_hot, arguments={"num_classes": self.symbols_number},
                output_shape=(None, self.output_history, self.symbols_number))(shifted_target_inputs)
            lambda_func = kl.Lambda(
                kb.flatten, output_shape=(self.output_history*self.symbols_number,))
            shifted_outputs = kl.TimeDistributed(lambda_func)(shifted_outputs)
            # shifted_outputs = kl.TimeDistributed(history_preparation_layer)(shifted_target_inputs)
            to_concatenate.append(shifted_outputs)
            if self.separate_symbol_history:
                to_concatenate.append(step_target_inputs)
        else:
            raise NotImplementedError
            # outputs_encoding = RNN(self.history_embedding_size,
            #                        return_sequences=True)(shifted_outputs)
            # if self.dropout > 0.0:
            #     outputs_encoding = kl.Dropout(self.dropout)(outputs_encoding)
            # shifted_outputs = HistoryAttention(
            #     shifted_target_inputs, outputs_encoding, self.output_history,
            #     self.symbols_number, self.history_embedding_size)
            # shifted_outputs_size = 2 * self.history_embedding_size
        # соединяем всё
        decoder_inputs = kl.Concatenate(name="decoder_inputs")(to_concatenate)
        if self.new:
            dsh_lstm_params = {"dense_units": self.dense_output_size,
                               "dense_activation": "tanh", "output_dropout": self.dropout}
            decoder_layer = DSH_LSTM(
                self.output_rnn_size, self.symbols_number, name="outputs",
                return_sequences=True, activity_regularizer=keras.regularizers.l2(0.0001),
                **dsh_lstm_params)
            outputs = decoder_layer(decoder_inputs)
        else:
            decoder_outputs = RNN(self.output_rnn_size, return_sequences=True)(decoder_inputs)
            if self.dropout > 0.0:
                decoder_outputs = kl.Dropout(self.dropout)(decoder_outputs)
            decoder_outputs = kl.TimeDistributed(kl.Dense(
                self.dense_output_size, activation="tanh", use_bias=False,
                input_shape=(self.output_rnn_size,)))(decoder_outputs)
            outputs = kl.TimeDistributed(kl.Dense(
                self.symbols_number, activation="softmax", name="outputs",
                activity_regularizer=keras.regularizers.l2(0.0001),
                input_shape=(self.dense_output_size,)))(decoder_outputs)
        model = Model(inputs, outputs)
        model.compile(optimizer=adam(clipnorm=5.0),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        # for i, layer in enumerate(model.layers):
        #     print("Layer {}:".format(i), layer.name)
        #     # theano.printing.debugprint(layer.input)
        #     theano.printing.debugprint(layer.output)
        #     print("")
        # sys.exit()
        encoder = kb.function([symbol_inputs], [lstm_outputs])
        decoder_inputs = [aligned_lstm_outputs, feature_inputs, shifted_target_inputs]
        if self.separate_symbol_history:
            decoder_inputs.append(step_target_inputs)
        # symbol_inputs нужно, чтобы не возникала ошибка в dropout
        if self.new:
            # args = decoder_inputs + [symbol_inputs, letter_indexes_inputs, kb.learning_phase()]
            # cell inputs
            cell_lstm_outputs = kl.Input(
                shape=(lstm_outputs_size,), name="cell_lstm_outputs", dtype='float32')
            cell_feature_embeddings = kl.Input(
                shape=(feature_inputs_length,), name="cell_feature_embeddings", dtype='float32')
            cell_shifted_answers = kl.Input(
                shape=(self.output_history,), name="cell_shifted_answers", dtype='uint8')
            cell_inputs = [cell_lstm_outputs, cell_feature_embeddings, cell_shifted_answers]
            if self.separate_symbol_history:
                cell_shifted_steps = kl.Input(
                    shape=(self.step_history,), name="cell_shifted_steps", dtype='uint8')
                cell_inputs.append(cell_shifted_steps)
            # history embeddings
            history_preparation_layer =\
                kl.Lambda(flattened_one_hot,
                          arguments={"num_classes": self.symbols_number},
                          output_shape=(self.output_history*self.symbols_number,))
            cell_history = history_preparation_layer(cell_shifted_answers)
            cell_to_concatenate = cell_inputs[:]
            cell_to_concatenate[2] = cell_history
            cell_to_decoder = kl.Concatenate()(cell_to_concatenate)
            # состояния LSTM с предыдущего шага
            cell_h_state = kl.Input(shape=(self.output_rnn_size,), dtype='float32')
            cell_c_state = kl.Input(shape=(self.output_rnn_size,), dtype='float32')
            cell_states = [cell_h_state, cell_c_state]
            cell = DSH_LSTMCell(self.output_rnn_size, self.symbols_number, **dsh_lstm_params)
            cell_outputs = DSH_LSTMCellWrapper(cell, name="cell")(cell_to_decoder, states=cell_states)
            decoder = Model(cell_inputs + cell_states, cell_outputs)
        else:
            args = decoder_inputs + [kb.learning_phase()]
            decoder = kb.function(args, [outputs])
        return model, encoder, decoder

    def _prepare_for_prediction(self, words, features,
                                buckets_with_indexes, known_answers=None):
        encoded_words = np.array(
            [([BEGIN] + [self.symbol_codes_.get(
                x, UNKNOWN) for x in word] + [END]) for word in words])
        encoded_words_by_buckets = [_make_table(encoded_words, length, indexes, fill_value=PAD)
                                    for length, indexes in buckets_with_indexes]
        # представления входных слов после lstm
        # encoder_outputs_by_buckets[i].shape = (self.models, lengths[i])
        encoder_outputs_by_buckets =\
            self._predict_encoder_outputs(encoded_words_by_buckets)
        features_by_buckets = [
            np.array([self.extract_features(features[i]) for i in bucket_indexes])
            for _, bucket_indexes in buckets_with_indexes]
        if self.separate_symbol_history:
            steps_by_buckets = [self._make_steps_shifted_output(L, len(indexes))
                                for (L, indexes) in buckets_with_indexes]
            shifted_targets_by_buckets = [self._make_symbols_shifted_output(L, len(indexes))
                                          for (L, indexes) in buckets_with_indexes]
        else:
            shifted_targets_by_buckets = [self._make_shifted_output(L, len(indexes))
                                          for (L, indexes) in buckets_with_indexes]
        if known_answers is not None:
            encoded_answers = np.array([([BEGIN] +
                                         [self.symbol_codes_.get(x, UNKNOWN) for x in word] +
                                         [END]) for word in known_answers])
            encoded_answers_by_buckets = [
                _make_table(encoded_answers, length, indexes, fill_value=PAD)
                for length, indexes in buckets_with_indexes]
        else:
            encoded_answers_by_buckets = [None] * len(buckets_with_indexes)
        # predictions = [(words_1, probs_1), ..., (words_m, probs_m)]
        inputs = [encoder_outputs_by_buckets, features_by_buckets,
                  shifted_targets_by_buckets]
        if self.separate_symbol_history:
            inputs.append(steps_by_buckets)
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
        words, features = [elem[0] for elem in data], [elem[1] for elem in data]
        output_lengths = [(2 * len(x) + self.max_length_shift_ + 3) for x in words]
        buckets_with_indexes = collect_buckets(
            output_lengths, self.bucket_lengths_, max_bucket_size=64)
        inputs, encoded_words_by_buckets, encoded_answers_by_buckets =\
            self._prepare_for_prediction(words, features, buckets_with_indexes, known_answers)
        answer = [[] for _ in words]
        for bucket_number, elem in enumerate(zip(*inputs)):
            if verbose > 0:
                print("Bucket {} of {} predicting".format(
                    bucket_number+1, len(buckets_with_indexes)))
            steps_history = elem[3] if self.separate_symbol_history else None
            bucket_words, bucket_probs = self._predict_on_batch(
                elem[0], elem[1], elem[2], steps_history, beam_width=beam_width,
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
        if self.new:
            # начальные состояния lstm. Они различны для каждой модели
            h_states = np.zeros(shape=(M, self.models_number, self.output_rnn_size))
            c_states = np.zeros(shape=(M, self.models_number, self.output_rnn_size))
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