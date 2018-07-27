import sys
from collections import defaultdict
import numpy as np
from sklearn.linear_model import LogisticRegression


from read import read_infile
from pyparadigm.paradigm import LcsSearcher
from pyparadigm.paradigm_detector import *
from neural.neural_LM import NeuralLM
from evaluate import evaluate


LCS_SEARCHER_PARAMS = {"method": "modified_Hulden", "remove_constant_variables": True}
DEFAULT_LM_PARAMS = {"nepochs": 50, "batch_size": 16,
                     "history": 5, "use_feats": True, "use_label": True,
                     "encoder_rnn_size": 64, "decoder_rnn_size": 64, "dense_output_size": 32,
                     "decoder_dropout": 0.2, "encoder_dropout": 0.2,
                     "feature_embeddings_size": 32, "feature_embedding_layers": 1,
                     "use_embeddings": False, "embeddings_size": 32, "use_full_tags": True,
                     "callbacks":
                         {"EarlyStopping": { "patience": 5, "monitor": "val_loss"}}
                     }
LM_KWARGS = {"return_letter_scores": True, "return_log_probs": True}


class ParadigmChecker:

    def __init__(self):
        self.lcs_searcher = LcsSearcher(**LCS_SEARCHER_PARAMS)
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.substitutors = dict()

    def train(self, data):
        paradigms = self.lcs_searcher.calculate_paradigms([tuple(elem[:2]) for elem in data])
        for (_, _, descr), (pattern, _) in zip(data, paradigms):
            descr = tuple(descr)
            self.patterns[descr][pattern] += 1
            if descr not in self.substitutors:
                self.substitutors[pattern] = ParadigmSubstitutor(pattern)
        return self

    def filter(self, data, answers, probs):
        answer = []
        for (word, descr), curr_answers, curr_probs in zip(data, answers, probs):
            patterns = self.patterns[tuple(descr)]
            if len(patterns) == 0:
                answer.append((curr_answers, curr_probs))
                continue
            possible_indexes = set()
            words_to_indexes = {word: i for i, word in enumerate(curr_answers)}
            for pattern in patterns:
                substitutor = self.substitutors[pattern]
                forms = [elem[1] for elem in substitutor._make_all_forms(word)]
                for form in forms:
                    index = words_to_indexes.get(form)
                    if index is not None:
                        possible_indexes.add(index)
                        if len(possible_indexes) == len(curr_answers):
                            break
                if len(possible_indexes) == len(curr_answers):
                    break
            if len(possible_indexes) == 0:
                answer.append((curr_answers, curr_probs))
                continue
            new_answer = [curr_answers[i] for i in sorted(possible_indexes)]
            new_probs = [curr_probs[i] for i in sorted(possible_indexes)]
            answer.append((new_answer, new_probs))
        answer = [list(zip(*elem)) for elem in answer]
        answer = [[[x[0]] + x[1] for x in elem] for elem in answer]
        return answer


class ParadigmLmClassifier:

    def __init__(self, forward_lm=None, reverse_lm=None, lm_params=None,
                 basic_model=None, use_paradigm_counts=False, tune_weights=None,
                 use_letter_scores=False, max_letter_score=-np.log(0.01),
                 max_lm_letter_score=-np.log(0.001),
                 basic_hyps_number=5, lm_hyps_number=5,
                 validation_split=0.2, random_state=187, verbose=1):
        self.lcs_searcher = LcsSearcher(**LCS_SEARCHER_PARAMS)
        self.patterns = defaultdict(lambda: defaultdict(int))
        self.substitutors = dict()
        self.forward_lm = forward_lm
        self.reverse_lm = reverse_lm
        self.lm_params = lm_params or DEFAULT_LM_PARAMS
        self.basic_model = basic_model
        self.use_paradigm_counts = use_paradigm_counts
        self.tune_weights = tune_weights
        self.use_letter_scores = use_letter_scores
        self.max_letter_score = max_letter_score
        self.max_lm_letter_score = max_lm_letter_score
        self.basic_hyps_number = basic_hyps_number
        self.lm_hyps_number = lm_hyps_number
        self.validation_split = validation_split
        self.random_state = random_state
        self.verbose = verbose
        self.predict = (self.predict_with_basic if self.basic_model is not None
                        else self.predict_without_basic)

    @property
    def weights_dim(self):
        return 2 + int(self.basic_model is not None) + int(self.use_paradigm_counts)

    def train(self, data, dev_data=None, save_forward_lm=None, save_reverse_lm=None):
        paradigms = self.lcs_searcher.calculate_paradigms([tuple(elem[:2]) for elem in data])
        for (_, _, descr), (pattern, _) in zip(data, paradigms):
            try:
                descr = tuple(descr)
                if descr not in self.substitutors:
                    self.substitutors[pattern] = ParadigmSubstitutor(pattern)
                self.patterns[descr][pattern] += 1
            except ValueError:
                pass
        if dev_data is None:
            np.random.seed(self.random_state)
            shuffled_data = data[:]
            np.random.shuffle(shuffled_data)
            train_data_size = int(len(data) * (1.0 - self.validation_split))
            data, dev_data = shuffled_data[:train_data_size], shuffled_data[train_data_size:]
        data_for_lm, dev_data_for_lm = [elem[1:] for elem in data], [elem[1:] for elem in dev_data]
        if self.forward_lm is None:
            self.forward_lm = NeuralLM(verbose=self.verbose, **self.lm_params).train(
                data_for_lm, dev_data_for_lm, save_file=save_forward_lm)
        if self.reverse_lm is None:
            self.reverse_lm = NeuralLM(reverse=True, verbose=self.verbose, **self.lm_params).train(
                data_for_lm, dev_data_for_lm, save_file=save_reverse_lm)
        if self.tune_weights:
            X_tune, y_tune = self._generate_data_for_tuning(dev_data)
            self.cls = LogisticRegression().fit(X_tune, y_tune)
            self.weights = self.cls.coef_[0]
            self.weights /= np.linalg.norm(self.weights) / 3
        else:
            self.weights = np.array([1.0, 1.0, 1.0])
        return self

    def _get_lm_score(self, score):
        if self.use_letter_scores:
            return sum([min(x, self.max_lm_letter_score) for x in score[0]])
        else:
            return score[1]

    def _generate_data_for_tuning(self, data):
        possible_forms, indexes, counts = [], [0], []
        descrs_for_prediction = []
        for word, corr_form, descr in data:
            patterns = self.patterns[tuple(descr)]
            curr_forms, counts_by_forms = set(), defaultdict(int)
            for pattern, count in patterns.items():
                substitutor = self.substitutors[pattern]
                forms = {elem[1] for elem in substitutor._make_all_forms(word)}
                for form in forms:
                    counts_by_forms[form] = max(counts_by_forms[form], count)
            curr_forms.add(corr_form)
            if len(curr_forms) == 1 and word != corr_form:
                curr_forms.add(word)
            possible_forms.extend(curr_forms)
            indexes.append(indexes[-1] + len(curr_forms))
            if self.use_paradigm_counts:
                counts.extend(np.log(1.0 + np.array([counts_by_forms[form] for form in list(curr_forms)])))
            else:
                counts.extend([0] * len(curr_forms))
            descrs_for_prediction.extend([descr] * len(curr_forms))
        to_predict = list(zip(possible_forms, descrs_for_prediction))
        forward_scores = self.forward_lm.predict(to_predict)
        reverse_scores = self.reverse_lm.predict(to_predict)
        X, y = [], []
        for i, ((word, corr_form, _), start) in enumerate(zip(data, indexes[:-1])):
            end = indexes[i+1]
            if start == end + 1:
                continue
            curr_forms = possible_forms[start:end]
            if corr_form not in curr_forms:
                print(i, word, curr_forms)
            corr_index = start + curr_forms.index(corr_form)
            arrays = [forward_scores, reverse_scores, counts]
            corr_data = np.array([array[corr_index] for array in arrays], dtype=float)
            curr_data = np.array([np.concatenate([array[start:corr_index], array[corr_index+1:end]])
                                  for array in arrays], dtype=float)
            X_curr = (curr_data.T - corr_data) / 10
            X.extend(np.vstack([X_curr, -X_curr]))
            y.extend([1.0] * (end-start-1) + [0.0] * (end-start-1))
        return X, y

    def _calculate_lm_scores(self, data):
        possible_forms, indexes, counts, descrs_for_prediction = [], [0], [], []
        for word, descr in data:
            patterns = self.patterns[tuple(descr)]
            curr_forms = set()
            counts_by_forms = defaultdict(int)
            for pattern, count in patterns.items():
                substitutor = self.substitutors[pattern]
                forms = {elem[1] for elem in substitutor._make_all_forms(word)}
                curr_forms.update(forms)
                for form in forms:
                    counts_by_forms[form] = max(counts_by_forms[form], count)
            possible_forms.extend(list(curr_forms))
            indexes.append(indexes[-1] + len(curr_forms))
            if self.use_paradigm_counts:
                counts.extend(np.log(1.0 + np.array([counts_by_forms[form] for form in list(curr_forms)])))
            descrs_for_prediction.extend([descr] * len(curr_forms))
        to_predict = list(zip(possible_forms, descrs_for_prediction))
        return possible_forms, indexes, self._collect_lm_scores(to_predict, counts)

    def _collect_lm_scores(self, data, counts=None):
        forward_scores = [self._get_lm_score(x) for x in self.forward_lm.predict(data, **LM_KWARGS)]
        reverse_scores = [self._get_lm_score(x) for x in self.reverse_lm.predict(data, **LM_KWARGS)]
        for_prediction = [forward_scores, reverse_scores]
        if self.use_paradigm_counts:
            for_prediction.append(counts if (counts is not None) else ([0.0] * len(data)))
        return np.array(for_prediction, dtype=float).T

    def _predict_forms(self, scores, forms, indexes, source_forms=None, n=5, min_prob=0.01):
        answer = []
        for i, start in enumerate(indexes[:-1]):
            end = indexes[i + 1]
            if start == end:
                answer.append(([source_forms[i]], [1.0]))
                continue
            curr_forms, curr_scores = forms[start:end], scores[start:end]
            curr_probs = np.exp(-curr_scores) / np.sum(np.exp(-curr_scores))
            form_indexes = np.argsort(curr_probs)[::-1]
            forms_to_return, probs = [], []
            for index in form_indexes[:n]:
                prob = curr_probs[index]
                if prob < min_prob:
                    break
                forms_to_return.append(curr_forms[index])
                probs.append(prob)
            answer.append((forms_to_return, probs))
        return answer

    def predict_without_basic(self, data, n=5, min_prob=0.01):
        possible_forms, indexes, for_prediction = self._calculate_lm_scores(data)
        scores = np.dot(for_prediction, self.weights) / 10
        source_forms = [elem[0] for elem in data]
        return self._predict_forms(
            scores, possible_forms, indexes, source_forms, n=n, min_prob=min_prob)

    def _transform_basic_score(self, score):
        return sum(min(x, self.max_letter_score) for x in score[1])

    def predict_with_basic(self, data, n=5, min_prob=0.01):
        basic_model_params = {"feat_column": 1, "return_log": True, "log_base": 2.0,
                              "beam_width": self.basic_hyps_number}
        basic_predictions = self.basic_model.predict(data, **basic_model_params)
        possible_lm_forms, group_bounds, lm_scores = self._calculate_lm_scores(data)
        group_forms, group_bounds_for_lm, group_bounds_for_basic = [], [0], [0]
        indexes_for_lm, indexes_for_basic = [], []
        scores = []
        new_data_for_lm, new_forms_for_basic = [], []
        for i, (curr_basic_predictions, start) in enumerate(zip(basic_predictions, group_bounds)):
            end = group_bounds[i+1]
            curr_basic_forms = [elem[0] for elem in curr_basic_predictions]
            curr_basic_scores = [self._transform_basic_score(elem) for elem in curr_basic_predictions]
            curr_lm_forms = possible_lm_forms[start:end]
            # new_bound_for_lm, new_bound_for_basic = group_bounds_for_lm[-1], group_bounds_for_basic[-1]
            curr_scores = []
            for j, (form, basic_score) in enumerate(zip(curr_basic_forms, curr_basic_scores)):
                if form in curr_lm_forms:
                    index = curr_lm_forms.index(form)
                    curr_score = [basic_score] + list(lm_scores[start+index])
                else:
                    curr_score = [basic_score] + [0.0] * (self.weights_dim - 1)
                    indexes_for_lm.append((i, j))
                    new_data_for_lm.append((form, data[i][1]))
                curr_scores.append(curr_score)
            lm_indexes = np.argsort(np.sum(lm_scores[start:end], axis=1))[:self.lm_hyps_number]
            curr_lm_forms = [curr_lm_forms[j] for j in lm_indexes]
            curr_lm_scores = [lm_scores[start+j] for j in lm_indexes]
            for form, score in zip(curr_lm_forms, curr_lm_scores):
                if form not in curr_basic_forms:
                    indexes_for_basic.append((i, len(curr_scores)))
                    curr_scores.append([0.0] + list(score))
                    curr_basic_forms.append(form)
                    new_forms_for_basic.append(form)
            scores.append(np.array(curr_scores))
            # group_bounds_for_lm.append(len(new_forms_for_lm))
            # group_bounds_for_basic.append(len(new_forms_for_basic))
            group_forms.extend(curr_basic_forms)
        new_data_for_basic = [data[i] for i, j in indexes_for_basic]
        new_basic_predictions = self.basic_model.predict(
            new_data_for_basic, known_answers=new_forms_for_basic, **basic_model_params)
        for (i, j), elem in zip(indexes_for_basic, new_basic_predictions):
            scores[i][j, 0] = self._transform_basic_score(elem[0])
        new_lm_scores = self._collect_lm_scores(new_data_for_lm)
        for (i, j), elem in zip(indexes_for_lm, new_lm_scores):
            scores[i][j, 1:] = elem
        indexes = [0] + list(np.cumsum([len(x) for x in scores]))
        scores = np.concatenate(scores, axis=0)
        # scores[:,1:] /= 2.5
        scores = np.dot(scores, self.weights)
        return self._predict_forms(scores, group_forms, indexes, n=n, min_prob=min_prob)





if __name__ == "__main__":
    infile = "conll2018/task1/all/belarusian-train-medium"
    data = read_infile(infile)
    paradigm_checker = ParadigmChecker()
    paradigm_checker.train(data)