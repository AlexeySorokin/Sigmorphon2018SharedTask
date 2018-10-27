from collections import defaultdict, OrderedDict
import numpy as np
import os
import sys
import bisect

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

from read import read_infile

class LabeledNgramModel:

    def __init__(self, max_ngram_length=3, all_ngram_length=3,
                 min_count=1, min_symbol_prob=0.01, reverse=False,
                 seed=157):
        self.max_ngram_length = max(max_ngram_length, all_ngram_length)
        self.all_ngram_length = all_ngram_length
        self.min_count = min_count
        self.reverse = reverse
        self.min_symbol_prob = min_symbol_prob
        self.seed = seed
        np.random.seed(self.seed)

    @property
    def labels_number(self):
        return len(self.labels_)

    def train(self, X):
        self.labels_ = sorted(set(elem[1] for elem in X))
        self.label_codes_ = {label: i for i, label in enumerate(self.labels_, 1)}
        counts = [defaultdict(self.default_array_func) for i in range(self.max_ngram_length)]
        for word, label in X:
            if self.reverse:
                word = word[::-1]
            word = "^" + word + "$"
            for L in range(1, min(self.max_ngram_length, len(word)) + 1):
                for start in range(len(word) - L + 1):
                    ngram = word[start:start+L]
                    counts[L - 1][ngram][0] += 1
                    counts[L - 1][ngram][self.label_codes_[label]] += 1
        self.max_length_ = max(len(word) for word, _ in X)
        self._make_trie(counts)
        return self

    @property
    def default_array_func(self, dtype="float"):
        return (lambda: np.array([0] * (self.labels_number + 1), dtype=dtype))

    def _make_trie(self, counts):
        self.trie_ = OrderedDict()
        total_counts = np.array([0] * (self.labels_number + 1), dtype="float")
        for letter, letter_counts in counts[0].items():
            if letter != "^":
                total_counts += letter_counts
        continuations_count = len(counts[0]) - 1
        total_counts[0] += (continuations_count + 1) * self.labels_number
        total_counts[1:] += (continuations_count + 1)
        self.unknown_word_prob = np.concatenate([[self.labels_number], [1.0] * self.labels_number]) / total_counts
        prev_nodes = [""]
        for L, curr_counts in enumerate(counts):
            curr_nodes = []
            continuations = defaultdict(lambda: defaultdict(self.default_array_func))
            continuation_counts = defaultdict(self.default_array_func)
            for ngram, ngram_counts in curr_counts.items():
                history, letter = ngram[:-1], ngram[-1]
                if L < self.all_ngram_length or ngram_counts[0] >= self.min_count:
                    if ngram != "^":
                        continuations[history][letter] = ngram_counts
                        continuation_counts[history] += np.minimum(ngram_counts, 1)
                    curr_nodes.append(ngram)
            for history in prev_nodes:
                if L > 0:
                    # history_counts = counts[L-1][history]
                    if len(continuations[history]) > 0:
                        history_counts = np.sum(list(continuations[history].values()), axis=0)
                    else:
                        history_counts = self.default_array_func()
                    history_cont_counts = continuation_counts[history]
                    alpha_pos = np.nan_to_num((history_counts[1:] - history_cont_counts[1:]) / history_counts[1:])
                    alpha_hist = np.nan_to_num(history_counts / (history_counts + history_cont_counts))
                else:
                    alpha_pos = np.zeros(shape=(self.labels_number,), dtype=float)
                    alpha_hist = np.zeros(shape=(self.labels_number+1,), dtype=float)
                children = dict()
                for letter, letter_counts in continuations[history].items():
                    if letter == "^":
                        continue
                    if L == 0:
                        letter_probs = np.copy(letter_counts)
                        letter_probs[0] += self.labels_number
                        letter_probs[1:] += 1
                        letter_probs /= total_counts
                    else:
                        letter_probs = np.zeros(shape=(self.labels_number+1), dtype="float")
                        parent_letter_probs = self.trie_[history[1:]][2][letter]
                        letter_probs[0] = alpha_hist[0] * letter_counts[0] / history_counts[0]
                        letter_probs[0] += (1 - alpha_hist[0]) * parent_letter_probs[0]
                        # (alpha+beta) * p(w | h, T) + (1 - alpha) p0(w | h, T) + (1 - beta) p(w | h', T)
                        letter_probs[1:] =\
                            (alpha_hist[1:]+alpha_pos) * np.nan_to_num(letter_counts / history_counts)[1:]
                        letter_probs[1:] += (1.0 - alpha_hist[1:]) * parent_letter_probs[1:]
                        letter_probs[1:] += (1.0 - alpha_pos) * letter_probs[0]
                        letter_probs[1:] /= 2.0
                    children[letter] = letter_probs
                self.trie_[history] = (alpha_pos, alpha_hist, children)
            prev_nodes = curr_nodes
        for history, (alpha_pos, alpha_hist, children) in self.trie_.items():
            children_letter_codes = list(children)
            children_letters = {letter: i for i, letter in enumerate(children)}
            children_probs = np.array(list(children.values()))
            self.trie_[history] = (alpha_pos, alpha_hist, children_letter_codes,
                                   children_letters, children_probs)
        return self

    def prob(self, history, letter, label=None, return_history=False):
        label = self.label_codes_.get(label, 0)
        while history not in self.trie_:
            history = history[1:]
        alpha_pos, alpha_hist, _, children_letters, children_probs = self.trie_[history]
        prob = None
        code = children_letters.get(letter)
        if code is not None:
            prob = children_probs[code, label]
            history += letter
        elif history != "":
            if label > 0:
                first_coef = 0.5 * (1.0 - alpha_hist[label])
                second_coef = 0.5 * (1.0 - alpha_pos[label-1]) * (1.0 - alpha_hist[0])
            else:
                first_coef, second_coef = 0.0, (1.0 - alpha_hist[0])
            while True:
                history = history[1:]
                alpha_pos, alpha_hist, _, children_letters, children_probs = self.trie_[history]
                code = children_letters.get(letter)
                if code is not None:
                    probs = children_probs[code]
                    prob = first_coef * probs[label] + second_coef * probs[0]
                    history += letter
                    break
                if history == "":
                    break
                if label > 0:
                    second_coef += 0.5 * first_coef * (1.0 - alpha_pos[label-1])
                    first_coef *= 0.5 * (1.0 - alpha_hist[label])
                second_coef *= (1.0 - alpha_hist[0])
        else:
            first_coef, second_coef = 1.0, 0.0
        if prob is None:
            history = ""
            prob = first_coef * self.unknown_word_prob[label] + second_coef * self.unknown_word_prob[0]
        return (prob, history) if return_history else prob

    def score(self, word, label=None, return_letter_probs=False):
        history = "^"
        probs, score = [], 0.0
        if self.reverse:
            word = word[::-1]
        for letter in word + "$":
            prob, history = self.prob(history, letter, label, return_history=True)
            probs.append((letter, prob))
            score += -np.log(prob)
        if return_letter_probs:
            return score, probs
        else:
            return score

    def generate_word(self, label, return_probs=False):
        label = self.label_codes_.get(label, 0)
        history, word, probs = "^", "", []
        for i in range(self.max_length_ + 4):
            low, up, sample_label = 0.0, 1.0, label
            while history not in self.trie_ or len(self.trie_[history][2]) == 0:
                history = history[1:]
            sample_history = history
            while True:
                if sample_history == "":
                    curr_letters, curr_probs = self.trie_[""][2], self.trie_[""][4][:, sample_label]
                    break
                alpha_pos, alpha_hist, children_letters, _, children_probs = self.trie_[sample_history]
                # выбираем, из какого распределения сэмплировать
                if sample_label > 0:
                    first = 0.5 * (alpha_pos[sample_label - 1] + alpha_hist[sample_label])
                    second = 0.5 * (1 + alpha_pos[sample_label-1])
                    levels = [first, second, 1.0]
                else:
                    levels = [alpha_hist[0], 1]
                distribution_index = bisect.bisect_left(levels, np.random.uniform())
                if distribution_index == 0:
                    curr_letters, curr_probs = children_letters, children_probs[:, sample_label]
                    break
                elif distribution_index == 1:
                    sample_history = sample_history[1:]
                elif distribution_index == 2:
                    sample_label = 0
            curr_probs = np.cumsum(curr_probs)
            coin = np.random.uniform(0, curr_probs[-1])
            index = bisect.bisect_left(curr_probs, coin)
            letter = curr_letters[index]
            prob, history = self.prob(history, letter, label, return_history=True)
            probs.append(prob)
            if letter != "$":
                word += letter
            else:
                break
        if self.reverse:
            word = word[::-1]
            probs[:-1] = probs[-2::-1]
        return (word, probs) if return_probs else word

    def __str__(self):
        answer = ""
        for key, (alpha_pos, alpha_hist, children) in self.trie_.items():
            answer += "\t".join([key, " ".join("{:.2f}".format(x) for x in alpha_pos),
                                 " ".join("{:.2f}".format(x) for x in alpha_hist)]) + "\n"
            for letter, probs in sorted(children.items()):
                answer += "{}\t{}\n".format(letter, " ".join("{:.2f}".format(x) for x in probs))
            answer += "\n"
        return answer


language, mode = "belarusian", "low"
corr_dir = os.path.join("..", "conll2018", "task1", "all")
infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
data = read_infile(infile)
data = [(elem[0], elem[2][0]) for elem in data]
model = LabeledNgramModel(max_ngram_length=3, all_ngram_length=3, min_count=2, reverse=True).train(data)
for _ in range(20):
    label = np.random.choice(list("NVA"))
    word, letter_probs = model.generate_word(label, return_probs=True)
    print(word, label)
print(" ".join("{}:{:.3f}".format(*elem) for elem in zip(word + "$", letter_probs)))
# dev_file = os.path.join(corr_dir, "{}-dev".format(language, mode))
# dev_data = read_infile(dev_file)
# dev_data = [(elem[0], elem[2][0]) for elem in dev_data]
# scores = []
# for word, label in dev_data[:10]:
#     score, letter_probs = model.score(word, label, return_letter_probs=True)
#     scores.extend(-np.log(elem[1]) for elem in letter_probs)
#     print(word, "{:.3f}".format(score))
#     print(" ".join("{}:{:.3f}".format(*elem) for elem in letter_probs))
# print("{:.3f}".format(np.mean(scores)))
