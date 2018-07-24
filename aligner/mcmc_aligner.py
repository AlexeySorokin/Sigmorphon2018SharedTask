import sys
import getopt
from collections import defaultdict
from time import time

import numpy as np

from pyparadigm.pyparadigm import LcsSearcher, fill_vars
from aligner._aligner import CythonAlignerImpl

MAX_DIFF = 50

np.set_printoptions(precision=2)

def log_add(x, y):
    if y > x:
        # делаем x > y
        y, x = x, y
    # -log(exp(-x) + exp(-y)) = x - log(1 + exp(x - y))
    if x - y > MAX_DIFF:
        return y
    return x - np.log(1.0 + np.exp(x - y))


def draw_min(x, y, z):
    return (-1 if x <= z else 1) if x <= y else (0 if y < z else 1)


def draw_sample(x, y, z):
    min_prob = min(x, y, z)
    if min_prob > 2:
        subv = min_prob - 2
        x, y, z = x - subv, y - subv, z - subv
    x, y, z = np.exp(-x), np.exp(-y), np.exp(-z)
    r = np.random.uniform() * (x + y + z)
    return -1 if r < x else 0 if r < x + y else 1


class AlignerImpl:

    EMPTY = 0

    def __init__(self, n_iter=10, prior=0.1, random_state=187, verbose=1):
        self.n_iter = n_iter
        self.prior = prior
        self.random_state = random_state
        self.verbose = verbose

    def _get_cost(self, x, y):
        cost = (self.counts[(x, y)] + self.prior) / (
            self.total_count + self.pairs_number * self.prior)
        return -np.log(cost)

    def _make_trellis(self, x, y):
        m, n = len(x), len(y)
        trellis = np.zeros(shape=(m+1, n+1), dtype=float)
        for j, b in enumerate(y, 1):
            trellis[0][j] = trellis[0][j-1] + self._get_cost(self.EMPTY, b)
        for i, a in enumerate(x, 1):
            trellis[i][0] = trellis[i-1][0] + self._get_cost(a, self.EMPTY)
        for i, a in enumerate(x, 1):
            for j, b in enumerate(y, 1):
                trellis[i][j] = log_add(
                    log_add(trellis[i-1][j] + self._get_cost(a, self.EMPTY),
                            trellis[i][j-1] + self._get_cost(self.EMPTY, b)),
                    trellis[i-1][j-1] + self._get_cost(a, b))
        return trellis

    def sample_path(self, trellis, x, y, best=False):
        m, n = trellis.shape
        i, j = m - 1, n - 1
        path = []
        while i + j > 0:
            if i == 0:
                j -= 1
                path.append((self.EMPTY, y[j]))
            elif j == 0:
                i -= 1
                path.append((x[i], self.EMPTY))
            else:
                func = draw_min if best else draw_sample
                move_key = func(trellis[i-1][j] + self._get_cost(x[i-1], self.EMPTY),
                                trellis[i-1][j-1] + self._get_cost(x[i-1], y[j-1]),
                                trellis[i][j-1] + self._get_cost(self.EMPTY, y[j-1]))
                pair = ((x[i-1], self.EMPTY) if move_key == -1 else
                        (x[i-1], y[j-1]) if move_key == 0 else (self.EMPTY, y[j-1]))
                path.append(pair)
                i = i-1 if move_key <= 0 else i
                j = j-1 if move_key >= 0 else j
        return path[::-1]

    def _make_initial(self, X):
        answer = []
        for i, (first, second) in enumerate(X):
            m, n = len(first), len(second)
            curr = list(zip(first, second))
            if m >= n:
                curr += [(x, self.EMPTY) for x in first[n:]]
            else:
                curr += [(self.EMPTY, y) for y in second[m:]]
            answer.append(curr)
        return answer

    def fit(self, X, pairs=None):
        if pairs is None:
            pairs = self._make_initial(X)
        self.counts = defaultdict(int)
        for elem in pairs:
            for x in elem:
                self.counts[x] += 1
        self.pairs_number = len(self.counts)
        self.total_count = sum(len(elem) for elem in pairs)
        np.random.seed(self.random_state)
        for i in range(self.n_iter):
            t1 = time()
            indexes = list(range(len(X)))
            np.random.shuffle(indexes)
            for j, index in enumerate(indexes, 1):
                first, second = X[index]
                if j % 5000 == 0 and self.verbose > 0:
                    print("Iteration {}, {} objects passed".format(i+1, j))
                trellis = self._make_trellis(first, second)
                for pair in pairs[index]:
                    self.counts[pair] -= 1
                    if self.counts[pair] == 0:
                        self.pairs_number -= 1
                self.total_count -= len(pairs[index])
                curr_pairs = self.sample_path(trellis, first, second)
                for pair in curr_pairs:
                    if self.counts[pair] == 0:
                        self.pairs_number += 1
                    self.counts[pair] += 1
                self.total_count += len(curr_pairs)
                pairs[index] = curr_pairs
            t2 = time()
            if self.verbose > 0:
                print("Iteration {}, {:.2f} seconds elapsed".format(i+1, t2 - t1))
        return self

    def predict(self, X):
        answer = [None] * len(X)
        for i, (x, y) in enumerate(X):
            trellis = self._make_trellis(x, y)
            if i in [1, 129]:
                print(trellis)
            answer[i] = self.sample_path(trellis, x, y, best=True)
        return answer


class Aligner:

    def __init__(self, n_iter=10, prior=0.1, init="greedy", init_params=None,
                 separate_endings=False,
                 words_to_debug=None, verbose=1, random_state=422):
        self.init = init
        self.init_params = init_params
        self.separate_endings = separate_endings
        self.words_to_debug = words_to_debug or set()
        self.verbose = verbose
        self._aligner = CythonAlignerImpl(
            n_iter, prior, separate_endings, verbose, random_state)

    def _make_symbol_table(self, X):
        symbols = set()
        for source, target in X:
            symbols.update(source)
            symbols.update(target)
        self.symbols_ = [""] + sorted(symbols)
        if self.verbose >= 1:
            print(",".join("{}-{}".format(x,i) for i, x in enumerate(self.symbols_)))
        self.encoding_ = {x: i for i, x in enumerate(self.symbols_)}

    @property
    def symbols_number(self):
        return len(self.symbols_)

    @property
    def output_symbols_number(self):
        return len(self.symbols_) * (int(self.separate_endings) + 1)

    def _make_initial(self, X):
        func = self._make_lcs_initial if self.init == "lcs" else self._make_greedy_initial
        return func(X)

    def _make_lcs_initial(self, X):
        if self.init_params is None:
            self.init_params = {"method": "modified_Hulden"}
        self.lcs_searcher = LcsSearcher(**self.init_params)
        paradigms_with_vars = self.lcs_searcher.calculate_paradigms(X, make_string=False)
        segments = [fill_vars(*elem) for elem in paradigms_with_vars]
        answer = []
        for (first_segments, second_segments) in segments:
            curr_answer = []
            for first, second in zip(first_segments, second_segments):
                curr_answer += list(zip(first, second))
                m, n = len(first), len(second)
                if m > n:
                    curr_answer += [(x, "") for x in first[n:]]
                else:
                    curr_answer += [("", y) for y in second[m:]]
            answer.append(curr_answer)
        return answer

    def _make_greedy_initial(self, X):
        """
        Создаёт жадное начальное выравнивание

        :param X:
        :return:
        """
        answer = []
        for first, second in X:
            curr_answer = list(zip(first, second))
            m, n = len(first), len(second)
            if m > n:
                curr_answer += [(x, "") for x in first[n:]]
            else:
                curr_answer += [("", y) for y in second[m:]]
            answer.append(curr_answer)
        return answer

    def encode(self, word):
        answer = []
        for a in word:
            code = self.encoding_.get(a)
            if code is None:
                code = self._tmp_encoding.get(a)
                if code is None:
                    code = len(self._tmp_symbols)
                    self._tmp_symbols.append(a)
                    self._tmp_encoding[a] = code
                code += self.output_symbols_number
            answer.append(code)
        return np.array(answer, dtype=np.intc)

    def decode(self, code):
        if code < len(self.symbols_):
            return self.symbols_[code]
        else:
            return self._tmp_symbols[code-self.output_symbols_number]

    def _encode_alignment(self, alignment):
        """
        Перекодирует выравнивание из пар символов в пары их кодов

        :param alignment:
        :return:
        """
        # к символам окончания прибавляем смещение
        offset = self.symbols_number if self.separate_endings else 0
        answer = [None] * len(alignment)
        for i, (x, y) in enumerate(alignment[::-1], 1):
            if x == y:
                offset = 0  # окончание закончилось
            answer[-i] = (self.encoding_[x], self.encoding_[y]+offset)
        return answer

    def _decode_alignment(self, alignment):
        """
        Перекодирует выравнивание из пар кодов в пары символов

        :param alignment:
        :return:
        """
        answer = [None] * len(alignment)
        for i, (x, y) in enumerate(alignment):
            if y >= self.symbols_number and y < self.output_symbols_number:
                y -= self.symbols_number
            answer[i] = (self.decode(x), self.decode(y))
        return answer

    def _find_word_indexes(self, X, words):
        if words is None:
            words = set()
        else:
            words = set(words)
        words |= set(self.words_to_debug)
        word_indexes = []
        for i, (first, second) in enumerate(X):
            if second in set(words):
                word_indexes.append(i)
        return word_indexes

    def align(self, X, initial=None, to_fit=True, words=None,
              save_initial=None, only_initial=False):
        word_indexes = self._find_word_indexes(X, words)
        # создаём символьную таблицу
        if to_fit:
            self._make_symbol_table(X)
            if initial is None:
                initial = self._make_initial(X)
            if save_initial:
                with open(save_initial, "w", encoding="utf8") as fout:
                    for (first, second), alignment in zip(X, initial):
                        fout.write("{} {}\n{}\n".format(
                            first, second, ",".join("{}-{}".format(*x) for x in alignment)))
        # временная кодировка для символов, не встречавшихся в обучении
        self._tmp_symbols, self._tmp_encoding = [], dict()
        X_encoded = [(self.encode(first), self.encode(second)) for first, second in X]
        if to_fit:
            initial = [self._encode_alignment(elem) for elem in initial]
            n_iter = 0 if only_initial else -1
            self._aligner.fit(X_encoded, initial, set(word_indexes), n_iter)
        answer = self._aligner.predict(X_encoded, set(word_indexes))
        answer = [self._decode_alignment(elem) for elem in answer]
        return answer


def read_infile(infile):
    answer = []
    with open(infile, "r", encoding="utf8") as fin:
        for line in fin:
            line = line.strip()
            splitted = line.split("\t")
            if len(splitted) != 3:
                continue
            answer.append([splitted[0], splitted[2]])
    return answer


def output_alignment(words, alignments, outfile, sep="-"):
    with open(outfile, "w", encoding="utf8") as fout:
        for (first, second), alignment in zip(words, alignments):
            fout.write("{}\t{}\n{}\n".format(
                first, second, ",".join(sep.join(x) for x in alignment)))


SHORT_OPTS = "I:"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    save_initial = None
    for opt, val in opts:
        if opt == "-I":
            save_initial = val
    infile, outfile = args
    data = read_infile(infile)
    aligner = Aligner(n_iter=1, separate_endings=True, init="lcs",
                      init_params={"gap": 2, "initial_gap": 3})
    aligned_data = aligner.align(data, save_initial=save_initial)
    # output_data(data, aligned_data, outfile)
    test_data = [("НЕНеЦ", "НЕНЦев")]
    aligned_data = aligner.align(test_data, words=["НЕНЦов"], to_fit=False)
    print(aligned_data)
