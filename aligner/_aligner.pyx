# cython: language_level=3, profile=True, linetrace=True

from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp.pair cimport pair
from libcpp cimport bool

import numpy as np
cimport numpy as np
cimport cython

import sys

from time import time
from collections import defaultdict

MAX_DIFF = 50
INF_COST = 1000

ctypedef int (*f_type)(double, double, double)

cdef extern from "aligner.cpp" :
    double log_add(double x, double y);
    double make_cost(double count, int total_count, int m, double prior);
    void fill_trellis(double* change_costs, double* removal_costs,
                      double* insertion_costs, double* trellis, int m, int n)
    void fill_trellis_1(double* trellis, int* first, int* second,
                        double* counts, int total_count, int total_number, 
                        double prior, int symbols_number, int m, int n)
    void fill_trellis_2(double* trellis_root, double* trellis_ending,
                        double* trellis_equal, int* first, int* second, 
                        double* counts, int total_count, int total_number, 
                        double prior, int symbols_number, int m, int n, int to_print)
    int draw_sample(double, double, double, int)
    int draw_sample2(double, double, int)

cdef int draw_min(double x, double y, double z):
    return (-1 if x <= z else 1) if x <= y else (0 if y < z else 1)

cdef class CythonAlignerImpl:

    cdef int n_iter, pairs_number, total_count, max_code, codes_number
    cdef int random_state, verbose
    cdef int EMPTY
    cdef double prior
    cdef bool separate_endings
    cdef double[:] counts
    
    def __init__(self, int n_iter=10, double prior=0.1, bool separate_endings=False,
                 int random_state=187, int verbose=1):
        self.n_iter = n_iter
        self.prior = prior
        self.separate_endings = separate_endings
        self.random_state = random_state
        self.verbose = verbose
        self.pairs_number = 0
        self.total_count = 0
        self.max_code = 0
        self.EMPTY = 0
        
    cdef int pair_code(self, int x, int y):
        x = min(x, self.max_code)
        y = min(y, self.codes_number - 1)
        return x * self.codes_number + y
    
    cdef double _get_cost(self, int x, int y):
        cdef double count = 1.0, prior = self.prior
        cdef double mult = 1.0
        cdef int pair_code
        
        if x < self.max_code and y < self.codes_number - 1:
            pair_code = self.pair_code(x, y)
        else:
            mult = self.prior * (self.max_code + self.codes_number) / self.total_count
            prior = 0.0
            if x < self.max_code or y < self.codes_number - 1 or x == y:
                pair_code = self.pair_code(x, y)
            else:
                pair_code = -1
        if pair_code >= 0:
            count = self.counts[pair_code] * mult + prior
            
        return make_cost(count, self.total_count, self.pairs_number, self.prior)

    cdef void _decrease_counts(self, list pairs):
        """
        Уменьшает счётчики пар из pairs
        """
        cdef int x, y, pair_code
        
        for pair in pairs:
            pair_code = self.pair_code(pair[0], pair[1])
            self.counts[pair_code] -= 1
            self._update_unknown_count(pair[0], pair[1], -1)
        self.total_count -= len(pairs)
        return
        
    def _increase_counts(self, list pairs):
        """
        Увеличивает счётчики пар из pairs
        """
        cdef int pair_code
        
        for pair in pairs:
            pair_code = self.pair_code(pair[0], pair[1])
            self.counts[pair_code] += 1
            self._update_unknown_count(pair[0], pair[1], 1)
        self.total_count += len(pairs)
        return
        
    cdef void _update_unknown_count(self, int x, int y , double value):
        """
        Увеличивает соответствующий счётчик 
        для неизвестного символа (self.max_code)
        на значение value
        """
        cdef int pair_code, y_
        # ключ, что пара состоит из разных символов x, y,
        # ни один из которых не является пустым
        cdef bool are_symbols_different = False
        
        if x == y:
            # увеличиваем счётчик с(UNK, UNK)
            x, y = self.max_code, self.codes_number-1
        elif x == self.EMPTY:
            y = self.codes_number-1  # (EMPTY, UNK)
        elif y == self.EMPTY or y == self.EMPTY + self.max_code:
            x = self.max_code  # (UNK, EMPTY)
        else:
            are_symbols_different = True
            y_, y = y, self.codes_number-1  # (x, UNK)
            value /= 2
        pair_code = self.pair_code(x, y)
        self.counts[pair_code] += value
        if are_symbols_different:
            pair_code = self.pair_code(self.max_code, y_)  # (UNK, y)
            self.counts[pair_code] += value
        return
        
        
    # @cython.profile
    cdef _make_trellis(self, np.ndarray[int,ndim=1] x, np.ndarray[int,ndim=1] y):
        cdef int m, n, i, j, a, b
        m, n = len(x), len(y)
        cdef np.ndarray[double, ndim=2] trellis = np.zeros([m+1, n+1], dtype=np.float)
        cdef np.ndarray[double, ndim=1] counts = np.zeros(
            dtype=np.float, shape=((self.max_code+1)*self.codes_number,))
        counts[:] = self.counts
        fill_trellis_1(<double*>trellis.data, <int*>x.data, <int*>y.data,
                       <double*>counts.data, self.total_count, self.pairs_number,
                       self.prior, self.max_code, m, n)
        return trellis
        
    # @cython.profile
    cdef _make_trellis_with_endings(self, np.ndarray[int,ndim=1] x,
                                    np.ndarray[int,ndim=1] y,
                                    bool to_print=False, object fout=None):
        cdef int i, j, a, b, m, n, pair_code
        cdef double prior = self.prior
        m, n = len(x), len(y)
        cdef np.ndarray[double, ndim=2] root_trellis =\
            np.zeros([m+1, n+1], dtype=np.float)
        cdef np.ndarray[double, ndim=2] ending_trellis = np.zeros([m+1, n+1], dtype=np.float)
        cdef np.ndarray[double, ndim=2] equal_trellis = np.zeros([m+1, n+1], dtype=np.float)
        cdef np.ndarray[double, ndim=1] counts =\
            np.zeros(dtype=np.float, shape=((self.max_code+1)*self.codes_number,))
        counts[:] = self.counts
        
        fill_trellis_2(<double*>root_trellis.data, <double*>ending_trellis.data,
                       <double*>equal_trellis.data, <int*>x.data, <int*>y.data,
                       <double*>counts.data, self.total_count, self.pairs_number,
                       prior, self.max_code, m, n, 0)
        if to_print:
            fout.write("\n{} {}\n".format(",".join([str(a) for a in list(x)]), 
                                          ",".join([str(b) for b in list(y)])))
            for a in list(x) + [self.EMPTY]:
                for b in list(y) + [self.EMPTY]:
                    pair_code = self.pair_code(a, b)
                    fout.write("{:.0f},{:.1f},{:.1f} ".format(
                        self.counts[pair_code], self._get_cost(a, b), 
                        self._get_cost(a, b+self.max_code)))
                fout.write("\n")
            for i in range(m+1):
                for j in range(n+1):
                    fout.write("{:.1f},{:.1f},{:.1f} ".format(
                        root_trellis[i,j], ending_trellis[i,j], equal_trellis[i,j]))
                fout.write("\n")
        return root_trellis, ending_trellis, equal_trellis

    cdef list sample_path(self, np.ndarray[int,ndim=1] x,
                          np.ndarray[int,ndim=1] y, bool best=False,
                          bool to_print=False, object fout=None):
        cdef int n = len(y)
        cdef list path
        cdef np.ndarray[double,ndim=2] trellis
        cdef np.ndarray[double,ndim=2] equal_trellis
        cdef np.ndarray[double,ndim=2] ending_trellis
        cdef np.ndarray[int, ndim=1] y_low = np.zeros(dtype=np.intc, shape=(n,))
        # в функцию вычисления решётки нужно передавать индексы
        # без различия корня и окончания
        for i in range(n):
            y_low[i] = y[i] if (y[i] < self.max_code or 
                                y[i] >= self.codes_number-1) else y[i] - self.max_code
        
        if self.separate_endings:
            trellis, ending_trellis, equal_trellis =\
                self._make_trellis_with_endings(x, y_low, to_print=to_print, fout=fout)
            path = self._sample_path_with_endings(
                trellis, ending_trellis, equal_trellis, x, y_low, 
                best, to_print=to_print, fout=fout)
            if to_print:
                fout.write(" ".join("{}-{}".format(*elem) for elem in path) + "\n")
        else:
            trellis = self._make_trellis(x, y)
            path = self._sample_path_simple(trellis, x, y, best)
        return path
        
    cdef list _sample_path_simple(
            self, np.ndarray[double,ndim=2] trellis, np.ndarray[int,ndim=1] x,
            np.ndarray[int,ndim=1] y, bool best=False):
        cdef int m, n, i, j, move_key
        cdef list path
        cdef tuple pair
        cdef f_type func
        
        # trellis = self._make_trellis(first, second)
        i, j = len(x), len(y)
        path = []
        while i + j > 0:
            if i == 0:
                j -= 1
                path.append((self.EMPTY, y[j]))
            elif j == 0:
                i -= 1
                path.append((x[i], self.EMPTY))
            else:
                # func = draw_min if best else draw_sample
                if best:
                    move_key = draw_min(trellis[i-1, j] + self._get_cost(x[i-1], self.EMPTY),
                                        trellis[i-1, j-1] + self._get_cost(x[i-1], y[j-1]),
                                        trellis[i, j-1] + self._get_cost(self.EMPTY, y[j-1]))
                else:
                    move_key = draw_sample(trellis[i-1, j] + self._get_cost(x[i-1], self.EMPTY),
                                           trellis[i-1, j-1] + self._get_cost(x[i-1], y[j-1]),
                                           trellis[i, j-1] + self._get_cost(self.EMPTY, y[j-1]),
                                           self.random_state)
                pair = ((x[i-1], self.EMPTY) if move_key == -1 else
                        (x[i-1], y[j-1]) if move_key == 0 else (self.EMPTY, y[j-1]))
                path.append(pair)
                i = i-1 if move_key <= 0 else i
                j = j-1 if move_key >= 0 else j
        return path[::-1]
        
    cdef list _sample_path_with_endings(self, np.ndarray[double,ndim=2] root_trellis, 
                                       np.ndarray[double,ndim=2] ending_trellis,
                                       np.ndarray[double,ndim=2] equal_trellis,
                                       np.ndarray[int,ndim=1] x,  np.ndarray[int,ndim=1] y,
                                       bool best=False, bool to_print=False, object fout=None):
        cdef int m, n, i, j, s, i_new, j_new
        cdef int upper, lower, lower_empty
        cdef bool is_root
        cdef double insertion_cost, change_cost, removal_cost;
        cdef double total_up_cost, total_diag_cost, total_left_cost;
        cdef np.ndarray[double,ndim=2] curr_trellis
        
        m, n = root_trellis.shape[0], root_trellis.shape[1]
        i, j = m - 1, n - 1
        path = []
        if best:
            is_root = (equal_trellis[i,j] < ending_trellis[i,j])
        else:
            is_root = draw_sample2(ending_trellis[i][j], equal_trellis[i][j], self.random_state)
        curr_trellis = root_trellis if is_root else ending_trellis
        
        while i + j > 0:
            if to_print:
                fout.write("{} {} {}\n".format(is_root, i, j))
            # s = i * (n+1) + j
            upper = x[i-1]
            lower = (y[j-1] if (is_root or y[j-1] >= self.codes_number-1) 
                     else y[j-1] + self.max_code)
            if i == 0:
                upper = self.EMPTY
                if not is_root:
                    if best:
                        is_root = (equal_trellis[i,j-1] < ending_trellis[i,j-1])
                    else:
                        is_root = draw_sample2(
                             ending_trellis[i,j-1], equal_trellis[i,j-1], self.random_state)
                j -= 1
            elif j == 0:
                lower = self.EMPTY
                if not is_root:
                    if best:
                        is_root = (equal_trellis[i-1,j] < ending_trellis[i-1,j])
                    else:
                        is_root = draw_sample2(
                            ending_trellis[i-1,j], equal_trellis[i-1,j], self.random_state)
                i -= 1
            else:
                # func = draw_min if best else draw_sample
                # заводим переменные
                lower_empty = self.EMPTY if is_root else self.EMPTY + self.max_code
                insertion_cost = self._get_cost(self.EMPTY, lower)
                removal_cost = self._get_cost(x[i-1], lower_empty)
                change_cost = self._get_cost(x[i-1], lower)
                if is_root:
                    if best:
                        move_key = draw_min(root_trellis[i, j-1] + insertion_cost,
                                            root_trellis[i-1, j-1] + change_cost,
                                            root_trellis[i-1, j] + removal_cost)
                    else:
                        move_key = draw_sample(root_trellis[i, j-1] + insertion_cost,
                                               root_trellis[i-1, j-1] + change_cost,
                                               root_trellis[i-1, j] + removal_cost,                                               
                                               self.random_state)
                    if to_print:
                        fout.write("{},{:.1f},{:.1f},{:.1f}\n".format(
                            move_key, root_trellis[i, j-1] + insertion_cost,
                            root_trellis[i-1, j-1] + change_cost,
                            root_trellis[i-1, j] + removal_cost))
                    
                else:
                    if not best:
                        total_up_cost = log_add(equal_trellis[i-1, j], 
                                                ending_trellis[i-1, j]) + removal_cost
                        if x[i-1] == y[j-1]:
                            total_diag_cost = INF_COST
                        else:
                            total_diag_cost = log_add(equal_trellis[i-1, j-1],  
                                                      ending_trellis[i-1, j-1]) + change_cost
                        total_left_cost = log_add(equal_trellis[i,j-1], 
                                                  ending_trellis[i,j-1]) + insertion_cost
                        move_key = draw_sample(total_left_cost, total_diag_cost, 
                                               total_up_cost, self.random_state)
                    else:
                        total_up_cost = min(equal_trellis[i-1, j], 
                                            ending_trellis[i-1, j]) + removal_cost
                        if x[i-1] == y[j-1]:
                            total_diag_cost = INF_COST
                        else:
                            total_diag_cost = min(equal_trellis[i-1,j-1], 
                                                  ending_trellis[i-1, j-1]) + change_cost
                        total_left_cost = min(equal_trellis[i,j-1], 
                                              ending_trellis[i,j-1]) + insertion_cost
                        move_key = draw_min(
                            total_left_cost, total_diag_cost, total_up_cost)
                    if to_print:
                        fout.write("{} {:.1f},{:.1f},{:.1f}\n".format(
                            move_key, total_left_cost, total_diag_cost, total_up_cost))
                    i_new = i-1 if move_key >= 0 else i
                    j_new = j-1 if move_key <= 0 else j
                    if to_print:
                        fout.write("{:.2f} {:.2f}\n".format(
                            equal_trellis[i_new, j_new], ending_trellis[i_new, j_new]))
                    if best:
                        is_root = draw_sample2(ending_trellis[i_new, j_new],
                                               equal_trellis[i_new, j_new],
                                               self.random_state)
                    else:
                        is_root = (equal_trellis[i_new, j_new] < ending_trellis[i_new, j_new])
                if move_key < 0:
                    upper = self.EMPTY
                elif move_key == 1:
                    lower = lower_empty
                i = i-1 if move_key >= 0 else i
                j = j-1 if move_key <= 0 else j
            path.append((upper, lower))
        return path[::-1]

    # @cython.profile
    def fit(self, list X, list pairs, int max_code, set indexes_to_print=None, int n_iter=-1):
        cdef int i, j, index, pair_code, x, y, k, l
        cdef double t1, t2
        cdef np.ndarray[int,ndim=1] first, second
        cdef np.ndarray[double,ndim=2] trellis
        cdef tuple pair
        cdef list elem, indexes, curr_pairs
        cdef object fout = None
        
        if n_iter == -1:
            # число итераций не задано
            n_iter = self.n_iter
        if indexes_to_print is None:
            indexes_to_print = set()
        # for pair in X:
        #     self.max_code = max(max(pair[0]), self.max_code)
        #     self.max_code = max(max(pair[1]), self.max_code)
        self.max_code = max_code
        # add 1 for unknown symbol
        self.codes_number = self.max_code * (int(self.separate_endings) + 1) + 1
        self.counts = np.zeros(dtype=np.float, shape=((self.max_code+1)*self.codes_number,))
        for i, elem in enumerate(pairs):
            for pair in elem:
                pair_code = self.pair_code(pair[0], pair[1])
                self.counts[pair_code] += 1
        self.pairs_number = (self.max_code + 1) * self.codes_number
        self.total_count = sum([len(elem) for elem in pairs])
        # создаём счётчики для неизвестного символа
        for i in range(self.max_code):
            for j in range(self.codes_number-1):
                pair_code = self.pair_code(i, j)
                if self.counts[pair_code] > 0:
                    self._update_unknown_count(i, j, self.counts[pair_code])
        np.random.seed(self.random_state)
        for i in range(n_iter):
            if len(indexes_to_print) > 0 and self.verbose > 0:
                fout = open("log_{}.out".format(i), "w")
                for k in range(self.max_code+1):
                    fout.write("{}\t".format(k))
                    for r in range(self.codes_number):
                        pair_code = self.pair_code(k, r)
                        if self.counts[pair_code] > 0:
                            fout.write("{}-{:.0f} ".format(r, self.counts[pair_code]))
                    fout.write("\n")
                fout.write("{}\n".format(self.total_count))
            t1 = time()
            indexes = list(range(len(X)))
            # np.random.shuffle(indexes)
            for j, index in enumerate(indexes):
                first, second = X[index][0], X[index][1]
                if j % 5000 == 0 and self.verbose > 0:
                    print("Iteration {}, {} objects passed".format(i+1, j))
                curr_pairs = self.sample_path(
                    first, second, best=False, to_print=(index in indexes_to_print), fout=fout)
                self._decrease_counts(pairs[index])
                self._increase_counts(curr_pairs)
                pairs[index] = curr_pairs
            t2 = time()
            if self.verbose > 0:
                print("Iteration {}, {:.2f} seconds elapsed".format(i+1, t2 - t1))
            if len(indexes_to_print) > 0 and self.verbose > 0:
                fout.close()
        return self

    cpdef predict(self, list X, set indexes_to_print=None):
        cdef int i
        cdef list answer
        cdef np.ndarray[int,ndim=1] first, second
        cdef object fout=None
        
        answer = [None] * len(X)
        if len(indexes_to_print) > 0 and self.verbose > 0:
            fout = open("log_predict.out", "w")
            for k in range(self.max_code+1):
                fout.write("{}\t".format(k))
                for r in range(self.codes_number):
                    pair_code = self.pair_code(k, r)
                    if self.counts[pair_code] > 0:
                        fout.write("{}-{:.0f} ".format(r, self.counts[pair_code]))
                fout.write("\n")
            fout.write("{}\n".format(self.total_count))
        # for i, (first, second) in enumerate(X):
        for i in range(len(X)):
            first, second = X[i][0], X[i][1]
            answer[i] = self.sample_path(
                first, second, best=True, to_print=(i in indexes_to_print), fout=fout)
        if len(indexes_to_print) > 0 and self.verbose > 0:
            fout.close()
        return answer
    