from collections import defaultdict
from sortedcontainers import SortedDict
from heapdict import heapdict

from read import read_infile

class PairEncoder:

    def __init__(self, min_count=50, max_number=50):
        self.min_count = min_count
        self.max_number = max_number
        self.pairs_ = []

    @property
    def pairs_number_(self):
        return len(self.pairs_)

    def symbol_repr(self, x):
        return (x, ) if (x[0] != "#") else self.pairs_[int(x[1:])]

    def bigram_repr(self, x):
        return self.symbol_repr(x[0]) + self.symbol_repr(x[1])

    def fit(self, data):
        self._data = [['BOW'] + list(x) + ['EOW'] for x in data]
        self.right_positions = [list(range(1, len(x))) + [None] for x in self._data]
        self.left_positions = [[None] + list(range(len(x) - 1))  for x in self._data]
        self._initialize()
        while len(self.pairs_) < self.max_number:
            bigram, count = self.bigram_counts.peekitem()
            count = -count
            if count < self.min_count:
                break
            self._update(bigram)
            print(self.pairs_number_, self.pairs_[-1], count)
        self.pair_codes_ = {pair: i for i, pair in enumerate(self.pairs_)}
        self.make_trie()
        return self

    def make_trie(self):
        self.trie_nodes = [dict()]
        self.node_depth = [0]
        self.is_node_terminal = [None]
        for i, x in enumerate(self.pairs_):
            curr = 0
            for j, a in enumerate(x):
                child = self.trie_nodes[curr].get(a)
                if child is None:
                    self.trie_nodes.append(dict())
                    self.node_depth.append(j+1)
                    self.is_node_terminal.append(None)
                    child = len(self.trie_nodes) - 1
                    self.trie_nodes[curr][a] = child
                curr = child
            self.is_node_terminal[curr] = i
        return self

    def _initialize(self):
        self.bigram_positions = defaultdict(set)
        for i, s in enumerate(self._data):
            for j in range(len(s) - 1):
                self.bigram_positions[tuple(s[j:j+2])].add((i, j))
        self.bigram_counts = heapdict()
        for bigram, positions in self.bigram_positions.items():
            self.bigram_counts[bigram] = -len(positions)
        return self

    def _decrease_count(self, bigram, i, start):
        self.bigram_counts[bigram] += 1
        self.bigram_positions[bigram].remove((i, start))
        if self.bigram_counts[bigram] == 0:
            self.bigram_counts.pop(bigram)
            self.bigram_positions.pop(bigram)
        return self

    def _increase_count(self, bigram, i, start):
        if bigram not in self.bigram_counts:
            self.bigram_counts[bigram] = -1
        else:
            self.bigram_counts[bigram] -= 1
        self.bigram_positions[bigram].add((i, start))
        return self

    def _update(self, bigram):
        new_symbol = "#{}".format(self.pairs_number_)
        for i, start in sorted(self.bigram_positions[bigram]):
            s = self._data[i]
            end = self.right_positions[i][start]
            right = self.right_positions[i][end]
            left = self.left_positions[i][start]
            if left is not None:
                self._decrease_count((s[left], s[start]), i, left)
                self._increase_count((s[left], new_symbol), i, left)
            if right is not None:
                self._decrease_count((s[end], s[right]), i, end)
                self._increase_count((new_symbol, s[right]), i, start)
                self.left_positions[i][right] = start
            self.right_positions[i][start] = right
            s[start], s[end] = new_symbol, None
        self.bigram_counts.pop(bigram)
        self.bigram_positions.pop(bigram)
        self.pairs_.append(self.bigram_repr(bigram))
        return self

    def transform(self, data):
        return [self.transform_string(x) for x in data]

    def transform_string(self, x):
        x = ['BOW'] + list(x) + ['EOW']
        answer = []
        curr, pos, root_pos = 0, 0, 0
        while pos < len(x):
            child = self.trie_nodes[curr].get(x[pos])
            if child is None:
                if self.is_node_terminal[curr] is not None:
                    answer.append('#{}'.format(self.is_node_terminal[curr]))
                    root_pos = pos
                else:
                    answer.append(x[root_pos])
                    root_pos += 1
                curr, pos = 0, root_pos
            else:
                curr, pos = child, pos + 1
        if curr != 0:
            answer.extend(x[root_pos:])
        return tuple(answer)

if __name__ == "__main__":
    data = read_infile("conll2018/task1/all/belarusian-train-medium")
    data = [x for elem in data for x in elem[:2]]
    pair_encoder = PairEncoder(min_count=50, max_number=50)
    pair_encoder.fit(data[:])
    for elem in data[:20]:
        print(elem, " ".join("_".join(pair_encoder.symbol_repr(x))
                             for x in pair_encoder.transform_string(elem)))