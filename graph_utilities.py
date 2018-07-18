import sys
from collections import defaultdict, OrderedDict
import copy

class TopologicalSorter:
    """
    Класс для выполнения топологической сортировки вершин графа
    """

    def __init__(self, graph):
        self.vertexes = graph.vertexes
        self.transitions = graph.transitions

    def component_topological_sort(self, source):
        """
        Топологическая сортировка связной компоненты графа self.graph,
        начиная с вершины source

        Возвращает:
        -----------
        order: list or None
            список вершин в порядке топологической сортировки
            или None, если такая сортировка невозможна
        """
        if source not in self.vertexes:
            return None
        color = {x: 'white' for x in self.vertexes}
        stack, process_flags_stack = [source], [False]
        order = []
        while len(stack) > 0:
            current, current_flag = stack[-1], process_flags_stack[-1]
            if current_flag:  # выход из рекурсии для вершины
                stack.pop()
                process_flags_stack.pop()
                color[current] = 'black'
                order = [current] + order
            else:
                if color[current] == 'grey':
                    # граф имеет циклы, сортировка невозможна
                    return None
                elif color[current] == 'white':
                    color[current] = 'grey'
                    process_flags_stack[-1] = True
                    for other in self.transitions[current]:
                        stack.append(other)
                        process_flags_stack.append(False)
                elif color[current] == 'black':
                    stack.pop()
                    process_flags_stack.pop()
        return order


class Graph:
    """
    Представление графа в виде списка вершин
    """

    def __init__(self, transitions=None):
        """
        Атрибуты:
        ---------
        transitions: list or None(default=None) --- список рёбер графа
        """
        if transitions is None:
            transitions = []
        try:
            transitions = list(transitions)
        except:
            raise TypeError("Transitions must be a list or None")
        self.vertexes = set()
        self.transitions = dict()
        for first, second in transitions:
            if first in self.vertexes:
                self.transitions[first].add(second)
            else:
                self.vertexes.add(first)
                self.transitions[first] = {second}
            if second not in self.vertexes:
                self.vertexes.add(second)
                self.transitions[second] = set()

    def find_longest_paths(self, source):
        """
        Находит самые длинные пути в ациклическом графе,
        начинающиеся в вершине source

        Аргументы:
        ----------
        source, object --- стартовая вершина

        Возвращает:
        -----------
        answer, list of lists
            список путей, представленных как список состояний
        """
        if not hasattr(self, "topological_order_"):
            self.topological_order_ = self.get_topological_sort(source)
        if self.topological_order_ is None:
            return []
        longest_path_lengths = {v: -1 for v in self.vertexes}
        longest_path_lengths[source] = 0
        max_length, furthest_vertexes = 0, {0}
        predecessors_in_longest_paths = {v: set() for v in self.vertexes}
        for first in self.topological_order_:
            length_to_source = longest_path_lengths[first]
            for second in self.transitions[first]:
                length_to_second = length_to_source + 1
                # обновляем максимальное расстояние
                if length_to_second > longest_path_lengths[second]:
                    longest_path_lengths[second] = length_to_second
                    predecessors_in_longest_paths[second] = {first}
                    # обновляем текущий список наиболее удалённых вершин
                    if length_to_second > max_length:
                        max_length = length_to_second
                        furthest_vertexes = {second}
                    elif length_to_second == max_length:
                        furthest_vertexes.add(second)
                elif length_to_second == longest_path_lengths[second]:
                    predecessors_in_longest_paths[second].add(first)
        answer = _backtrace_longest_paths(furthest_vertexes, source, max_length,
                                          predecessors_in_longest_paths)
        return answer

    def get_topological_sort(self, source):
        """
        Находит топологическую сортировку вершин, начиная с заданной
        """
        topological_sorter = TopologicalSorter(self)
        return topological_sorter.component_topological_sort(source)

    def find_maximal_paths(self, source):
        """
        Находит такие пути в ациклическом графе,
        начинающиеся в вершине source, что их нельзя продолжить дальше

        Аргументы:
        ----------
        source, object --- стартовая вершина

        Возвращает:
        -----------
        answer, list of lists
            список путей, представленных как список состояний,
            в порядке убывания их длины
        """
        if not hasattr(self, "_reverse_transitions_"):
            self._make_reverse_transitions()
        stock_vertexes =\
            [v for v in self.vertexes if (v not in self.transitions
                                          or len(self.transitions[v]) == 0)]
        paths = [[v] for v in stock_vertexes]
        answer = []
        for path in paths:
            v = path[-1]
            if v == source:
                answer.append(path[::-1])
                continue
            for other in self._reverse_transitions_[v]:
                paths.append(path + [other])
        return sorted(answer, key=len, reverse=True)

    def _make_reverse_transitions(self):
        """
        Строит граф с обратными рёбрами
        """
        self._reverse_transitions_ = dict()
        for v in self.vertexes:
            self._reverse_transitions_[v] = set()
        for first, first_transitions in self.transitions.items():
            for second in first_transitions:
                self._reverse_transitions_[second].add(first)
        return


def _backtrace_longest_paths(finals, source, length, predecessors):
    """
    Восстанавливает самые длинные пути с помощью обратных ссылок
    """
    longest_paths_stack = []
    answer = []
    for vertex in finals:
        longest_paths_stack.append((vertex, length, [vertex]))
    for curr_vertex, curr_length, curr_path in longest_paths_stack:
        if curr_length == 0:
            if curr_vertex == source:
                answer.append(curr_path[::-1])
        else:
            for next_vertex in predecessors[curr_vertex]:
                longest_paths_stack.append((next_vertex, curr_length - 1,
                                            curr_path + [next_vertex]))
    return answer


class TrieNode:
    """
    Класс для представления узла дерева
    """
    NOT_A_NODE = None

    def __init__(self, index, parent=None, data=None, is_terminal=False):
        self.index = index
        self.data = data
        self.is_terminal = False
        self.children = dict()
        self.parent = parent

    def set_child(self, c, child):
        self.children[c] = child

    def set_parent(self, parent):
        self.parent = parent

    def child(self, c):
        return self.children.get(c, TrieNode.NOT_A_NODE)

    def has_child(self, c):
        return (c in self.children)


class Trie:

    def __init__(self, default=None):
        self.default = default
        self.root = TrieNode(0, data=default)
        self.nodes = [self.root]
        self.nodes_number = 1
        self.size = 0

    def __len__(self):
        return self.size

    def __contains__(self, item):
        curr = self.root
        for a in item:
            curr = curr.child(a)
            if curr == TrieNode.NOT_A_NODE:
                return False
        return curr.is_terminal

    def __setitem__(self, key, value):
        curr = self.root
        for i, a in enumerate(key):
            child = curr.child(a)
            if child == TrieNode.NOT_A_NODE:
                curr = self._add_descendant(curr, key[i:])
                break
            else:
                curr = child
        if not curr.is_terminal:
            self.size += 1
        curr.is_terminal = True
        curr.data = value

    def __getitem__(self, item):
        node = self.get_node(item)
        if node is None:
            raise KeyError("{0} is not in trie".format(key))
        return node.data

    def get(self, key, default=None):
        node = self.get_node(key)
        return node.data if node else default

    def _add_descendant(self, curr, key):
        for a in key:
            node = TrieNode(index=self.nodes_number, parent=curr, data=self.default)
            curr.set_child(a, node)
            curr = node
            self.nodes.append(node)
            self.nodes_number += 1
        return curr

    def get_node(self, key):
        curr = self.root
        for a in key:
            curr = curr.child(a)
            if curr == TrieNode.NOT_A_NODE:
                return None
        if curr.is_terminal:
            return curr
        else:
            return None

    def partial_path(self, key):
        """
        Возвращает максиммальный путь от корня, помеченный префиксом key
        """
        answer = [self.root]
        curr = self.root
        for a in key:
            curr = curr.child(a)
            if curr == TrieNode.NOT_A_NODE:
                break
            answer.append(curr)
        return answer

    def path(self, key):
        """
        Возвращает все вершины на ветке, помеченной key
        """
        answer = self.partial_path(key)
        if len(answer) == len(key) + 1:
            return answer
        else:
            return None

    def max_prefix(self, key):
        """
        Возвращает максимальный префикс key, принадлежащий данному дереву
        """
        answer = self.partial_path(key)
        return key[:(len(answer) - 1)]

    def traverse(self, only_terminals=False, return_keys=False):
        """
        Обход дерева в глубину
        """
        stack, key, color = [[self.root, "", False]], "", False
        answer = []
        while len(stack) > 0:
            prev_color = color
            node, letter, color = stack[-1]
            key = key[:-1] if prev_color else key
            if color:
                if node.is_terminal >= only_terminals:
                    to_append = (node, key) if return_keys else node
                    answer.append(to_append)
                stack.pop()
            else:
                key += letter
                stack[-1][2] = True
                for c, child in node.children.items():
                    stack.append([child, c, False])
        return answer

    def iterate(self, return_keys=True, return_values=True):
        """
        аналог метода items
        """
        if return_keys:
            if return_values:
                extractor = (lambda node, key: (key, node.data))
            else:
                extractor = (lambda node, key: key)
        else:
            if return_values:
                extractor = (lambda node, key: node.data)
            else:
                raise NotImplementedError("Either return_keys or return_values should be given")
        return [extractor(*elem) for elem in self.traverse(only_terminals=True, return_keys=True)]

    def print_all(self):
        offsets = [0] * len(self.nodes)
        traversal = self.traverse(only_terminals=False, return_keys=False)[::-1]
        for node in traversal:
            offset = offsets[node.index]
            for child in node.children.values():
                offsets[child.index] = offset + 1
            print(" " * offsets[node.index], end="")
            print("{0} {1} {2}".format(
                node.index, " ".join("{0} {1}".format(first, second.index)
                                     for first, second in node.children.items()),
                node.data))
        print("")

    def __str__(self):
        return ("{{{0}}}".format(", ".join(
            ("{0}: {1}".format(key.__repr__(), node.data.__repr__())
             for node, key in self.traverse(only_terminals=True, return_keys=True)))))

def prune_dead_branches(trie):
    traversal = trie.traverse()
    has_terminal_descendants = [False] * trie.nodes_number
    for node in traversal:
        if (node.is_terminal or
                any(has_terminal_descendants[child.index] for child in node.children.values())):
            has_terminal_descendants[node.index] = True
    pruned_trie = Trie()
    if has_terminal_descendants[trie.root.index]:
        new_node_indexes, new_nodes_number = [-1] * trie.nodes_number, 0
        for i, (node, flag) in enumerate(zip(trie.nodes, has_terminal_descendants)):
            if flag:
                new_node_indexes[i] = new_nodes_number
                new_nodes_number += 1
        new_nodes = [TrieNode(i) for i in range(new_nodes_number)]
        for node, new_index in zip(trie.nodes, new_node_indexes):
            if new_index >= 0:
                new_node = new_nodes[new_index]
                new_node.is_terminal = node.is_terminal
                new_node.data = node.data
                if node.parent is not None:
                    new_parent_index = new_node_indexes[node.parent.index]
                    new_node.set_parent(new_nodes[new_parent_index])
                for c, child in node.children.items():
                    new_child_index = new_node_indexes[child.index]
                    if new_child_index >= 0:
                        new_node.set_child(c, new_nodes[new_child_index])
        pruned_trie.nodes = new_nodes
        pruned_trie.root = new_nodes[0]
        pruned_trie.nodes_number = new_nodes_number
        pruned_trie.size = len(trie)
    return pruned_trie


def make_affix_trie(words, affix_type='suffix', max_length=-1):
    if affix_type not in ['prefix', 'suffix']:
        raise ValueError("Affix type should be 'prefix' or 'suffix'.")
    if max_length == -1:
        affix_extractor = (lambda x:x) if affix_type=='prefix' else (lambda x: x[::-1])
    elif affix_type == 'suffix':
        affix_extractor = lambda x: x[-1:-max_length-1:-1]
    elif affix_type == 'prefix':
        affix_extractor = lambda x: x[:max_length]
    affix_trie = Trie(default=0)
    for word in words:
        affix = affix_extractor(word)
        node = affix_trie.get_node(affix)
        if node is not None:
            node.data += 1
        else:
            affix_trie[affix] = 1
    traversal = affix_trie.traverse(only_terminals=False, return_keys=False)
    for node in traversal:
        for c, child in node.children.items():
            node.data += child.data
    return affix_trie


def prune_by_counts(affix_trie, threshold, min_count=-1, simple=False):
    """
    Удаляет из бора все вершины, в дереве ниже которых
    находится меньше threshold элементов. Если 0 <= threshold < 1,
    то threshold = N * threshold, где N --- число терминальных вершин в дереве

    делает терминальными все листья, а также все вершины, для которых
    число элементов в их неотмеченных поддеревьях превышает threshold.
    """
    if 0.0 <= threshold and threshold < 1.0:
        counts_sum = sum(node.data for node in affix_trie.nodes if node.is_terminal)
        threshold = int(threshold * counts_sum)
    threshold = max(threshold, min_count)
    new_trie = copy.copy(affix_trie)
    has_terminal_descendants = [False] * len(new_trie.nodes)
    for node, key in new_trie.traverse(return_keys=True):
        if not simple:
            sum_of_unlabeled_children_data = 0
            for child in node.children.values():
                if has_terminal_descendants[child.index]:
                    has_terminal_descendants[node.index] = True
                else:
                    sum_of_unlabeled_children_data += child.data
            node.is_terminal = (sum_of_unlabeled_children_data >= threshold)
            has_terminal_descendants[node.index] |= node.is_terminal
        else:
            node.is_terminal = (node.data >= threshold)
    new_trie = prune_dead_branches(new_trie)
    return new_trie


def calculate_affixes_to_remove(data, affix_type, max_length, threshold, min_count):
    data_by_tasks = defaultdict(list)
    if len(data) == 0:
        return []
    if (isinstance(data[0], tuple) or isinstance(data[0], list)) and len(data[0]) == 2:
        return_mode = 'dict'
    elif isinstance(data[0], str):
        return_mode = 'list'
    else:
        print(data[0])
        raise TypeError("Data must be a list of the form "
                        "[(word, task_key), ...] or [word, ...]")
    if return_mode == 'list':
        data = [(word, None) for word in data]
    for word, key in data:
        data_by_tasks[key].append(word)
    answer = OrderedDict()
    affix_modifier = (lambda x: x[::-1]) if affix_type=='suffix' else (lambda x:x)
    for key, words in data_by_tasks.items():
        affix_trie = make_affix_trie(words, affix_type=affix_type, max_length=max_length)
        curr_threshold = max(int(len(words) * threshold), min_count)
        affix_trie = prune_by_counts(affix_trie, curr_threshold)
        traversal = affix_trie.traverse(return_keys=True, only_terminals=True)
        affixes = [affix_modifier(word) for _, word in traversal if word != ""]
        answer[key] = affixes
    return answer if return_mode == 'dict' else answer[None]


def test_graph():
    """
    Тесты
    """
    transitions = [(0, 1), (0, 3), (1, 2), (3, 2), (2, 4), (3, 4)]
    graph = Graph(transitions)
    print(graph.find_longest_paths(0))

def test_trie():
    trie = Trie()
    for s in ['abc', 'cac', 'ac', 'bc', 'c', 'cab']:
        trie[s] = s
    print(trie)
    trie.get_node('cac').is_terminal = False
    trie.get_node('bc').is_terminal = False
    trie['ba'] = 'ba'
    trie = prune_dead_branches(trie)
    print(trie)


def test_affix_trie():
    words = ['abc', 'ba', 'bc', 'abcc', 'ba', 'b', 'cbaca']
    affix_trie = make_affix_trie(words, affix_type='prefix', max_length=4)
    new_trie = prune_by_counts(affix_trie, threshold=2)
    # affix_trie.print_all()
    print(affix_trie.max_prefix('bccac'))
    # new_trie.print_all()
    print(new_trie.max_prefix('bccac'))

if __name__ == "__main__":
    test_affix_trie()
