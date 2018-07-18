import sys
import bisect

import numpy as np

from cutility import _find_indexes, _extract_ordered_sequences

def extract_ordered_sequences(lists, min_differences=None, max_differences=None,
                              strict_min=True, strict_max=False):
    '''
    Аргументы:
    ----------
    lists: list of lists
    список списков L1, ..., Lm
    min_differences: array-like, shape=(m, ) or None(default=None)
    набор [d1, ..., dm] минимальных разностей
    между соседними элементами порождаемых списков
    min_differences: array-like, shape=(m, ) or None(default=None)
    набор [d'1, ..., d'm] минимальных разностей
    между соседними элементами порождаемых списков

    Возвращает:
    -----------
    генератор списков [x1, ..., xm], x1 \in L1, ..., xm \in Lm
    d'1 >= x1 >= d1, x1 + d2' >= x2 > x1 + d2, ..., x1 + d(m-1)' >= xm > x(m-1) + dm
    если min_differences=None, то возвращаются просто
    строго монотонные последовательности
    '''
    # m = len(lists)
    # if m == 0:
    #     return []
    # if min_differences is None:
    #     min_differences = [0.0] * m
    #     min_differences[0] = -np.inf
    # else:
    #     if len(min_differences) != m:
    #         raise ValueError("Lists and min_differences must have equal length")
    # if max_differences is None:
    #     max_differences = [np.inf] * m
    # else:
    #     if len(max_differences) != m:
    #         raise ValueError("Lists and min_differences must have equal length")
    # if any(x < y for x, y in zip(max_differences, min_differences)):
    #     return []
    # lists = [sorted(lst) for lst in lists]
    # list_lengths = [len(x) for x in lists]
    # if any(x == 0 for x in list_lengths):
    #     return []
    # min_indexes = [find_first_larger_indexes([x + d for x in lists[i]], lists[i+1], strict_min)
    #                for i, d in enumerate(min_differences[1:])]
    # max_indexes = [find_first_larger_indexes([x + d for x in lists[i]], lists[i+1], strict_max)
    #                for i, d in enumerate(max_differences[1:])]
    # answer = []
    # # находим минимальную позицию первого элемента
    # startpos = bisect.bisect_left(lists[0], min_differences[0])
    # if startpos == list_lengths[0]:
    #     return []
    # endpos = bisect.bisect_right(lists[0], max_differences[0])
    # sequence, index_sequence = [lists[0][startpos]], [startpos]
    # # TO BE MODIFIED
    # while len(sequence) > 0:
    #     '''
    #     обходим все монотонные последовательности в лексикографическом порядке
    #     '''
    #     i, li = len(index_sequence) - 1, index_sequence[-1]
    #     if i < m - 1 and min_indexes[i][li] < max_indexes[i][li]:
    #         '''
    #         добавляем минимально возможный элемент в последовательность
    #         '''
    #         next_index = min_indexes[i][li]
    #         i += 1
    #         index_sequence.append(next_index)
    #         sequence.append(lists[i][next_index])
    #     else:
    #         if i == m - 1:
    #             '''
    #             если последовательность максимальной длины
    #             то обрезаем последний элемент
    #             чтобы сохранить все возможные варианты для последней позиции
    #             '''
    #             index_sequence.pop()
    #             sequence.pop()
    #             i -= 1
    #             curr_max_index = max_indexes[i][index_sequence[-1]] if i >= 0 else endpos
    #             while li < curr_max_index:
    #                 answer.append(sequence + [lists[-1][li]])
    #                 li += 1
    #             if i < 0:
    #                 break
    #         '''
    #         пытаемся перейти к следующей
    #         в лексикографическом порядке последовательности
    #         '''
    #         index_sequence[-1] += 1
    #         while i > 0 and index_sequence[-1] == max_indexes[i-1][index_sequence[-2]]:
    #             '''
    #             увеличиваем последний индекс
    #             если выходим за границы, то
    #             укорачиваем последовательность и повторяем процедуру
    #             '''
    #             index_sequence.pop()
    #             i -= 1
    #             index_sequence[-1] += 1
    #         if i == 0 and index_sequence[0] == endpos:
    #             break
    #         sequence[i:] = [lists[i][index_sequence[-1]]]
    answer = _extract_ordered_sequences(
        lists, min_differences=min_differences, max_differences=max_differences,
        strict_min=strict_min, strict_max=strict_max)
    return answer

def find_first_larger_indexes(first, second, strict=True):
    '''
    Аргументы:
    ---------
    first, second: array,
    упорядоченные массивы длины m и n типа type
    strict: bool,
    индикатор строгости неравенства

    Возвращает:
    indexes: array, shape=(m,)
    упорядоченный массив длины m, indexes[i] равен минимальному j,
    такому что first[i] < second[j] (в случае strict=True)
    или first[i] <= second[j] (в случае strict=False)
    '''
    return _find_indexes(first, second, strict)

def _find_first_indexes(first, second, pred):
    '''
    Аргументы:
    ---------
    first, second: array,
    упорядоченные массивы длины m и n типа type
    pred: type->(type->bool),
    предикат, монотонный по второму аргументу и антимонотонный по первому

    Возвращает:
    indexes: array, shape=(m,)
    упорядоченный массив длины m, indexes[i] равен минимальному j,
    такому что pred(first[i], second[j])=True или n, если такого i нет
    '''
    m, n = len(first), len(second)
    i, j = 0, 0
    indexes = [-1] * m
    while i < m and j < n:
        if pred(first[i], second[j]):
            indexes[i] = j
            i += 1
        else:
            j += 1
    if i < m:
        indexes[i:m] = [n] * (m - i)
    return indexes

def generate_monotone_sequences(first, length, upper, min_differences=None,
                                max_differences=None):
    if length == 0:
        return []
    lists = [list(range(first, upper)) for _ in range(length)]
    if min_differences is not None:
        if len(min_differences) != length - 1:
            raise ValueError("Min_differences must be of length (length - 1)")
    else:
        min_differences = [0] * length
        min_differences[0] = first
    if max_differences is not None:
        if len(max_differences) != length - 1:
            raise ValueError("Max_differences must be of shape (length - 1)")
    else:
        max_differences = [(upper - first)] * length
        max_differences[0] = first
    return extract_ordered_sequences(lists, min_differences, max_differences)


def find_optimal_cover_elements(lists):
    """
    Аргументы:
    -----------
    lists: iterable
        список списков

    Возвращает:
  ------------
    optimal_covers: list of sets
      список множеств S минимального размера,
        таких что в каждом элементе lists хотя бы один элемент l \subseteq S
    """
    optimal_covers, optimal_cover_size = {tuple()}, 0
    for lst in lists:
        new_optimal_cover_size = None
        new_optimal_covers = set()
        # всё время поддерживаем минимальное покрытие
        # для уже обработанного участка lists
        if len(lst) > 0:
            for elem in lst:
                for cover in optimal_covers:
                    new_cover = set(cover) | set(elem)
                    if (new_optimal_cover_size is None
                            or len(new_cover) < new_optimal_cover_size):
                        new_optimal_covers = {tuple(sorted(new_cover))}
                        new_optimal_cover_size = len(new_cover)
                    elif len(new_cover) == new_optimal_cover_size:
                        new_optimal_covers.add(tuple(sorted(new_cover)))
            optimal_covers = new_optimal_covers
            optimal_cover_size = new_optimal_cover_size
    return optimal_covers, optimal_cover_size


def make_tail_maximal_indexes(lst):
    """
    Аргументы:
    -----------
    lst: list, список объектов, поддерживающих операцию сравнения

    Возвращает:
    -----------
    tail_maximal_indexes: list of ints,
        tail_maximal_indexes[j] = argmax(lst[j:])
    """
    length = len(lst)
    if length == 0:
        return []
    answer, curr_max, curr_argmax = [None] * length, lst[-1], length - 1
    answer[-1] = length - 1
    for i, v in enumerate(lst[::-1][1:], 2):
        if v > curr_max:
            curr_max, curr_argmax = v, length - i
        answer[length - i] = curr_argmax
    return answer[::-1]

if __name__ == "__main__":
    l1 = [1, 4, 8]
    l2 = [3, 7, 8]
    l3 = [2, 4, 7, 9]
    lists = [l1, l2, l3]
    min_differences = [1, 1, 0]
    max_differences = [5, 4, 3]
    for elem in extract_ordered_sequences(lists, min_differences, max_differences):
        print(elem)
    print("")
    for elem in generate_monotone_sequences(3, 3, 8):
        print(elem)
    lists = [[[]], [[3]], [[3]]]
    print(find_optimal_cover_elements(lists))
