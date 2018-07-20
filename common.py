import bisect
from itertools import chain

import numpy as np


def to_one_hot(indices, num_classes):
    """
    :param indices: np.array, dtype=int
    :param num_classes: int, число классов
    :return: answer, np.array, shape=indices.shape+(num_classes,)
    """
    shape = indices.shape
    indices = np.ravel(indices)
    answer = np.zeros(shape=(indices.shape[0], num_classes), dtype=int)
    answer[np.arange(indices.shape[0]), indices] = 1
    return answer.reshape(shape+(num_classes,))


def make_bucket_lengths(lengths, buckets_number=None, max_bucket_length=None):
    m = len(lengths)
    lengths = sorted(lengths)
    last_bucket_length, bucket_lengths = 0, []
    if buckets_number is None:
        if max_bucket_length is not None:
            buckets_number = (m - 1) // max_bucket_length + 1
        else:
            raise ValueError("Either buckets_number or max_bucket_length must be given.")
    for i in range(buckets_number):
        # могут быть проблемы с выбросами большой длины
        level = (m * (i + 1) // buckets_number) - 1
        curr_length = lengths[level]
        if curr_length > last_bucket_length:
            bucket_lengths.append(curr_length)
            last_bucket_length = curr_length
    return bucket_lengths


def collect_buckets(lengths, buckets_number=None, max_bucket_length=-1):
    bucket_lengths = make_bucket_lengths(lengths, buckets_number, max_bucket_length)
    indexes = [[] for length in bucket_lengths]
    for i, length in enumerate(lengths):
        index = bisect.bisect_left(bucket_lengths, length)
        indexes[index].append(i)
    if max_bucket_length != -1:
        bucket_lengths = list(chain.from_iterable(
            ([L] * ((len(curr_indexes)-1) // max_bucket_length + 1))
            for L, curr_indexes in zip(bucket_lengths, indexes)
            if len(curr_indexes) > 0))
        indexes = [curr_indexes[start:start+max_bucket_length]
                   for curr_indexes in indexes
                   for start in range(0, len(curr_indexes), max_bucket_length)]
    return [(L, curr_indexes) for L, curr_indexes
            in zip(bucket_lengths, indexes) if len(curr_indexes) > 0]


def make_table(data, length, indexes, fill_value=None, fill_with_last=False):
    """
    Погружает строки data с номерами из indexes
    в таблицу ширины length, дополняя их справа
    либо значением fill_value, либо последним значением, увеличенным на 1

    letter_positions: list of lists of int
    length: int
    indexes: list of ints
    """
    answer = np.zeros(shape=(len(indexes), length), dtype=int)
    if fill_value is not None:
        answer.fill(fill_value)
    for i, index in enumerate(indexes):
        curr = data[index]
        L = len(curr)
        answer[i,:L] = curr
        if fill_with_last:
            answer[i,L:] = curr[-1] + 1
    return answer


def generate_data(X, indexes_by_buckets, batches_indexes, batch_size,
                   symbols_number, shuffle=True, weights=None, nepochs=None):
    nsteps = 0
    while nepochs is None or nsteps < nepochs:
        if shuffle:
            for elem in indexes_by_buckets:
                np.random.shuffle(elem)
            np.random.shuffle(batches_indexes)
        for i, start in batches_indexes:
            curr_bucket, bucket_size = X[i], len(X[i][0])
            end = min(bucket_size, start + batch_size)
            curr_indexes = indexes_by_buckets[i][start:end]
            to_yield = [elem[curr_indexes] for elem in curr_bucket[:-1]]
            y_to_yield = curr_bucket[-1][curr_indexes]
            # веса объектов
            # преобразуем y_to_yield в бинарный формат
            y_to_yield = to_one_hot(y_to_yield, symbols_number)
            # yield (to_yield, y_to_yield, weights_to_yield)
            if weights is None:
                yield (to_yield, y_to_yield)
            else:
                weights_to_yield = weights[i][curr_indexes]
                yield (to_yield, y_to_yield, weights_to_yield)
        nsteps += 1