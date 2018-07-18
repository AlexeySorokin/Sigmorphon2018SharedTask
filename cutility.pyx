from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "_cutility.cpp" :
    vector[int] & _find_first_indexes(
        const vector[int] & first, const vector[int] & second,
        vector[int] & indexes, bool strict)
    vector [ vector[int] ] & _extract_ordered_sequences_(
        vector [ vector [int] ] & lists, const vector [int] & min_differences,
        const vector [int] & max_difference, bool strict_min, bool strict_max)
    
def _find_indexes(first, second, strict=True):
    # cdef vector[double] first_i, second_i;
    cdef vector[int] first_i, second_i;
    cdef vector[int] answer_i;
    for a in first:
        first_i.push_back(a)
    for b in second:
        second_i.push_back(b)
    answer_i.resize(first_i.size())
    _find_first_indexes(first_i, second_i, answer_i, strict)
    return answer_i

def _extract_ordered_sequences(lists, min_differences=None, max_differences=None,
                               strict_min=True, strict_max=False):
    cdef vector [vector[int]] lists_;
    cdef vector [int] curr_list;
    for i, elem in enumerate(lists):
        curr_list.resize(len(elem));
        for j, x in enumerate(elem):
            curr_list[j] = x;
        lists_.push_back(curr_list);
    cdef vector[int] min_differences_;
    if min_differences is not None:
        if len(min_differences) != len(lists):
            raise ValueError("lists and min_differences should have equal length")
        for x in min_differences:
            min_differences_.push_back(x);
    cdef vector[int] max_differences_;
    if max_differences is not None:
        if len(max_differences) != len(lists):
            raise ValueError("lists and max_differences should have equal length")
        for x in max_differences:
            max_differences_.push_back(x);
    cdef vector [ vector [int] ] answer = _extract_ordered_sequences_(
        lists_, min_differences_, max_differences_, strict_min, strict_max)
    return answer

