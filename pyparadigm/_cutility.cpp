#include <vector>
#include <algorithm>
#include <limits>
#include <iostream>

using std::vector;
using std::less;
using std::numeric_limits;
using std::sort;
using std::lower_bound;
using std::upper_bound;
using std::cout;
using std::endl;

template <class T>
vector<int> & _find_first_indexes(
        const vector<T> & first, const vector<T> & second, 
        vector<int> & indexes, bool strict=true)
{
    int m = first.size();
    int n = second.size();
    int i = 0;
    int j = 0;
    indexes.resize(m);
    while (i < m && j < n)
    {
        bool pred_value;
        if (strict)
            pred_value = (first[i] < second[j]);
        else
            pred_value = (first[i] <= second[j]);
        if (pred_value)
        {
            indexes[i] = j;
            ++i;
        }
        else
        {
            j++;
        }
    }
    if (i < m)
    {
        for(j = i; j < m; ++j)
        {
            indexes[j] = n;
        }
    }
    return indexes;
}


vector < vector<int> > _extract_ordered_sequences_(
    vector < vector <int> > & lists, const vector <int> & min_differences=vector<int>(), 
    const vector <int> & max_differences=vector<int>(), bool strict_min=true, bool strict_max=false)
{
    /*
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
    */
    vector <vector <int> > answer;
    int m = lists.size();
    if (m == 0)
        return answer;
    // min_differences не может быть пустым, так что это точно аргумент по умолчанию
    vector <int> min_differences_, max_differences_;
    if (min_differences.size() == 0)
    {
        min_differences_ = vector <int> (m, 0);
        min_differences_[0] = 1 - numeric_limits<int>::max() / 10;
    }
    else
        min_differences_ = min_differences;
    // else
    // {
        // if len(min_differences) != m:
            // raise ValueError("Lists and min_differences must have equal length")
    // }
    if (max_differences.size() == 0)
    {
        max_differences_ = vector <int> (m, numeric_limits<int>::max() / 10);
    }
    else
        max_differences_ = max_differences;
    // {
        // if len(max_differences) != m:
            // raise ValueError("Lists and min_differences must have equal length")
    // }
    for (vector<int>::const_iterator first = max_differences_.begin(), second = min_differences_.begin();
         first != max_differences_.end() && second != min_differences_.end(); ++first, ++second)
    {
         if (*first < *second)
         {
             return answer;
         }
    }
    for (vector < vector <int> >::iterator iter = lists.begin(); iter != lists.end(); ++iter)
    {
        sort(iter->begin(), iter->end());
    }
    vector <int> list_lengths = vector<int>(m);
    for (int i = 0; i < m; ++i)
    {
        list_lengths[i] = lists[i].size();
        if (list_lengths[i] == 0)
            return answer;
    }
    vector < vector <int> > min_indexes(m-1);
    vector < vector <int> > max_indexes(m-1);
    for (int i = 0; i < m - 1; ++i)
    {
            vector <int> curr_shifted_for_min(lists[i].size());
            vector <int> curr_shifted_for_max(lists[i].size());
            for (int j = 0; j < lists[i].size(); ++j)
            {
                curr_shifted_for_min[j] = lists[i][j] + min_differences_[i+1];
                curr_shifted_for_max[j] = lists[i][j] + max_differences_[i+1];
                // cout << curr_shifted_for_min[j] << " " << curr_shifted_for_max[j] << endl;
            }
            _find_first_indexes(curr_shifted_for_min, lists[i+1], min_indexes[i], strict_min);
            _find_first_indexes(curr_shifted_for_max, lists[i+1], max_indexes[i], strict_max);
    }
    // for (vector < vector <int> >::const_iterator iter = min_indexes.begin(); iter != min_indexes.end(); ++iter)
    // {
        // for(vector <int>::const_iterator curr=iter->begin(); curr != iter->end(); ++curr)
        // {
            // cout << *curr << " ";
        // }
        // cout << endl;
    // }
    // cout << endl;
    // for (vector < vector <int> >::const_iterator iter = max_indexes.begin(); iter != max_indexes.end(); ++iter)
    // {
        // for(vector <int>::const_iterator curr=iter->begin(); curr != iter->end(); ++curr)
        // {
            // cout << *curr << " ";
        // }
        // cout << endl;
    // }
    // cout << endl;
    // находим минимальную позицию первого элемента
    vector <int>::iterator startpos_iter =\
        lower_bound(lists[0].begin(), lists[0].end(), min_differences_[0]);
    if (startpos_iter == lists[0].end())
        return answer;
    int startpos = startpos_iter - lists[0].begin();
    vector <int>::iterator endpos_iter = upper_bound(lists[0].begin(), lists[0].end(), max_differences_[0]);
    int endpos = endpos_iter - lists[0].begin();
    vector <int> sequence(1, *startpos_iter);
    vector <int> index_sequence(1, startpos);
    // int iterations_count = 0;
    // TO BE MODIFIED
    while (sequence.size() > 0)
    {
        // обходим все монотонные последовательности в лексикографическом порядке
        // ++iterations_count;
        // if (iterations_count >= 20)
            // return answer;
        int i = index_sequence.size() - 1;
        int li = index_sequence[i];
        // cout << "Here!" << endl;
        // for (vector<int>::const_iterator iter = sequence.begin(); iter != sequence.end(); ++iter)
        // {
            // cout << *iter << " ";
        // }
        // cout << endl;
        // for (vector<int>::const_iterator iter = index_sequence.begin(); iter != index_sequence.end(); ++iter)
        // {
            // cout << *iter << " ";
        // }
        // cout << endl << i << " " << li << " ";
        if (i < m - 1 && min_indexes[i][li] < max_indexes[i][li])
        {
            // добавляем минимально возможный элемент в последовательность
            int next_index = min_indexes[i][li];
            ++i;
            index_sequence.push_back(next_index);
            sequence.push_back(lists[i][next_index]);
        }
        else
        {
            if (i == m - 1)
            {
                // если последовательность максимальной длины, то обрезаем последний элемент
                // чтобы сохранить все возможные варианты для последней позиции
                index_sequence.pop_back();
                sequence.pop_back();
                --i;
                int curr_max_index = (i >= 0) ? max_indexes[i][index_sequence[i]] : endpos;
                while (li < curr_max_index)
                {
                    // iterations_count++;
                    // if (iterations_count == 20)
                        // return answer;
                    vector <int> new_sequence = sequence;
                    new_sequence.push_back(lists[i+1][li]);
                    answer.push_back(new_sequence);
                    // for (vector<int>::const_iterator iter = new_sequence.begin(); iter != new_sequence.end(); ++iter)
                    // {
                        // cout << " " << *iter;
                    // }
                    // cout << endl;
                    ++li;
                }
                if (i < 0)
                    break;
            }
            // пытаемся перейти к следующей в лексикографическом порядке последовательности
            index_sequence[i] += 1;
            while (i > 0 && index_sequence[i] == max_indexes[i-1][index_sequence[i-1]])
            {
                // увеличиваем последний индекс, если выходим за границы, то
                // укорачиваем последовательность и повторяем процедуру
                index_sequence.pop_back();
                --i;
                ++index_sequence[i];
            }
            if (i == 0 && index_sequence[0] == endpos)
                break;
            sequence.resize(i+1);
            sequence[i] = lists[i][index_sequence[i]];
        }
    }
    return answer;
}
