#include <math.h>
#include <stdlib.h>
#include <iostream>

using std::cerr;

int MAX_DIFF = 50;
double INF = 1000.0;
int first_call = 1;
int EMPTY = 0;

// typedef int (*f_type)(double, double, double)

// double MIN3(double x, double y, double x)
// {
    // return (x ? x <= z : z) ? x <= y : (y ? y < z : z);
// }

double make_cost(double count, int total_count, int m, double prior);

struct ForCostParameters
{
    int max_code;
    int row_length;
    double* counts;
    int total_count;
    int pairs_number;
    double prior;
    
    ForCostParameters(int max_code_, int row_length_, double* counts_,
                      int total_count_, int pairs_number_, double prior_):
        max_code(max_code_), row_length(row_length_), counts(counts_),
        total_count(total_count_), pairs_number(pairs_number_), prior(prior_)
        {};
};

double log_add(double x, double y)
{
    double temp;
    if (y > x)
    {
        // делаем x > y
        temp = x;
        x = y;
        y = temp;
    }
    // -log(exp(-x) + exp(-y)) = x - log(1 + exp(x - y))
    if (x >= INF && y >= INF)
        return INF;
    if (x - y > MAX_DIFF)
        return y;
    return x - log(1.0 + exp(x - y));
}

int draw_sample2(double x, double y, int random_state)
{
    double min_prob, subv, r;
    if (first_call)
    {
        srand(random_state);
        first_call = 0;
    }
    
    if (x > y + MAX_DIFF)
    {
        return 1;
    }
    else if (y > x + MAX_DIFF)
    {
        return 0;
    }
    min_prob = (x < y) ? x : y;
    if (min_prob > 2)
    {
        subv = min_prob - 2;
        x = x - subv;
        y = y - subv;
    }
    x = exp(-x);
    y = exp(-y);
    r = ((double)rand() / RAND_MAX) * (x + y);
    return (r <= x) ? 0 : 1;  
}

int draw_sample(double x, double y, double z, int random_state)
{
    double min_prob, subv, r;
    if (first_call)
    {
        srand(random_state);
        first_call = 0;
    }
    
    min_prob = (x < y) ? ((x < z) ? x : z) : ((y < z) ? y : z);
    if (min_prob > 2)
    {
        subv = min_prob - 2;
        x = x - subv;
        y = y - subv;
        z = z - subv;
    }
    x = exp(-x);
    y = exp(-y);
    z = exp(-z);
    r = ((double)rand() / RAND_MAX) * (x + y + z);
    // printf("%.2f %.2f %.2f %.2f %.2f\n", x , y, z, x+y+z, r);
    return (r <= x) ? -1 : (r < (x + y) ? 0 : 1);
}

int make_pair_code(int i, int j, int m, int n)
{
    return n * (i < m ? i : m-1) + (j < n ? j : n-1);
}

double _get_cost(int x, int y, ForCostParameters data)
{
    double count = 1.0, mult = 1.0, prior = data.prior;
    int pair_code = -1;
    
    // max_code --- максимальное разрешённое значение x
    // row_length-1 --- максимальное разрешённое значение x
    if (x < data.max_code && y < data.row_length - 1)
        pair_code = make_pair_code(x, y, data.max_code+1, data.row_length);
    else
    {
        mult = data.prior * (data.max_code + data.row_length) / data.total_count;
        prior = 0.0;
        if (x < data.max_code || y < data.row_length - 1 || x == y)
            pair_code = make_pair_code(x, y, data.max_code+1, data.row_length);
    }
    if (pair_code >= 0)
        count = data.counts[pair_code] * mult + prior;
    return make_cost(count, data.total_count, data.pairs_number, data.prior);
}

double make_cost(double count, int total_count, int m, double prior)
{
    return -log(count / double(total_count + m * prior));
}

void fill_trellis_1(double* trellis, int* first, int* second,
                    double* counts, int total_count, int total_number, 
                    double prior, int symbols_number, int m, int n)
{
    ForCostParameters data = ForCostParameters(
        symbols_number, symbols_number+1, counts, total_count, total_number, prior);
    
    trellis[0] = 0.0;
    for (int j = 1; j < n+1; ++j)
    {
        // int pair_code = second[j-1];
        // trellis[j] = trellis[j-1] + make_cost(
            // counts[pair_code]+prior, total_count, total_number, prior);
        trellis[j] = trellis[j-1] + _get_cost(0, j, data);
    }
        
    for (int i = 1; i < m+1; ++i)
    {
        int pair_code = (symbols_number+1) * first[i-1];
        trellis[(n+1)*i] = trellis[(n+1)*(i-1)] + _get_cost(first[i-1], 0, data);
    }
    
    for (int i = 1; i < m+1; ++i)
    {
        // int removal_code = (symbols_number+1) * first[i-1];
        // double removal_cost = make_cost(
            // counts[removal_code]+prior, total_count, total_number, prior);
        double removal_cost = _get_cost(first[i-1], 0, data);
        for (int j = 1; j < n+1; ++j)
        {
            int s = (n+1)* i + j;
            // int insertion_code = second[j-1];
            // double insertion_cost = make_cost(
                // counts[insertion_code]+prior, total_count, total_number, prior);
            double insertion_cost = _get_cost(0, second[j-1], data);
            // int change_code = (symbols_number+1) * first[i-1] + second[j-1];
            // double change_cost = make_cost(
                // counts[change_code]+prior, total_count, total_number, prior);
            double change_cost = _get_cost(first[i-1], second[j-1], data);
            // printf("%d %d %d %d %d %d %d % d %.2lf %.2lf %.2lf\n", 
                // i, j, first[i-1], second[j-1], s, removal_code, insertion_code, change_code,
                // insertion_cost, removal_cost, change_cost);
            trellis[s] = log_add(
                log_add(trellis[s-n-1] + removal_cost, trellis[s-n-2] + change_cost),
                trellis[s-1] + insertion_cost);
        }
    }
    return;
}

void fill_trellis(double* change_costs, double* insertion_costs,
                  double* removal_costs, double* trellis, int m, int n)
{
    trellis[0] = 0.0;
    for (int j = 1; j < n+1; ++j)
        trellis[j] = trellis[j-1] + insertion_costs[j-1];
    for (int i = 1; i < m+1; ++i)
        trellis[(n+1)*i] = trellis[(n+1)*(i-1)] + removal_costs[i-1];
    for (int i = 1; i < m+1; ++i)
    {
        for (int j = 1; j < n+1; ++j)
        {
            int s = (n+1)* i + j;
            trellis[s] = log_add(
                log_add(trellis[s-n-1] + removal_costs[i-1],
                        trellis[s-n-2] + change_costs[n*(i-1)+j-1]),
                trellis[s-1] + insertion_costs[j-1]);
        }
    }
    return;
}

void fill_trellis_2(double* trellis_root, double* trellis_ending,
                    double* trellis_equal, int* first, int* second, 
                    double* counts, int total_count, int total_number, 
                    double prior, int symbols_number, int m, int n)
{
    int row_length = 2 * symbols_number + 1;  // длина строки в матрице счётчиков
    ForCostParameters data = ForCostParameters(
        symbols_number, row_length, counts, total_count, total_number, prior);
    // cerr << "fill trellis entered\n";
    bool* can_be_stem = new bool[(m+1)*(n+1)];  // находим, какие позиции решётки соответствуют основе
    for (int i = 0, s = 0; i < m + 1; ++i)
    {
        for (int j = 0; j < n + 1; ++j, ++s)
        {
            // can_be_stem[i][j] = (first[i] == second[j])
            can_be_stem[s] = (i < m && j < n && first[i] == second[j]);
        }
    }
    // cerr << "stem_e filled\n";
    for (int i = m - 2, s = (n+1)*(m-1)-2; i >= 0; --i, --s)
    {
        for (int j = n-1; j >= 0; --j, --s)
        {
            // can_be_stem[i][j] = \exists i' >= i (first[i'] == second[j])
            can_be_stem[s] |= can_be_stem[s+n+1];
        }
    }
    for (int i = m - 1, s = (n+1)*m-3; i >= 0; --i, s-=2)
    {
        for (int j = n-2; j >= 0; --j, --s)
        {
            // can_be_stem[i][j] = \exists j' >= j \exists i' >= i (first[i'] == second[j'])
            can_be_stem[s] |= can_be_stem[s+1];
        }
    }
    // cerr << "stem_E filled\n";
    for (int i = 1, s = n + 2; i < m + 1; ++i, ++s)
    {
        for (int j = 1; j < n + 1; ++j, ++s)
        {
            can_be_stem[s] |= (first[i-1] == second[j-1]);
        }
    }
    
    // cerr << "can_be_stem filled\n";
    // дополнительная решётка для случаев, когда последние символы обоих слов совпадают
    // нужна для корректного вычисления вероятностей основы и не основы
    // считаем, что в начале слова состояние "последние буквы равны"
    trellis_root[0] = trellis_ending[0] = INF;
    trellis_equal[0] = 0.0;
    // инициализация: только вставки
    for (int j = 1; j < n+1; ++j)
    {
        int pair_code = second[j-1];
        double prev_value = (j == 1) ? 0.0 : trellis_root[j-1];
        if (can_be_stem[j])
        {
            // trellis_root[j] = prev_value + make_cost(
                // counts[pair_code]+prior, total_count, total_number, prior);
            trellis_root[j] = prev_value + _get_cost(0, second[j-1], data);
        }
        else
        {
            trellis_root[j] = INF;
        }
        prev_value = (j == 1) ? 0.0 : trellis_ending[j-1];
        // разность между кодом символа в окончании и корне всегда равна symbols_number
        // trellis_ending[j] = prev_value + make_cost(
            // counts[pair_code+symbols_number]+prior, total_count, total_number, prior);
        trellis_ending[j] = prev_value + _get_cost(0, second[j-1]+symbols_number, data);
        trellis_equal[j] = INF;
    }
    // инициализация: только удаления
    for (int i = 1, s = n+1; i < m+1; ++i, s += n+1)
    {
        int pair_code = row_length * first[i-1];
        double prev_value = (i == 1) ? 0.0 : trellis_root[s-n-1];
        if (can_be_stem[s])
        {
            // trellis_root[s] = prev_value + make_cost(
                // counts[pair_code]+prior, total_count, total_number, prior);
            trellis_root[s] = prev_value + _get_cost(first[i-1], EMPTY, data);
        }
        else
        {
            trellis_root[s] = INF;
        }
        prev_value = (i == 1) ? 0.0 : trellis_ending[s-n-1];
        // разность между кодом символа в окончании и корне всегда равна symbols_number
        // trellis_ending[s] = prev_value + make_cost(
            // counts[pair_code+symbols_number]+prior, total_count, total_number, prior);
        trellis_ending[s] = prev_value + _get_cost(first[i-1], EMPTY+symbols_number, data);
        trellis_equal[s] = INF;
    }
    for (int i = 1, s = n + 2; i < m+1; ++i, ++s)
    {
        // стоимость удаления
        int removal_code = row_length * first[i-1];
        // double removal_cost = make_cost(
            // counts[removal_code]+prior, total_count, total_number, prior);
        double removal_cost = _get_cost(first[i-1], EMPTY, data);
        double ending_removal_cost = _get_cost(first[i-1], EMPTY+symbols_number, data);
        // double ending_removal_cost = make_cost(
            // counts[removal_code+symbols_number]+prior, total_count, total_number, prior);
        for (int j = 1; j < n+1; ++j, ++s)
        {
            // стоимость вставки
            int insertion_code = second[j-1];
            // double insertion_cost = make_cost(
                // counts[insertion_code]+prior, total_count, total_number, prior);
            double insertion_cost = _get_cost(EMPTY, second[j-1], data);
            // double ending_insertion_cost = make_cost(
                // counts[insertion_code+symbols_number]+prior, total_count, total_number, prior);
            double ending_insertion_cost = _get_cost(
                EMPTY, second[j-1]+symbols_number, data);
            // int change_code = row_length * first[i-1] + second[j-1];
            double change_cost = _get_cost(first[i-1], second[j-1], data);
            // double change_cost = make_cost(
                // counts[change_code]+prior, total_count, total_number, prior);
            double ending_change_cost = _get_cost(
                first[i-1], second[j-1]+symbols_number, data);
            // double ending_change_cost = make_cost(
                // counts[change_code+symbols_number]+prior, total_count, total_number, prior);
            // стоимости с предыдущих шагов
            double left_root_cost = log_add(trellis_root[s-1], trellis_equal[s-1]);
            double left_ending_cost = log_add(trellis_ending[s-1], trellis_equal[s-1]);
            double up_root_cost = log_add(trellis_root[s-n-1], trellis_equal[s-n-1]);
            double up_ending_cost = log_add(trellis_ending[s-n-1], trellis_equal[s-n-1]);
            double diag_root_cost = log_add(trellis_root[s-n-2], trellis_equal[s-n-2]);
            double diag_ending_cost = log_add(trellis_ending[s-n-2], trellis_equal[s-n-2]);
            // последняя операция не тождественная
            trellis_root[s] = can_be_stem[s] ? log_add(
                left_root_cost + insertion_cost, up_root_cost + removal_cost) : INF;
            trellis_ending[s] = log_add(left_ending_cost + ending_insertion_cost, 
                                        up_ending_cost + ending_removal_cost);
            if (first[i-1] == second[j-1])
            {
                    trellis_equal[s] = diag_root_cost + change_cost;
            }
            else
            {
                    trellis_equal[s] = INF;
                    if (can_be_stem[s])
                    {
                        // если не может быть основы, то не пересчитываем значение
                        trellis_root[s] = log_add(trellis_root[s], diag_root_cost + change_cost);
                    }
                    trellis_ending[s] = log_add(trellis_ending[s], 
                                                diag_ending_cost + ending_change_cost);
            }
        }
    }
    for (int s = 0; s < (m+1)*(n+1); ++s)
    {
        if (trellis_equal[s] < INF)
            trellis_root[s] = log_add(trellis_root[s], trellis_equal[s]);
    }
        
    delete[] can_be_stem;
    return;
}
    
// int draw_min(double x, double y, double z):
    // return (-1 ? x <= z : 1) ? x <= y : (0 ? y < z : 1)

// int draw_sample(double x, double y, double z):
    // double min_prob, subv, r;
    
    // min_prob = MIN3(x, y, z);
    // if (min_prob > 2)
        // subv = min_prob - 2;
        // x = x - subv;
        // y = y - subv;
        // z = z - subv;
    // x = exp(x);
    // y = exp(y);
    // z = exp(z);
    // r = (rand() / RAND_MAX) * (x + y + z);
    // return (-1 ? r < x : (0 ? r < x + y : 1));