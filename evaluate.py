import sys
import getopt
import os

from Levenshtein import distance

from read import read_languages_infile, read_infile, MODES

WIDTHS = [16, 6]


def prettify_metrics(metrics):
    format_string, metrics_data = [], []
    for i, elem in enumerate(metrics):
        width = WIDTHS[i] if i < len(WIDTHS) else None
        format_string.append(get_format_string(elem, width=width))
        if isinstance(elem, list):
            metrics_data.append(" ".join(str(x) for x in elem))
        elif isinstance(elem, dict):
            metrics_data.append(" ".join("{}:{:.2f}".format(*x) for x in sorted(elem.items())))
        else:
            metrics_data.append(elem)
    format_string = "\t".join(format_string) + "\n"
    return format_string.format(*metrics_data), metrics_data

def evaluate(test_data, corr_data):
    corr, cov, total, total_dist, best_dist = 0, 0, len(test_data), 0.0, 0.0
    for i, (test_elem, corr_elem) in enumerate(zip(test_data, corr_data), 1):
        test_words = test_elem[1]
        corr_word = corr_elem[1]
        if corr_word == test_words[0]:
            corr += 1
        else:
            best_distance = distance(corr_word, test_words[0])
            total_dist += best_distance
        if any(x == corr_word for x in test_words):
            cov += 1
            continue
        for word in test_words[:1]:
            d = distance(corr_word, word)
            if d == 1:
                best_dist += d
                break
            best_distance = min(best_distance, d)
        else:
            best_dist += best_distance
    answer = [100 * corr / total, 100 * cov / total, total_dist / total, best_dist / total]
    return answer


def evaluate_files(infile, corr_file, sep=" "):
    test_data, corr_data = read_infile(infile), read_infile(corr_file)
    for elem in test_data:
        elem[1] = elem[1].split(sep)
    return evaluate(test_data, corr_data)


def get_format_string(x, width=None):
    if isinstance(x, float):
        return "{:.2f}"
    elif width is not None and isinstance(x, str):
        return "{{:<{}}}".format(width)
    else:
        return "{}"

SHORT_OPTS = "l:o:t:c:vs:m:T"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages, outfile, verbose, sep, model_name = None, None, False, " ", None
    test_dir, corr_dir = "baseline-results", "conll2018/task1/all"
    suffix = "dev"
    for opt, val in opts:
        if opt == "-l":
            languages = read_languages_infile(val)
        elif opt == "-o":
            outfile = val
        elif opt == "-t":
            test_dir = val
        elif opt == "-c":
            corr_dir = val
        elif opt == "-v":
            verbose = True
        elif opt == "-s":
            sep = val
        elif opt == "-m":
            model_name = val
        elif opt == "-T":
            suffix = "test"
    if languages is None:
        languages = [tuple(elem.rsplit("-", maxsplit=2)[:2]) for elem in os.listdir(test_dir)]
    results = []
    for language, mode in languages:
        filename = "{}-{}-out".format(language, mode)
        if model_name is not None:
            filename = "{}-{}".format(model_name, filename)
        infile = os.path.join(test_dir, filename)
        if os.path.exists(infile):
            corr_file = os.path.join(corr_dir, "{}-{}".format(language, suffix))
            results.append([language, mode] + list(evaluate_files(infile, corr_file, sep=sep)))
    results.sort(key=lambda x: (x[0], MODES.index(x[1])))
    if outfile is not None:
        fout = open(outfile, "w", encoding="utf8")
    for curr_results in results:
        format_string, data = [], []
        for i, elem in enumerate(curr_results):
            width = WIDTHS[i] if i < len(WIDTHS) else None
            format_string.append(get_format_string(elem, width=width))
            if isinstance(elem, list):
                data.append(" ".join(str(x) for x in elem))
            elif isinstance(elem, dict):
                data.append(" ".join("{}:{:.2f}".format(*x) for x in sorted(elem.items())))
            else:
                data.append(elem)
        format_string = "\t".join(format_string) + "\n"
        if outfile is not None:
            fout.write(format_string.format(*data))
        if verbose:
            print(format_string.format(*data), end="")
    if outfile is not None:
        fout.close()


