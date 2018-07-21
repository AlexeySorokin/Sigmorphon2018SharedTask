import sys
import getopt
import os

from Levenshtein import distance

from read import read_languages_infile, read_infile, MODES

WIDTHS = [16, 6]


def evaluate_files(infile, corr_file):
    test_data, corr_data = read_infile(infile), read_infile(corr_file)
    corr, cov, total, total_dist, best_dist = 0, 0, len(test_data), 0.0, 0.0
    for test_elem, corr_elem in zip(test_data, corr_data):
        test_words = test_elem[1].split()
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


def get_format_string(x, width=None):
    if isinstance(x, float):
        return "{:.2f}"
    elif width is not None and isinstance(x, str):
        return "{{:<{}}}".format(width)
    else:
        return "{}"

SHORT_OPTS = "l:o:t:c:v"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages, outfile, verbose = None, None, False
    test_dir, corr_dir = "conll2018/task1/baseline-results", "conll2018/task1/all"
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
    if languages is None:
        languages = [tuple(elem.rsplit("-", maxsplit=2)[:2]) for elem in os.listdir(test_dir)]
    results = []
    for language, mode in languages:
        infile = os.path.join(test_dir, "{}-{}-out".format(language, mode))
        corr_file = os.path.join(corr_dir, "{}-dev".format(language))
        results.append([language, mode] + list(evaluate_files(infile, corr_file)))
    if outfile is not None:
        results.sort(key = lambda x: (x[0], MODES.index(x[1])))
        with open(outfile, "w", encoding="utf8") as fout:
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
                fout.write(format_string.format(*data))
                if verbose:
                    print(format_string.format(*data), end="")

