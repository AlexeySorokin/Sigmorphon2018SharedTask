import sys
import getopt
import os
import ujson as json

from read import read_languages_infile, read_infile
from inflector import Inflector
from mcmc_aligner import Aligner


def read_params(infile):
    with open(infile, "r", encoding="utf8") as fin:
        params = json.load(fin)
    if "model" not in params:
        params["model"] = dict()
    return params

SHORT_OPTS = "l:t:o:"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = os.path.join("conll2018", "task1", "all")
    for opt, val in opts:
        if opt == "-l":
            languages = read_languages_infile(val)
        elif opt == "-o":
            outfile = val
        elif opt == "-c":
            corr_dir = val
    if languages is None:
        languages = [elem.rsplit("-", maxsplit=2) for elem in os.listdir(corr_dir)]
        languages = [(elem[0], elem[2]) for elem in languages if elem[1] == "train" and len(elem) >= 3]
    params = read_params(args[0])
    results = []
    for language, mode in languages:
        infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
        data, dev_data = read_infile(infile), None
        if mode != "high":
            dev_file = os.path.join(corr_dir, "{}-dev".format(language))
            dev_data = read_infile(dev_file)
        data_for_alignment = [elem[:2] for elem in data]
        aligner = Aligner(n_iter=1, separate_endings=True, init="lcs",
                          init_params={"gap": 2, "initial_gap": 3})
        aligned_data = aligner.align(data_for_alignment, save_initial=False)
        inflector = Inflector(**params["model"]).train(data, dev_data=dev_data)
        # print(inflector.label_codes_)
