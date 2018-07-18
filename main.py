import sys
import getopt
import os

from read import read_languages_infile, read_infile
from mcmc_aligner import Aligner

SHORT_OPTS = "l:t:o:"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = "conll2018\\task1\\all"
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
    results = []
    for language, mode in languages:
        infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
        data = read_infile(infile)
        data_for_alignment = [elem[:2] for elem in data]
        aligner = Aligner(n_iter=1, separate_endings=True, init="lcs",
                          init_params={"gap": 2, "initial_gap": 3})
        aligned_data = aligner.align(data_for_alignment, save_initial=False)
        print(aligned_data[0])
