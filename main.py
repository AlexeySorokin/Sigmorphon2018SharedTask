import sys
import getopt
import os
import ujson as json

from read import read_languages_infile, read_infile
from inflector import Inflector, load_inflector
from mcmc_aligner import Aligner

DEFAULT_PARAMS = {"beam_width": 1}

def read_params(infile):
    with open(infile, "r", encoding="utf8") as fin:
        params = json.load(fin)
    if "model" not in params:
        params["model"] = dict()
    if "predict" not in params:
        params["predict"] = dict()
    return params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SHORT_OPTS = "l:o:S:L:m:tT"

if __name__ == "__main__":
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = os.path.join("conll2018", "task1", "all")
    out_dir, save_dir, load_dir, model_name = "results", None, None, None
    to_train, to_test = True, True
    for opt, val in opts:
        if opt == "-l":
            languages = read_languages_infile(val)
        elif opt == "-o":
            out_dir = val
        elif opt == "-c":
            corr_dir = val
        elif opt == "-S":
            save_dir = val
        elif opt == "-L":
            load_dir = val
        elif opt == "-m":
            model_name = val
        elif opt == "-t":
            to_train = False
        elif opt == "-T":
            to_test = False
    if languages is None:
        languages = [elem.rsplit("-", maxsplit=2) for elem in os.listdir(corr_dir)]
        languages = [(elem[0], elem[2]) for elem in languages if elem[1] == "train" and len(elem) >= 3]
    params = read_params(args[0])
    results = []
    model_format_string = '{1}-{2}' if model_name is None else '{0}-{1}-{2}'
    for language, mode in languages:
        infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
        test_file = os.path.join(corr_dir, "{}-dev".format(language))
        data, dev_data, test_data = read_infile(infile), None, read_infile(test_file)
        data *= params.get("data_multiple", 1)
        if mode != "high":
            dev_data = test_data
        # data_for_alignment = [elem[:2] for elem in data]
        # aligner = Aligner(n_iter=1, separate_endings=True, init="lcs",
        #                   init_params={"gap": 2, "initial_gap": 3})
        # aligned_data = aligner.align(data_for_alignment, save_initial=False)
        filename = model_format_string.format(model_name, language, mode)
        load_file = os.path.join(load_dir, filename + ".json") if load_dir is not None else None
        if load_file and os.path.exists(load_file):
            inflector = load_inflector(load_file)
        else:
            inflector = Inflector(**params["model"])
        save_file = os.path.join(save_dir, filename + ".json") if save_dir is not None else None
        if to_train:
            inflector.train(data, dev_data=dev_data, save_file=save_file)
        if to_test:
            answer = inflector.predict(data[:10], **params["predict"])
            outfile = os.path.join(out_dir, filename) if out_dir is not None else None
            if outfile is not None:
                with open(outfile, "w", encoding="utf8") as fout:
                    for source, predictions in zip(data, answer):
                        word, descr = source[0], source[2]
                        for prediction in predictions:
                            fout.write("{}\t{}\t{}\t{:.2f}\n".format(
                                word, ";".join(descr), prediction[0], 100 * prediction[2]))
