import sys
import getopt
import os
import ujson as json

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from inflector import Inflector, load_inflector, predict_missed_answers
from write import output_analysis

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
SHORT_OPTS = "l:o:S:L:m:tTP:"

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.4
    kbt.set_session(tf.Session(config=config))
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = os.path.join("conll2018", "task1", "all")
    save_dir, load_dir, model_name = None, None, None
    analysis_dir, pred_dir = "results", "predictions"
    to_train, to_test = True, True
    predict_dir, to_predict = None, False
    for opt, val in opts:
        if opt == "-l":
            languages = read_languages_infile(val)
        elif opt == "-a":
            analysis_dir = val
        elif opt == "-o":
            pred_dir = val
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
        elif opt == "-P":
            predict_dir, to_predict = val, True
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
        dev_data = test_data[:]
        # test_data = test_data[:20]
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
            inflector.train(data, dev_data=dev_data, save_file=save_file,
                            alignments_outfile="alignments-{}.out".format(filename))
        if to_test:
            answer = inflector.predict(test_data, **params["predict"])
            outfile = os.path.join(analysis_dir, filename) if analysis_dir is not None else None
            if outfile is not None:
                with open(outfile, "w", encoding="utf8") as fout:
                    for source, predictions in zip(test_data, answer):
                        word, descr = source[0], source[2]
                        for prediction in predictions:
                            fout.write("{}\t{}\t{}\t{:.2f}\n".format(
                                word, ";".join(descr), prediction[0], 100 * prediction[2]))
            pred_file = os.path.join(pred_dir, filename+"-out") if pred_dir is not None else None
            if pred_file is not None:
                with open(pred_file, "w", encoding="utf8") as fout:
                    for source, predictions in zip(test_data, answer):
                        predicted_words = [elem[0] for elem in predictions]
                        fout.write("\t".join([source[0], "#".join(predicted_words), ";".join(source[2])]) + "\n")
            answers_for_missed = predict_missed_answers(test_data, answer, inflector, **params["predict"])
            analysis_file = os.path.join(analysis_dir, filename+"-analysis") if analysis_dir is not None else None
            output_analysis(test_data, answer, analysis_file,
                            answers_for_missed=answers_for_missed)
        if to_predict:
            predict_file = os.path.join(corr_dir, "{}-covered-test".format(language))
            data = read_infile(predict_file, feat_column=1)
            answer = inflector.predict(data, **params["predict"])
            outfile = os.path.join(predict_dir, filename)
            with open(outfile, "w", encoding="utf8") as fout:
                for source, predictions in zip(test_data, answer):
                    predicted_words = [elem[0] for elem in predictions]
                    for word in predicted_words:
                        fout.write("\t".join([source[0], word, ";".join(source[2])]) + "\n")

