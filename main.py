import sys
import getopt
import os
import ujson as json

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from inflector import Inflector, load_inflector, predict_missed_answers
from paradigm_classifier import ParadigmChecker
from write import output_analysis

DEFAULT_PARAMS = {"beam_width": 1}

def read_params(infile):
    with open(infile, "r", encoding="utf8") as fin:
        params = json.load(fin)
    params["use_lm"] = params.get("use_lm", False)
    if "model" not in params:
        params["model"] = dict()
    if "predict" not in params:
        params["predict"] = dict()
    return params

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
SHORT_OPTS = "l:o:S:L:m:tTP:p"

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=config))
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = os.path.join("conll2018", "task1", "all")
    save_dir, load_dir, model_name = None, None, None
    analysis_dir, pred_dir = "results", "predictions"
    to_train, to_test = True, True
    predict_dir, to_predict = None, False
    use_paradigms = False
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
        elif opt == "-p":
            use_paradigms = True
    if languages is None:
        languages = [elem.rsplit("-", maxsplit=2) for elem in os.listdir(corr_dir)]
        languages = [(elem[0], elem[2]) for elem in languages if elem[1] == "train" and len(elem) >= 3]
    params = read_params(args[0])
    results = []
    model_format_string = '{1}-{2}' if model_name is None else '{0}-{1}-{2}'
    print(sorted(languages))
    for language, mode in sorted(languages):
        print(language, mode)
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
            lm_dir = params.get("lm_dir")
            lm_name = params.get("lm_name")
            lm_file = "{}-{}.json".format(language, mode)
            if lm_name is not None:
                lm_file = lm_name + "-" + lm_file
            lm_file = os.path.join(lm_dir, lm_file)
            use_lm = params["use_lm"] and os.path.exists(lm_file)
            inflector = Inflector(use_lm=use_lm, lm_file=lm_file, **params["model"])
        save_file = os.path.join(save_dir, filename + ".json") if save_dir is not None else None
        if to_train:
            inflector.train(data, dev_data=dev_data, save_file=save_file)
        if use_paradigms:
            paradigm_checker = ParadigmChecker().train(data)
        if to_test:
            answer = inflector.predict(test_data, **params["predict"])
            if use_paradigms:
                data_to_filter = [(elem[0], elem[2]) for elem in test_data]
                words_in_answer  = [[x[0] for x in elem] for elem in answer]
                probs_in_answer = [[x[1:] for x in elem]for elem in answer]
                answer = paradigm_checker.filter(data_to_filter, words_in_answer, probs_in_answer)
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
            answer = inflector.predict(data, feat_column=-1, **params["predict"])
            outfile = os.path.join(predict_dir, filename+"-out")
            if not os.path.exists(outfile):
                continue
            with open(outfile, "w", encoding="utf8") as fout:
                for source, predictions in zip(data, answer):
                    predicted_words = [elem[0] for elem in predictions]
                    for word in predicted_words[:1]:
                        fout.write("\t".join([source[0], word, ";".join(source[-1])]) + "\n")

