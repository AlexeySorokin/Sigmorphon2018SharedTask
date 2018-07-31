import sys
import getopt
import os
import ujson as json
import numpy as np

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from inflector import Inflector, load_inflector, predict_missed_answers
from neural.neural_LM import NeuralLM
from paradigm_classifier import ParadigmChecker
from evaluate import evaluate, prettify_metrics
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
SHORT_OPTS = "l:o:S:L:m:tTP:pC:eE:"

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=config))
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    languages = None
    corr_dir = os.path.join("conll2018", "task1", "all")
    save_dir, load_dir, model_name = None, None, None
    analysis_dir, pred_dir, to_evaluate, eval_outfile = "results", "predictions", False, None
    to_train, to_test = True, True
    predict_dir, to_predict = None, False
    use_paradigms, use_lm = False, False
    lm_config_path = None
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
        elif opt == "-C":
            lm_config_path = val
        elif opt == "-e":
            to_evaluate = True
        elif opt == "-E":
            eval_outfile = val
    if languages is None:
        languages = [elem.rsplit("-", maxsplit=2) for elem in os.listdir(corr_dir)]
        languages = [(elem[0], elem[2]) for elem in languages if elem[1] == "train" and len(elem) >= 3]
    params = read_params(args[0])
    results = []
    model_format_string = '{1}-{2}' if model_name is None else '{0}-{1}-{2}'
    print(sorted(languages))
    metrics = []
    for language, mode in sorted(languages):
        print(language, mode)
        infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
        test_file = os.path.join(corr_dir, "{}-dev".format(language))
        data, dev_data, test_data = read_infile(infile), None, read_infile(test_file)
        data *= params.get("data_multiple", 1)
        dev_data = test_data[:]
        # data_for_alignment = [elem[:2] for elem in data]
        # aligner = Aligner(n_iter=1, separate_endings=True, init="lcs",
        #                   init_params={"gap": 2, "initial_gap": 3})
        # aligned_data = aligner.align(data_for_alignment, save_initial=False)
        filename = model_format_string.format(model_name, language, mode)
        load_file = os.path.join(load_dir, filename + ".json") if load_dir is not None else None
        if load_file and os.path.exists(load_file):
            inflector = load_inflector(load_file, verbose=0)
            for param in ["nepochs", "batch_size"]:
                value = params["model"].get(param)
                if value is not None:
                    inflector.__setattr__(param, value)
        else:
            if params["use_lm"]:
                lm_dir = params.get("lm_dir")
                lm_name = params.get("lm_name")
                lm_file = "{}-{}.json".format(language, mode)
                if lm_name is not None:
                    lm_file = lm_name + "-" + lm_file
                if lm_dir is not None:
                    lm_file = os.path.join(lm_dir, lm_file)
                if not os.path.exists(lm_file):
                    data_for_lm = [elem[1:] for elem in data]
                    dev_data_for_lm = [elem[1:] for elem in dev_data]
                    with open(lm_config_path, "r", encoding="utf8") as fin:
                        lm_params = json.load(fin)
                    lm = NeuralLM(**lm_params)
                    lm.train(data_for_lm, dev_data_for_lm, save_file=lm_file)
                use_lm = True
            else:
                use_lm, lm_file = False, None
            print(use_lm)
            inflector = Inflector(use_lm=use_lm, lm_file=lm_file, **params["model"])
        save_file = os.path.join(save_dir, filename + ".json") if save_dir is not None else None
        if to_train:
            inflector.train(data, dev_data=dev_data, save_file=save_file)
        if use_paradigms:
            paradigm_checker = ParadigmChecker().train(data)
        if to_test:
            alignment_data = [elem[:2] for elem in data]
            # inflector.evaluate(test_data[:20], alignment_data=alignment_data)
            # sys.exit()
            answer = inflector.predict(test_data, **params["predict"])
            # if use_paradigms:
            #     data_to_filter = [(elem[0], elem[2]) for elem in test_data]
            #     words_in_answer  = [[x[0] for x in elem] for elem in answer]
            #     probs_in_answer = [[x[1:] for x in elem]for elem in answer]
            #     answer = paradigm_checker.filter(data_to_filter, words_in_answer, probs_in_answer)
            # outfile = os.path.join(analysis_dir, filename) if analysis_dir is not None else None
            # if outfile is not None:-
            #     with open(outfile, "w", encoding="utf8") as fout:
            #         for source, predictions in zip(test_data, answer):
            #             word, descr = source[0], source[2]
            #             for prediction in predictions:
            #                 fout.write("{}\t{}\t{}\t{:.2f}\n".format(
            #                     word, ";".join(descr), prediction[0], 100 * prediction[2]))
            pred_file = os.path.join(pred_dir, filename+"-out") if pred_dir is not None else None
            if pred_file is not None:
                with open(pred_file, "w", encoding="utf8") as fout:
                    for source, predictions in zip(test_data, answer):
                        predicted_words = [elem[0] for elem in predictions]
                        fout.write("\t".join([source[0], "#".join(predicted_words), ";".join(source[2])]) + "\n")
            if to_evaluate:
                answer_to_evaluate = [(word, [x[0] for x in elem], feats)
                                      for (word, _, feats), elem in zip(test_data, answer)]
                curr_metrics = evaluate(answer_to_evaluate, dev_data)
                curr_metrics = prettify_metrics(curr_metrics)
                metrics.append(((language, mode), curr_metrics[1]))
                print(curr_metrics[0], end="")
            # if inflector.use_lm:
            #     inflector.lm_.rebuild(test=False)
            #     lm_group_lengths = [0] + list(np.cumsum([len(predictions) for predictions in answer]))
            #     data_for_lm_scores = []
            #     for source, predictions in zip(test_data, answer):
            #         data_for_lm_scores.extend([(elem[0], source[2]) for elem in predictions])
            #     lm_scores = inflector.lm_.predict(data_for_lm_scores, return_letter_scores=True,
            #                                       return_log_probs=False)
            #     lm_scores_by_groups = [lm_scores[start:lm_group_lengths[i+1]]
            #                            for i, start in enumerate(lm_group_lengths[:-1])]
            #     with open("dump_1.out", "w", encoding="utf8") as fout:
            #         for source, predictions, curr_lm_scores in\
            #                 zip(test_data, answer, lm_scores_by_groups):
            #             predicted_words = [elem[0] for elem in predictions]
            #             for elem, (letter_scores, word_score) in zip(predictions, curr_lm_scores):
            #                 fout.write("\t".join([source[0], ";".join(source[2]), elem[0],
            #                                       "-".join("{:.2f}".format(100*x) for x in elem[1])]) + "\n")
            #                 word = list(elem[0]) + ['EOW']
            #                 fout.write(" ".join("{}-{:.2f}".format(x, 100*y)
            #                                     for x, y in zip(word, letter_scores)) + "\t")
            #                 fout.write("{:.2f}\n".format(word_score))

            # answers_for_missed = predict_missed_answers(test_data, answer, inflector, **params["predict"])
            # analysis_file = os.path.join(analysis_dir, filename+"-analysis") if analysis_dir is not None else None
            # output_analysis(test_data, answer, analysis_file,
            #                 answers_for_missed=answers_for_missed)
        if to_predict:
            predict_file = os.path.join(corr_dir, "{}-covered-test".format(language))
            data = read_infile(predict_file, feat_column=1)
            answer = inflector.predict(data, feat_column=-1, **params["predict"])
            outfile = os.path.join(predict_dir, filename+"-out")
            if not os.path.exists(predict_dir):
                continue
            with open(outfile, "w", encoding="utf8") as fout:
                for source, predictions in zip(data, answer):
                    predicted_words = [elem[0] for elem in predictions]
                    for word in predicted_words[:1]:
                        fout.write("\t".join([source[0], word, ";".join(source[-1])]) + "\n")
    if len(metrics) > 0:
        for elem in metrics:
            print("-".join(elem[0]), " ".join("{:.2f}".format(x) for x in elem[1]))
        if eval_outfile is not None:
            with open(eval_outfile, "w", encoding="utf8") as fout:
                for elem in metrics:
                    fout.write("{}{}\n".format("-".join(elem[0]),
                                               " ".join("{:.2f}".format(x) for x in elem[1])))


