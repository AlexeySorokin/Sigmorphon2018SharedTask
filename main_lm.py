import sys
import os
import ujson as json
import getopt

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_infile, read_languages_infile
from inflector import load_inflector
from neural.neural_LM import NeuralLM, load_lm
from paradigm_classifier import ParadigmLmClassifier
from evaluate import evaluate, WIDTHS, get_format_string


cls_config = {"use_paradigm_counts": False, "verbose": 0}
languages = ["kurmanji"]
modes = ["high"] * 1

SHORT_OPTS = "M:m:sgl:p:S:t"

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=config))
    # config_file = sys.argv[1]
    metrics = []
    use_model, model_dir, model_name, use_model_scores = False, None, None, True
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    language_file, test_data_dir, to_generate_patterns, tune_weights = None, "conll2018/task1/all", False, False
    predictions_dir, submissions_dir = None, None
    for opt, val in opts:
        if opt == "-M":
            use_model, model_dir = True, val
        elif opt == "-m":
            model_name = val
        elif opt == "-s":
            use_model_scores = False
        elif opt == "-g":
            to_generate_patterns = True
        elif opt == "-l":
            language_file = val
        elif opt == "-p":
            predictions_dir = val
        elif opt == "-S":
            submissions_dir = val
        elif opt == "-t":
            tune_weights = True
    if language_file is not None:
        languages = read_languages_infile(language_file)
    else:
        languages = list(zip(languages, modes))
    for language, mode in languages:
        input_dir = os.path.join("conll2018", "task1", "all")
        infile = os.path.join(input_dir, "{}-train-{}".format(language, mode))
        dev_file = os.path.join(input_dir, "{}-dev".format(language))
        if use_model:
            model_file = "{}-{}.json".format(language, mode)
            if model_name is not None:
                model_file = model_name + "-" + model_file
            model_file = os.path.join(model_dir, model_file)
        if not use_model or not os.path.exists(model_file):
            basic_model = None
        else:
            basic_model = load_inflector(model_file)
        data, dev_data = read_infile(infile), read_infile(dev_file)
        # with open(config_file, "r", encoding="utf8") as fin:
        #     params = json.load(fin)
        forward_save_file = "language_models/{}-{}.json".format(language, mode)
        forward_lm = load_lm(forward_save_file) if os.path.exists(forward_save_file) else None
        reverse_save_file = "language_models/reverse-{}-{}.json".format(language, mode)
        reverse_lm = load_lm(reverse_save_file) if os.path.exists(reverse_save_file) else None
        cls = ParadigmLmClassifier(basic_model=basic_model, use_basic_scores=use_model_scores,
                                   to_generate_patterns=to_generate_patterns,
                                   tune_weights=tune_weights,
                                   forward_lm=forward_lm, reverse_lm=reverse_lm, **cls_config)
        cls.train(data, dev_data, save_forward_lm=forward_save_file, save_reverse_lm=reverse_save_file)
        data_to_predict = [(x[0], x[2]) for x in dev_data]
        answer = cls.predict(data_to_predict, n=5)
        if predictions_dir is not None:
            outfile = os.path.join(predictions_dir, "{}-{}-out".format(language, mode))
            if not os.path.exists(predictions_dir):
                continue
            with open(outfile, "w", encoding="utf8") as fout:
                for source, predictions in zip(data, answer):
                    predicted_words = [elem[0] for elem in predictions]
                    fout.write("\t".join([source[0], "#".join(predicted_words), ";".join(source[-1])]) + "\n")
            print("Predicted for {}-{}".format(language, mode))
        answer_to_evaluate = [[word, elem[0], feats] for (word, feats), elem in zip(data_to_predict, answer)]
        curr_metrics = evaluate(answer_to_evaluate, dev_data)
        format_string, metrics_data = [], []
        for i, elem in enumerate(curr_metrics):
            width = WIDTHS[i] if i < len(WIDTHS) else None
            format_string.append(get_format_string(elem, width=width))
            if isinstance(elem, list):
                metrics_data.append(" ".join(str(x) for x in elem))
            elif isinstance(elem, dict):
                metrics_data.append(" ".join("{}:{:.2f}".format(*x) for x in sorted(elem.items())))
            else:
                metrics_data.append(elem)
        format_string = "\t".join(format_string) + "\n"
        metrics.append(metrics_data)
        if not cls.verbose:
            print(language, format_string.format(*metrics_data), end="")
        if submissions_dir is not None:
            predict_file = os.path.join(test_data_dir, "{}-covered-test".format(language))
            data = read_infile(predict_file, feat_column=1)
            answer = cls.predict(data, n=5)
            outfile = os.path.join(submissions_dir, "{}-{}-out".format(language, mode))
            if not os.path.exists(submissions_dir):
                continue
            with open(outfile, "w", encoding="utf8") as fout:
                for source, predictions in zip(data, answer):
                    predicted_words = [elem[0] for elem in predictions]
                    for word in predicted_words[:1]:
                        fout.write("\t".join([source[0], word, ";".join(source[-1])]) + "\n")
            print("Predicted for {}-{}".format(language, mode))
    if len(languages) > 0:
        for (language, mode), curr_metrics in zip(languages, metrics):
            print("{:<24}{:<6}".format(language, mode), format_string.format(*curr_metrics), end="")