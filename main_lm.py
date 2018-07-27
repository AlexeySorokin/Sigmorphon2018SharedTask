import sys
import os
import ujson as json
import getopt

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_infile
from inflector import load_inflector
from neural.neural_LM import NeuralLM, load_lm
from paradigm_classifier import ParadigmLmClassifier
from evaluate import evaluate, WIDTHS, get_format_string


cls_config = {"tune_weights": False, "use_paradigm_counts": False, "verbose": 0}
languages = ["spanish"]
modes = ["low"] * 1

SHORT_OPTS = "M:m:"

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=config))
    config_file = sys.argv[1]
    metrics = []
    use_model, model_dir, model_name = False, None, None
    opts, args = getopt.getopt(sys.argv[1:], SHORT_OPTS)
    for opt, val in opts:
        if opt == "-M":
            use_model, model_dir = True, val
        elif opt == "-m":
            model_name = val
    for language, mode in zip(languages, modes):
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
        cls = ParadigmLmClassifier(basic_model=basic_model, forward_lm=forward_lm,
                                   reverse_lm=reverse_lm, **cls_config)
        cls.train(data, dev_data, save_forward_lm=forward_save_file, save_reverse_lm=reverse_save_file)
        data_to_predict = [(x[0], x[2]) for x in dev_data]
        answer = cls.predict(data_to_predict, n=5)
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
    if len(languages) > 0 and cls.verbose:
        for language, curr_metrics in zip(languages, metrics):
            print(language, format_string.format(*curr_metrics), end="")