import os
import numpy as np
import itertools

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from neural.neural_LM import NeuralLM
from read import read_infile

mode = "medium"


DEFAULT_LM_PARAMS = {"nepochs": 50, "batch_size": 16,
                     "history": 5, "use_feats": True, "use_label": True,
                     "encoder_rnn_size": 64, "decoder_rnn_size": 64, "dense_output_size": 32,
                     "decoder_dropout": 0.2, "encoder_dropout": 0.2,
                     "feature_embeddings_size": 32, "feature_embedding_layers": 1,
                     "use_embeddings": True, "embeddings_size": 32, "use_full_tags": True,
                     "callbacks":
                         {"EarlyStopping": { "patience": 5, "monitor": "val_loss"}}
                     }


KEYS = ["labels", "feats", "reverse", "bigrams"]

if __name__ == "__main__":
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.3
    kbt.set_session(tf.Session(config=config))
    languages = ["belarusian"]

    corr_dir = os.path.join("conll2018", "task1", "all")
    use_label, use_bigram_loss, reverse = True, True, True
    for language in languages:
        infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
        data = read_infile(infile, for_lm=True)
        dev_file = os.path.join(corr_dir, "{}-dev".format(language, mode))
        dev_data = read_infile(dev_file, for_lm=True)
        for (use_bigram_loss, use_feats) in itertools.product([False, True], [False, True]):
            model = NeuralLM(use_bigram_loss=use_bigram_loss, use_label=use_label,
                             use_feats=use_feats, nepochs=30, reverse=reverse)
            model.train(data, dev_data)
            answer = model.predict(dev_data, return_letter_scores=True)
            os.makedirs("dump", exist_ok=True)
            outfile = "probs"
            FLAGS = [use_label, use_feats, reverse, use_bigram_loss]
            for key, flag in zip(KEYS, FLAGS):
                if flag:
                    outfile += "_" + key
            total_score = 0.0
            with open("dump/{}-{}.out".format(mode, outfile), "w", encoding="utf8") as fout:
                for (word, tags), (letter_probs, score) in zip(dev_data, answer):
                    fout.write("{}\t{}\t{:.2f}\n".format(word, ";".join(tags), score))
                    word_to_print = (word + "$") if not reverse else ("^" + word)
                    fout.write(" ".join(["{}:{:.3f}".format(*elem) for elem in zip(word_to_print, letter_probs)]) + "\n")
                    total_score += score / len(word_to_print)
            print(FLAGS, "{:.3f}".format(score / len(answer)))



