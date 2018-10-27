import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from neural.neural_LM import NeuralLM, load_lm
from ngram.ngram import LabeledNgramModel
from paradigm_classifier import ParadigmLmClassifier

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
kbt.set_session(tf.Session(config=config))


def generate_auxiliary(data, dev_data, n, lm_file, outfile, min_label_count=5):
    forward_lm = load_lm(lm_file)
    splitted_path = os.path.split(lm_file)
    reverse_lm_file = "reverse-" + splitted_path[-1]
    reverse_lm = load_lm(os.path.join(*(splitted_path[:-1] + (reverse_lm_file,))))
    answer = augment_data(data, n, forward_lm, reverse_lm,
                          dev_data, min_label_count=min_label_count)
    with open(outfile, "w", encoding="utf8") as fout:
        for elem in answer:
            fout.write("{}\t{}\t{}\n".format(elem[0], elem[1], ";".join(elem[2])))
    return answer


def augment_data(data, n, forward_lm, reverse_lm, dev_data=None, min_label_count=5):
    data_for_lm = [[elem[0], elem[2][0]] for elem in data]
    good_bigrams = set()
    for _, word, _ in data:
        word = "^" + word + "$"
        for start in range(len(word) - 1):
            good_bigrams.add(word[start:start+2])
    labels_by_pos = defaultdict(dict)
    for _, _, label in data:
        labels_by_pos[label[0]]["_".join(label)] = min_label_count
    # labels_by_pos = {label: list(values) for label, values in labels_by_pos.items()}
    ngram_model = LabeledNgramModel(max_ngram_length=3, reverse=True).train(data_for_lm)
    auxiliary_source = ngram_model.generate_words(n=n * 5)
    data_for_augmentation = []
    cls = ParadigmLmClassifier(forward_lm=forward_lm, reverse_lm=reverse_lm)
    cls.train(data)
    # for descr, pattern_counts in cls.patterns.items():
    #     labels_by_pos[descr[0]]["_".join(descr)] += len(pattern_counts)
    for pos, elem in labels_by_pos.items():
        descrs = list(elem.keys())
        probs = np.fromiter(elem.values(), dtype=float)
        probs /= np.sum(probs)
        labels_by_pos[pos] = (descrs, probs)
    for word, label in auxiliary_source:
        descrs, probs = labels_by_pos[label]
        index = np.random.multinomial(1, probs).argmax()
        full_label = descrs[index]
        data_for_augmentation.append((word, full_label.split("_")))

    generated = cls.predict(data_for_augmentation, predict_no_forms=True)
    answer = []
    for (lemma, label), (elem, probs) in zip(data_for_augmentation, generated):
        if len(probs) > 0:
            for word, prob in zip(elem, probs):
                if prob < 0.25:
                    continue
                bigrams = [("^" + word + "$")[start:start+2] for start in range(len(word)+1)]
                if all(bigram in good_bigrams for bigram in bigrams):
                    answer.append((lemma, word, label))
            if len(answer) == n:
                break
    return answer
