import os
from collections import defaultdict

import numpy as np
import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from neural.neural_LM import NeuralLM, load_lm
from augmentation.ngram import LabeledNgramModel
from augmentation.paradigm_augmentation import update_checker
from paradigm_classifier import ParadigmLmClassifier

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
kbt.set_session(tf.Session(config=config))


def select_ngram_model(data, dev_data, min_ngram_length, max_ngram_length, reverse):
    min_ngram_length, max_ngram_length = min(min_ngram_length, max_ngram_length), max(min_ngram_length, max_ngram_length)
    if reverse is None:
        reverse = [False, True]
    else:
        reverse = [reverse]
    best_L, best_reverse, best_score = min_ngram_length, reverse[0], np.inf
    data_for_lm = [[elem[0], elem[2][0]] for elem in data]
    dev_data_for_lm = [[elem[0], elem[2][0]] for elem in dev_data]
    for L in range(min_ngram_length, max_ngram_length):
        for curr_reverse in reverse:
            model = LabeledNgramModel(
                all_ngram_length=L, max_ngram_length=L, reverse=curr_reverse).train(data_for_lm)
            scores = []
            for word, label in dev_data_for_lm:
                score, letter_probs = model.score(word, label, return_letter_probs=True)
                scores.extend(-np.log(elem[1]) for elem in letter_probs)
        # print(word, "{:.3f}".format(score))
        # print(" ".join("{}:{:.3f}".format(*elem) for elem in letter_probs))
            score = np.mean(scores)
            print(L, curr_reverse, "{:.3f}".format(score))
            if score < best_score:
                best_L, best_reverse, best_score = L, curr_reverse, score
                best_model = model
    print("Best: ", best_L, best_reverse, "{:.3f}".format(best_score))
    return model                                     
            
    

def generate_auxiliary(data, dev_data, n, lm_file, outfile,
                       min_label_count=5, to_update_checker=False,
                       min_ngram_length=3, max_ngram_length=3, 
                       reverse=None, to_save=True):
    forward_lm = load_lm(lm_file)
    splitted_path = os.path.split(lm_file)
    reverse_lm_file = "reverse-" + splitted_path[-1]
    reverse_lm = load_lm(os.path.join(*(splitted_path[:-1] + (reverse_lm_file,))))
    model = select_ngram_model(data, dev_data, min_ngram_length, max_ngram_length, reverse)
    answer = augment_data(model, data, n, forward_lm, reverse_lm,
                          dev_data, min_label_count=min_label_count,
                          to_update_checker=to_update_checker)
    if to_save:
        with open(outfile, "w", encoding="utf8") as fout:
            for elem in answer:
                fout.write("{}\t{}\t{}\n".format(elem[0], elem[1], ";".join(elem[2])))
    return answer


def augment_data(model, data, n, forward_lm, reverse_lm, dev_data=None,
                 min_label_count=5, to_update_checker=False):
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
    # ngram_model = LabeledNgramModel(max_ngram_length=3, reverse=True).train(data_for_lm)
    auxiliary_source = model.generate_words(n=n * 5)
    data_for_augmentation = []
    cls = ParadigmLmClassifier(forward_lm=forward_lm, reverse_lm=reverse_lm)
    cls.train(data)
    if to_update_checker:
        update_checker(cls)
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
                    break
            if len(answer) >= n:
                break
    return answer
