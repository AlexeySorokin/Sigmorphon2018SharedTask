import os
import numpy as np
from itertools import product

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_infile
from neural.neural_LM import NeuralLM
from paradigm_classifier import ParadigmChecker
from pyparadigm.paradigm_detector import ParadigmSubstitutor, constants_to_pattern

def extract_change(first, second):
    i = 0
    for i in range(min(len(first), len(second))):
        if first[i] != second[i]:
            break
    j = 0
    end = min(len(first), len(second)) - i
    for j in range(end):
        if first[-j-1] != second[-j-1]:
            break
    return (first[i:len(first)-j], second[i:len(second)-j])


def update_checker(checker, only_new_labels=True):
    change_patterns = set()
    for i, (label, label_patterns) in enumerate(checker.patterns.items()):
        label_patterns = [ParadigmSubstitutor(descr) for descr in label_patterns]
        for j, (other, other_patterns) in enumerate(checker.patterns.items()):
            other_patterns = [ParadigmSubstitutor(descr) for descr in other_patterns]
            if j == i:
                continue
            if len(label) != len(other):
                continue
            positions = [k for k, (x, y) in enumerate(zip(label, other)) if x != y]
            if len(positions) != 1 or positions[0] == 0:
                continue
            pos = positions[0]
            grammeme_pair = (label[pos], other[pos])
            for elem in product(label_patterns, other_patterns):
                first_pattern = elem[0].get_const_fragments()
                second_pattern = elem[1].get_const_fragments()
                if first_pattern[0] != second_pattern[0]:
                    continue
                change_positions = [k for k, (x, y) in enumerate(zip(first_pattern[1], second_pattern[1])) if x != y]
                if len(change_positions) != 1:
                    continue
                change_pos = change_positions[0]
                affix_pair = (first_pattern[1][change_pos], second_pattern[1][change_pos])
                change = extract_change(*affix_pair)
                if change is not None and change[0] != "":
                    # print("_".join(first_pattern[1]), "_".join(second_pattern[1]))
                    # print(elem[0].descr, elem[1].descr)
                    key = (label[0], len(label), positions[0], len(first_pattern[0]), change_pos) + grammeme_pair + change
                    change_patterns.add(key)
    to_update = set()
    for label, label_patterns in checker.patterns.items():
        label_patterns = [ParadigmSubstitutor(descr) for descr in label_patterns]
        for change_pattern in change_patterns:
            tag, length, pos, pattern_length, change_pos = change_pattern[:5]
            grammemes = change_pattern[5:7]
            if label[0] != tag or len(label) != length or label[pos] != grammemes[0]:
                continue
            new_label = label[:pos] + (grammemes[1],) + label[pos + 1:]
            for pattern in label_patterns:
                source, dest = pattern.get_const_fragments()
                if len(dest) != pattern_length:
                    continue
                upper, lower = change_pattern[-2:]
                if upper not in dest[change_pos]:
                    continue
                new_part = dest[change_pos].replace(upper, lower, 1)
                new_dest = dest[:change_pos] + [new_part] + dest[change_pos + 1:]
                new_descr = (constants_to_pattern(source), constants_to_pattern(new_dest))
                if new_label not in checker.patterns or\
                        (not only_new_labels and new_descr not in checker.patterns[new_label]):
                    to_update.add((new_label, new_descr))
    for label, descr in to_update:
        checker.patterns[label][descr] += 1
        checker.substitutors[descr] = ParadigmSubstitutor(descr)
    return checker

# if __name__ == "__main__":
#     config = tf.ConfigProto()
#     config.gpu_options.per_process_gpu_memory_fraction = 0.3
#     kbt.set_session(tf.Session(config=config))
#
#     languages = ["albanian"]
#     for language in languages:
#         print(language)
#         corr_dir = os.path.join("conll2018", "task1", "all")
#         infile = os.path.join(corr_dir, "{}-train-{}".format(language, "low"))
#         data = read_infile(infile)
#         dev_file = os.path.join(corr_dir, "{}-dev".format(language))
#         dev_data = read_infile(dev_file)
#
#         checker = ParadigmChecker().train(data)
#         change_patterns = set()
#         for i, (label, label_patterns) in enumerate(checker.patterns.items()):
#             label_patterns = [ParadigmSubstitutor(descr) for descr in label_patterns]
#             for j, (other, other_patterns) in enumerate(checker.patterns.items()):
#                 other_patterns = [ParadigmSubstitutor(descr) for descr in other_patterns]
#                 if j == i:
#                     continue
#                 if len(label) != len(other):
#                     continue
#                 positions = [k for k, (x, y) in enumerate(zip(label, other)) if x != y]
#                 if len(positions) != 1 or positions[0] == 0:
#                     continue
#                 pos = positions[0]
#                 grammeme_pair = (label[pos], other[pos])
#                 for elem in product(label_patterns, other_patterns):
#                     first_pattern = elem[0].get_const_fragments()
#                     second_pattern = elem[1].get_const_fragments()
#                     if first_pattern[0] != second_pattern[0]:
#                         continue
#                     change_positions = [k for k, (x, y) in enumerate(zip(first_pattern[1], second_pattern[1])) if x != y]
#                     if len(change_positions) != 1:
#                         continue
#                     change_pos = change_positions[0]
#                     affix_pair = (first_pattern[1][change_pos], second_pattern[1][change_pos])
#                     change = extract_change(*affix_pair)
#                     if change is not None and change[0] != "":
#                         # print("_".join(first_pattern[1]), "_".join(second_pattern[1]))
#                         # print(elem[0].descr, elem[1].descr)
#                         key = (label[0], len(label), positions[0], len(first_pattern), change_pos) + grammeme_pair + change
#                         change_patterns.add(key)
#         for label, label_patterns in checker.patterns.items():
#             label_patterns = [ParadigmSubstitutor(descr) for descr in label_patterns]
#             for change_pattern in change_patterns:
#                 tag, length, pos, pattern_length, change_pos = change_pattern[:5]
#                 grammemes = change_pattern[5:7]
#                 if label[0] != tag or len(label) != length or label[pos] != grammemes[0]:
#                     continue
#                 new_label = label[:pos] + (grammemes[1],) + label[pos+1:]
#                 for pattern in label_patterns:
#                     source, dest = pattern.get_const_fragments()
#                     if len(source) != pattern_length:
#                         continue
#                     upper, lower = change_pattern[-2:]
#                     if upper not in dest[change_pos]:
#                         continue
#                     new_part = dest[change_pos].replace(upper, lower, 1)
#                     new_dest = dest[:change_pos] + [new_part] + dest[change_pos+1:]
#                     new_descr = (constants_to_pattern(source), constants_to_pattern(new_dest))
#                     if new_label in checker.patterns and new_descr in checker.patterns[new_label]:
#                         print("Old", new_label, new_descr)
#                     elif new_label in checker.patterns:
#                         print("New", new_label, new_descr)
#                     else:
#                         print("new label", new_label, new_descr)
