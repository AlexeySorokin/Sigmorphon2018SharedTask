from collections import  defaultdict


def output_word_predictions(corr_data, curr_answers, fout, missed_answer=None,
                            has_component_probs=False, has_alignment=False, end=""):
    lemma, word, descr = corr_data[:3]
    corr_index, corr_prob = len(curr_answers)-1, 0.0
    for i, elem in enumerate(curr_answers):
        if word == elem[0]:
            corr_index, corr_prob = i, elem[2]
            break
    fout.write("{}\t{}\t{}\t{:.2f}\n{}\n".format(
        lemma, ";".join(descr), word, 100 * corr_prob, "-" * 40))
    if missed_answer is not None:
        symbols = list(word) + ["$"]
        fout.write(",".join("{}-{:.2f}".format(a, 100*x)
                            for a, x in zip(symbols, missed_answer[1][1:])) + "\n")
        if has_component_probs:
            fout.write(",".join("{}-{:.2f}".format(a, 100*x)
                       for a, x in zip(symbols, missed_answer[3][1:])) + "\n")
            fout.write(",".join("{}-{:.2f}".format(a, 100*x)
                       for a, x in zip(symbols, missed_answer[4][1:])))
            if len(missed_answer) > 5:
                fout.write("\t{:.2f}".format(missed_answer[5]))
            fout.write("\n")
        fout.write("-" * 40 + "\n")
    for elem in curr_answers[:corr_index+1]:
        curr_word, letter_probs, prob = elem[:3]
        # alignment = elem[3] if has_alignment else None
        symbols = list(curr_word) + ["$"]
        fout.write("{0}\t{2:.1f}\n{1}\n".format(
            curr_word, ",".join("{}-{:.1f}".format(a, 100*x)
                                for a, x in zip(symbols, letter_probs[1:])), 100*prob))
        if has_component_probs:
            fout.write(",".join(
                "{}-{:.1f}".format(a, 100*x) for a, x in zip(symbols, elem[3][1:])) + "\n")
            fout.write(",".join(
                "{}-{:.1f}".format(a, 100*x) for a, x in zip(symbols, elem[4][1:])))
            if len(elem) > 5:
                fout.write("\t{:.1f}".format(elem[5]))
            fout.write("\n")
        # if alignment is not None:
        #     fout.write(make_alignment_string(lemma, alignment, curr_word) + "\n")
    fout.write(end+"\n")


def output_analysis(test_data, answers, outfile, has_alignment=False,
                    answers_for_missed=None, has_component_probs=False):
    if answers_for_missed is not None:
        answers_for_missed = dict(answers_for_missed)
    indexes_by_descrs = defaultdict(list)
    counts_by_descrs = defaultdict(lambda: [0,0])
    for i, ((lemma, word, descr), curr_answers) in enumerate(zip(test_data, answers)):
        descr = ";".join(descr)
        counts_by_descrs[descr][int(word == curr_answers[0][0])] += 1
        indexes_by_descrs[descr].append(i)
    with open(outfile, "w", encoding="utf8") as fout:
        for descr, (false_count, corr_count) in sorted(
                counts_by_descrs.items(), key=(lambda x: x[1][1] / (x[1][0] + x[1][1]))):
            quality = corr_count / (corr_count+false_count)
            fout.write("{}\tПравильно: {}, Всего: {}, Качество: {:.2f}\n{}\n".format(
                ",".join("{}={}".format(*x) for x in zip(*descr)),
                corr_count, corr_count+false_count, 100*quality, "="*40+"\n"))
            for index in indexes_by_descrs[descr]:
                output_word_predictions(test_data[index], answers[index], fout,
                                        missed_answer=answers_for_missed.get(index),
                                        has_component_probs=has_component_probs,
                                        has_alignment=has_alignment, end="-"*40)
            fout.write("\n")