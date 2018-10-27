import os

from read import read_infile


def extract_ngrams(word, L):
    word = "^" + word + "$"
    return [word[start:start+L] for start in range(len(word) - L + 1)]

def analyze_ngrams(source_data, test_data, max_length=2, min_bad_count=None):
    if min_bad_count is None:
        min_bad_count = [1] * (max_length - 1)
    bad_word_count = [0] * (max_length - 1)
    total_ngram_count, bad_ngram_count = [0] * (max_length - 1), [0] * (max_length - 1)
    source_ngrams = [set() for L in range(2, max_length+1)]
    for lemma, word, _ in source_data:
        for i in range(2, max_length+1):
            source_ngrams[i-2].update(extract_ngrams(word, i))
    for lemma, word, _ in test_data:
        for i in range(2, max_length + 1):
            lemma_ngrams = extract_ngrams(lemma, i)
            word_ngrams = extract_ngrams(word, i)
            has_bad_ngrams = 0
            for ngram in word_ngrams:
                total_ngram_count[i-2] += 1
                if ngram not in source_ngrams[i-2] and ngram not in lemma_ngrams:
                    bad_ngram_count[i-2] += 1
                    has_bad_ngrams += 1
                    if has_bad_ngrams == min_bad_count[i-2]:
                        bad_word_count[i-2] += 1
                        # has_bad_ngrams = True
    for L in range(2, max_length + 1):
        print("{} bad {}-grams out of {}".format(bad_ngram_count[L-2], L, total_ngram_count[L-2]))
        print("{} words with bad {}-grams out of {}".format(bad_word_count[L-2], L, len(test_data)))

if __name__ == "__main__":
    languages = ["belarusian"]
    for language in languages:
        source_data = read_infile(os.path.join("conll2018", "task1", "all", "{}-train-low".format(language)))
        dev_data = read_infile(os.path.join("conll2018", "task1", "all", "{}-dev".format(language)))
        print(language)
        analyze_ngrams(source_data, dev_data, 4, [1, 2, 2])