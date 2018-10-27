import os

import tensorflow as tf
import keras.backend.tensorflow_backend as kbt

from read import read_languages_infile, read_infile
from neural.neural_LM import NeuralLM
from ngram.ngram import LabeledNgramModel

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.3
kbt.set_session(tf.Session(config=config))

LM_PARAMS = {"nepochs": 20, "batch_size": 16, "reverse": True,
             "history": 5, "use_feats": False, "use_label": True,
             "encoder_rnn_size": 64, "decoder_rnn_size": 64, "dense_output_size": 32,
             "decoder_dropout": 0.2, "encoder_dropout": 0.2, "decoder_layers": 1,
             "feature_embeddings_size": 32, "feature_embedding_layers": 1,
             "use_embeddings": False, "embeddings_size": 32, "use_full_tags": False,
             "callbacks":
                 {"EarlyStopping": { "patience": 5, "monitor": "val_loss"}}
             }

if __name__ == "__main__":
    language, mode = "belarusian", "low"
    corr_dir = os.path.join("conll2018", "task1", "all")
    infile = os.path.join(corr_dir, "{}-train-{}".format(language, mode))
    dev_file = os.path.join(corr_dir, "{}-dev".format(language, mode))
    data, dev_data = read_infile(infile), read_infile(dev_file)

    ngram_model = LabeledNgramModel(max_ngram_length=3, reverse=True)
    auxiliary_source = ngram_model.generate_words(n=1000)


    data_for_lm = [[elem[0], elem[2][:1]] for elem in data]
    dev_data_for_lm = [[elem[0], elem[2][:1]] for elem in dev_data]

    lemma_lm = NeuralLM(**LM_PARAMS)
    lemma_lm.train(data_for_lm, dev_data_for_lm, save_file="language_models/lm_lemmas/belarusian-low")
    # answer = lemma_lm.predict(dev_data_for_lm, return_letter_scores=True)
    # print(answer[0])
    for elem in lemma_lm.sample(32, batch_size=4):
        print("\t".join(["".join(elem[0][-4::-1]), "_".join(elem[1])]))