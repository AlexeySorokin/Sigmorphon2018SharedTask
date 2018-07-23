import numpy as np

# import theano.tensor as tT
# import theano

import keras.backend as kb
import keras.layers as kl
from keras.engine import Layer, Model
from keras.engine.topology import InputSpec
from keras.initializers import Constant
from keras.callbacks import Callback, ProgbarLogger


class BasicMetricsProgbarLogger(ProgbarLogger):

    BASIC_METRICS = ["loss", "outputs_loss", "acc", "outputs_acc", "val_morphemes_acc"]

    def __init__(self, verbose, count_mode='steps'):
        # Ignore the `verbose` argument specified in `fit()` and pass `count_mode` upstream
        self.verbose = verbose
        super(BasicMetricsProgbarLogger, self).__init__(count_mode)

    def on_train_begin(self, logs=None):
        # filter out the training metrics
        self.params['metrics'] = [m for m in self.params['metrics']
                                  if (m in self.BASIC_METRICS or
                                      (m[:4] == "val_" and m[4:] in self.BASIC_METRICS))]
        self.epochs = self.params['epochs']


class ClassCrossEntropy:

    def __init__(self, labels, weights):
        self.labels = labels
        self.weights = weights

    def __call__(self, y_true, y_pred):
        loss = kb.categorical_crossentropy(y_true, y_pred)
        for label, weight in zip(self.labels, self.weights):
            curr_y_true = y_true[...,label]
            curr_y_pred = y_pred[...,label]
            loss += weight * kb.binary_crossentropy(curr_y_true, curr_y_pred)
        return loss


def gated_sum(X, disable_first=False):
    first, second, sigma = X
    tiling_shape = [kb.shape(first)[i] for i in range(kb.ndim(first))]
    for i in range(kb.ndim(sigma)-1):
        tiling_shape[i] = 1
    while kb.ndim(sigma) < kb.ndim(first):
        sigma = kb.expand_dims(sigma, -1)
    if disable_first:
        second *= (1.0 - first)
    answer = first * sigma + second * (1.0 - sigma)
    return answer

## routines for building layers

def flattened_one_hot(x, num_classes):
    one_hot_x = kb.one_hot(x, num_classes)
    answer = kb.batch_flatten(one_hot_x)
    return answer

# def probs_func_theano():
#     probs, proposal = tT.dvector(), tT.dvector()
#     sigma, threshold = tT.dscalar(), tT.dscalar()
#     prop = tT.where(probs >= threshold,  proposal, 0.0)
#     indices = prop.nonzero()
#     answer = tT.zeros_like(probs)
#     answer = tT.set_subtensor(
#         answer[indices], tT.exp(sigma * tT.log(probs[indices]) +
#                                 (1.0 - sigma) * tT.log(proposal[indices])))
#     answer /= tT.sum(answer)
#     return theano.function([probs, proposal, sigma, threshold], [answer], on_unused_input='ignore')

def weighted_sum(first, second, sigma, first_threshold=-np.inf, second_threshold=np.inf):
    first_normalized = first - kb.logsumexp(first, axis=-1)[...,None]
    second_normalized = second - kb.logsumexp(second, axis=-1)[...,None]
    # sigma.shape = (1,), first_normalized.shape = (T1, ..., Tm, d)
    # logit_probs.shape = (T1, ..., Tm, d)
    logit_probs = first_normalized * sigma + second_normalized * (1.0 - sigma)
    # logit_probs = kb.batch_dot(first_normalized, sigma) + kb.batch_dot(second_normalized, 1.0 - sigma)
    first_mask = (first_normalized < first_threshold).nonzero()
    logit_probs = kb.T.set_subtensor(logit_probs[first_mask], -np.inf)
    second_mask = (second_normalized < second_threshold).nonzero()
    logit_probs = kb.T.set_subtensor(logit_probs[second_mask], -np.inf)
    return logit_probs

def weighted_combination(first, second, features,
                         first_threshold=None, second_threshold=None,
                         from_logits=False, return_logits=False):
    """
    Комбинация базовых (first) и вспомогательных (second) экспоненциальных вероятностей
    :param probs:
    :param proposal:
    :param threshold:
    :return:
    """
    if not from_logits:
        first_shape = first._keras_shape
        first = kb.clip(first, 1e-10, 1.0)
        second = kb.clip(second, 1e-10, 1.0)
        first_, second_ = kb.log(first), kb.log(second)
        first_._keras_shape = second_._keras_shape = first_shape
    else:
        first_, second_ = first, second
    # sigma = kl.Dense(1, activation="sigmoid",
    #                  kernel_initializer='random_uniform', bias_initializer='ones')(features)
    # sigma = sigma[...,0]
    sigma = 1.0 * kb.ones_like(first, dtype="float64")
    while sigma.ndim < first.ndim:
        sigma = kb.expand_dims(sigma, axis=-1)
    first_threshold = (np.log(first_threshold)
                       if first_threshold is not None else -np.inf)
    second_threshold = (np.log(second_threshold)
                        if second_threshold is not None else -np.inf)
    result = weighted_sum(first_, second_, sigma, first_threshold, second_threshold)
    if not return_logits:
        result = kb.exp(result)
        # result = kb.T.set_subtensor(result[0], result[0]-r[...,None])
        # result = kb.T.set_subtensor(result[0], result[0]+r[...,None])
        result /= kb.sum(result, axis=-1)[...,None]
        result *= kb.sum(first, axis=-1)[...,None]
        # r = sigma[:,0,0]
        # r = kb.print_tensor(r)
    return result


# class WeightsCallback(Callback):
#
#     def __init__(self, model, symbols=None, dumpfile=None, predictions_file=None):
#         self.model = model
#         self.symbols = symbols
#         self.dumpfile = dumpfile
#         self.group_bounds = None
#         self.vectors = []
#         self.predictions_file = predictions_file
#
#     def on_train_begin(self, logs=None):
#         self.weights = []
#
#     def on_epoch_begin(self, epoch, logs=None):
#         self.weights.append([])
#
#     def on_batch_end(self, batch, logs=None):
#         self.weights[-1].append(self.model.get_weights())
#
#     def on_epoch_end(self, epoch, logs=None):
#         print("\r", end="")
#         last_weights = self.weights[-1][-1]
#         weights, bias = last_weights[0][:,0], last_weights[1][0]
#         if self.group_bounds is None:
#             self.group_bounds = [0, weights.shape[0]]
#         if epoch < 2 or epoch == self.params['epochs']-1:
#             with open(self.dumpfile, "a" if epoch > 0 else "w", encoding="utf8") as fout:
#                 fout.write("Epoch {}\n".format(epoch))
#                 for i, start in enumerate(self.group_bounds[:-1]):
#                     end = self.group_bounds[i+1]
#                     fout.write(",".join("{:.2f}".format(x) for x in weights[start:end]) + ";")
#                 fout.write("{:.2f}\n".format(bias))
#                 for key, indices in self.vectors:
#                     key = ",".join("=".join(elem) for elem in zip(*key))
#                     scores = self._make_scores(weights, bias, indices)
#                     fout.write("{}\t{}\n".format(key, self._print_scores(scores)))
#                 if hasattr(self, "validation_data") and self.predictions_file is not None:
#                     outfile = "{}_{}.out".format(self.predictions_file, epoch)
#                     self._output_analysis(list(self.validation_data), outfile)
#         # print("\n")
#
#     def _make_scores(self, weights, bias, indices):
#         indices = np.nonzero(indices)
#         basic_score = np.sum(weights[indices]) + bias
#         scores = np.array([basic_score])
#         for i, start in enumerate(self.group_bounds[1:-1], 1):
#             end = self.group_bounds[i+1]
#             if end == start + 1:
#                 # current feature in active or not
#                 scores = np.array([scores, scores + weights[start]])
#             elif end > start + 1:
#                 # exactly one of group features is active
#                 scores = np.array([scores + weights[j] for j in range(start, end)])
#         scores = kb.sigmoid(scores).eval()
#         if scores.ndim > 1:
#             scores = np.transpose(scores, list(range(scores.ndim))[::-1])
#         return scores
#
#     def _print_scores(self, scores):
#         scores = np.atleast_2d(scores)
#         return ";".join(",".join("{:.2f}".format(x) for x in elem) for elem in scores)
#
#     def set_group_bounds(self, group_bounds):
#         self.group_bounds = group_bounds
#
#     def set_vectors(self, vectors):
#         self.vectors = vectors
#
#     def _output_analysis(self, data, outfile):
#         data, one_hot_answers = [x[0] for x in data], [x[1] for x in data]
#         predicted_probs = [self.model.predict(bucket) for bucket in data]
#         predictions = [np.argmax(probs, axis=-1) for probs in predicted_probs]
#         answers = [None] * len(one_hot_answers)
#         for i, elem in enumerate(one_hot_answers):
#             k, m = elem.shape[:2]
#             nonzero_columns = np.nonzero(elem)[-1]
#             answers[i] = nonzero_columns.reshape((k, m))
#         analysis = []
#         for elem in zip(data, predicted_probs, predictions, answers):
#             analysis.extend(self._extract_analysis(*elem))
#         with open(outfile, "w", encoding="utf8") as fout:
#             for corr, predicted, data in analysis:
#                 fout.write("{}\t{}".format(corr, predicted))
#                 newline = True
#                 for symbol_data, error_data in data:
#                     if len(error_data) == 0:
#                         fout.write("\n" if newline else "\t")
#                         fout.write("{} {:.1f},{:.1f},{:.1f}".format(
#                             symbol_data[0], 100*symbol_data[1], 100*symbol_data[2], 100*symbol_data[3]))
#                         newline = False
#                     else:
#                         fout.write("\n")
#                         fout.write("{} {:.1f},{:.1f},{:.1f}".format(
#                             symbol_data[0], 100*symbol_data[1], 100*symbol_data[2], 100*symbol_data[3]))
#                         for elem in error_data:
#                             fout.write("\t{} {:.1f},{:.1f},{:.1f}".format(
#                                 elem[0], 100*elem[1], 100*elem[2], 100*elem[3]))
#                         newline = True
#                 fout.write("\n\n")
#         return
#
#     def _extract_analysis(self, source, probs, predicted, corr):
#         base_probs, lm_probs = source[:2]
#         answer = []
#         for base_word_probs, lm_word_probs, word_probs, letters, corr_letters in\
#                 zip(base_probs, lm_probs, probs, predicted, corr):
#             i, j = 1, 1 # positions in letters and corr_letters
#             has_error, word_data = False, []
#             corr_word, predicted_word = "", ""
#             while i < len(letters) and j < len(corr_letters):
#                 letter, corr_letter = letters[i], corr_letters[j]
#                 curr_probs, curr_lm_probs = word_probs[i], lm_word_probs[i]
#                 if letter == END and corr_letter == END:
#                     break
#                 if corr_letter == STEP_CODE and letter == STEP_CODE:
#                     i, j = i+1, j+1
#                     continue
#                 prob, base_prob = curr_probs[corr_letter], base_word_probs[i, corr_letter]
#                 lm_letter = corr_letter - int(corr_letter > STEP_CODE)
#                 lm_prob = curr_lm_probs[lm_letter] if corr_letter != STEP_CODE else 0.0
#                 symbol_data = (self.symbols[corr_letter], prob, base_prob, lm_prob)
#                 if letter != corr_letter:
#                     symbol_indexes = np.where(curr_probs > curr_probs[corr_letter])[0]
#                     symbol_indexes = sorted(
#                         symbol_indexes, key=(lambda x: curr_probs[x]), reverse=True)[:5]
#                     candidate_lm_probs = [(curr_lm_probs[x - int(x > STEP_CODE)]
#                                            if x != STEP_CODE else 0.0) for x in symbol_indexes]
#                     error_data = [(self.symbols[x], curr_probs[x], base_word_probs[i,x], p)
#                                   for x, p in zip(symbol_indexes, candidate_lm_probs)]
#                 else:
#                     error_data = []
#                 word_data.append((symbol_data, error_data))
#                 i += int(letter != STEP_CODE)
#                 j += int(corr_letter != STEP_CODE)
#                 has_error |= (letter != corr_letter)
#                 if letter > STEP_CODE:
#                     predicted_word += self.symbols[letter]
#                 if corr_letter > STEP_CODE:
#                     corr_word += self.symbols[corr_letter]
#             if has_error:
#                 answer.append((corr_word, predicted_word, word_data))
#         return answer



class WeightedCombinationLayer(Layer):

    """
    A class for weighted combination of probability distributions
    """

    def __init__(self, input_dim, features_dim, first_threshold=None,
                 second_threshold=None, from_logits=False, return_logits=False,
                 bias_initializer=1.0, **kwargs):
        # if 'input_shape' not in kwargs:
        #     kwargs['input_shape'] = [(None, input_dim,), (None, input_dim)]
        super(WeightedCombinationLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.features_dim = features_dim
        self.first_threshold = first_threshold
        self.second_threshold = second_threshold
        self.from_logits = from_logits
        self.return_logits = return_logits
        self.bias_initializer = bias_initializer
        self.input_spec = [InputSpec(axes={-1:input_dim}),
                           InputSpec(axes={-1:input_dim}),
                           InputSpec(axes={-1:features_dim})]

    def build(self, input_shape):
        assert len(input_shape) == 3
        assert input_shape[0] == input_shape[1]
        input_dim, features_dim = input_shape[0][-1], input_shape[2][-1]
        self.features_kernel = self.add_weight(
            shape=(features_dim, 1), initializer="random_uniform", name='kernel')
        self.features_bias = self.add_weight(
            shape=(1,), initializer=Constant(self.bias_initializer), name='bias')
        super(WeightedCombinationLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        # symbols = inputs[0], encodings = input[1]
        assert isinstance(inputs, list) and len(inputs) == 3
        first, second, features = inputs[0], inputs[1], inputs[2]
        if not self.from_logits:
            first_shape = first._keras_shape
            first = kb.clip(first, 1e-10, 1.0)
            second = kb.clip(second, 1e-10, 1.0)
            first_, second_ = kb.log(first), kb.log(second)
            first_._keras_shape = second_._keras_shape = first_shape
        else:
            first_, second_ = first, second
        # contexts.shape = (M, T, left)
        embedded_features = kb.dot(features, self.features_kernel)
        bias = self.features_bias
        while bias.ndim < embedded_features.ndim:
            bias = kb.expand_dims(bias, axis=0)
        sigma = kb.sigmoid(embedded_features+bias)[...,0]
        while sigma.ndim < first.ndim:
            sigma = kb.expand_dims(sigma, axis=-1)
        # sigma = kb.ones_like(first, dtype="float64") - 0.00001 * sigma
        first_threshold = (np.log(self.first_threshold)
                           if self.first_threshold is not None else -np.inf)
        second_threshold = (np.log(self.second_threshold)
                            if self.second_threshold is not None else -np.inf)
        result = weighted_sum(first_, second_, sigma, first_threshold, second_threshold)
        probs = kb.exp(result)
        probs /= kb.sum(probs, axis=-1)[...,None]
        probs *= kb.sum(first, axis=-1)[...,None]
        if self.return_logits:
            return [probs, result]
        return probs

    def compute_output_shape(self, input_shape):
        first_shape = input_shape[0]
        if self.return_logits:
            return [first_shape, first_shape]
        return first_shape


if __name__ == "__main__":
    probs = np.array([0.1, 0.5, 0.2, 0.2])
    proposal = np.array([0.4, 0.1, 0.3, 0.2])
    sigma, threshold = 0.5, 0.15
    func = probs_func_theano()
    print(func(probs, proposal, sigma, threshold))




