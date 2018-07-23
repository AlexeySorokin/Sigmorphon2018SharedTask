import keras.layers as kl
import keras.backend as kb

if kb.backend() == "theano":
    from cells_theano import make_history_theano
elif kb.backend() == "tensorflow":
    from common_tensorflow import batch_shifted_fill


def MultiConv1D(inputs, layers, windows, filters, dropout=0.0, activation=None):
    if isinstance(windows, int):
        windows = [windows]
    if isinstance(filters, int):
        filters = [filters]
    if len(filters) == 1 and len(windows) > 1:
        filters = [filters] * len(windows)
    outputs = []
    for window_size, filters_number in zip(windows, filters):
        curr_output = kl.Conv1D(filters_number, window_size,
                                padding="same", activation=activation)(inputs)
        for i in range(layers - 1):
            if dropout > 0.0:
                curr_output = kl.Dropout(dropout)(curr_output)
            curr_output = kl.Conv1D(filters_number, window_size,
                                    padding="same", activation=activation)(curr_output)
        outputs.append(curr_output)
    answer = outputs[0] if (len(outputs) == 1) else kl.Concatenate()(outputs)
    return answer


def calculate_history_shape(shape, h, flatten, only_last=False):
    if len(shape) == 2 or not flatten:
        shape = shape[:2] + (h,) + shape[2:]
    elif shape[2] is not None:
        shape = shape[:2] + (h * shape[2],) + shape[3:]
    else:
        shape = shape[:2] + (None,) + shape[3:]
    if only_last and shape[1] is not None:
        shape = (shape[0], 1) + shape[2:]
    return shape


def make_history(X, h, r, pad, right_pad=None,flatten=False,
                 only_last=False, calculate_keras_shape=False):
    if kb.backend() == "theano":
        answer = make_history_theano(X, h, pad, flatten=flatten)
    else:
        answer = batch_shifted_fill(X, h, pad, r=r, right_pad=right_pad, flatten=flatten)
    if only_last:
        # answer = answer[:,-1:]
        answer = answer[:,-1:]
    if calculate_keras_shape:
        if not hasattr(answer, "_keras_shape") and hasattr(X, "_keras_shape"):
            answer._keras_shape = calculate_history_shape(
                X._keras_shape, h+r, flatten, only_last=only_last)
    return answer


def History(X, h, r=0, flatten=False, only_last=False):
    """
    For each timestep collects h previous elements of the tensor, including current

    X: a Keras tensor, at least 2D, of shape (B, L, ...)
    h: int, history length
    flatten: bool, default=False,
        whether to concatenate h previous elements of the tensor (flatten=True),
        or stack then using a new dimension (flatten=False)
    """
    pad, right_pad = kb.zeros_like(X[0][0]), kb.zeros_like(X[0][0])
    arguments = {"h": h, "r": r, "pad": pad, "right_pad": right_pad,
                 "flatten": flatten, "only_last": only_last}
    output_shape = lambda x: calculate_history_shape(x, h+r, flatten, only_last=only_last)
    return kl.Lambda(make_history, arguments=arguments, output_shape=output_shape)(X)

def TemporalDropout(inputs, dropout=0.0):
    """
    Drops with :dropout probability temporal steps of input 3D tensor
    """
    # TO DO: adapt for >3D tensors
    if dropout == 0.0:
        return inputs
    inputs_func = lambda x: kb.ones_like(inputs[:, :, 0:1])
    inputs_mask = kl.Lambda(inputs_func)(inputs)
    inputs_mask = kl.Dropout(dropout)(inputs_mask)
    tiling_shape = [1, 1, kb.shape(inputs)[2]] + [1] * (kb.ndim(inputs) - 3)
    inputs_mask = kl.Lambda(kb.tile, arguments={"n": tiling_shape},
                            output_shape=inputs._keras_shape[1:])(inputs_mask)
    answer = kl.Multiply()([inputs, inputs_mask])
    return answer

def local_dot_attention(values, queries, keys, normalize_logits=True):
    input_length = kb.shape(queries)[1]
    queries = kb.reshape(queries, (-1, queries.shape[2]))
    keys = kb.reshape(keys, (-1, keys.shape[-2], keys.shape[-1]))
    values = kb.reshape(values, (-1, values.shape[-2], values.shape[-1]))
    logits = kb.batch_dot(keys, queries, axes=[2, 1])
    if normalize_logits:
        logits /= kb.sqrt(kb.cast(kb.shape(keys)[-1], "float32"))
    scores = kb.softmax(logits)
    tiled_scores = kb.tile(scores[:,:,None], [1, 1, kb.shape(values)[-1]])
    answer = kb.sum(tiled_scores * values, axis=1)
    answer = kb.reshape(answer, (-1, input_length, answer.shape[-1]))
    return answer

def LocalAttention(inputs, keys_size, values_size, h, r, activation=None):
    queries = kl.Dense(keys_size, activation=activation)(inputs)
    keys = kl.Dense(keys_size, activation=activation)(inputs)
    window_keys = History(keys, h, r, flatten=False)
    values = kl.Dense(values_size, activation=activation)(inputs)
    window_values = History(values, h, r, flatten=False)
    # answer = local_dot_attention(queries, window_keys, window_values)
    # output_shape = lambda x: x[:-1] + (values_size,)
    return kl.Lambda(local_dot_attention, arguments={"queries": queries, "keys": window_keys})(window_values)
