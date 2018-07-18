import keras.layers as kl
import keras.backend as kb


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
