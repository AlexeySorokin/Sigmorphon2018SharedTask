import keras.backend as kb
import tensorflow as tf


def gather_indexes(x):
    """
    Returns a tensor C such that C[i, j] = A[i, B[i, j]]
    :param A:
    :param B:
    :return:
    """
    A, B = x
    first_dim_indexes = kb.expand_dims(tf.range(tf.shape(B)[0]), -1)
    first_dim_indexes = kb.tile(first_dim_indexes, [1, tf.shape(B)[1]])
    indexes = tf.stack([first_dim_indexes, B], axis=-1)
    return tf.gather_nd(A, indexes)


##### Constructing histories #####

def _broadcast_pad(pad, a, reps):
    """
    Broadcasts padding tensor before attaching it to a

    pad: a tensor to attach, must have 0, 1 or 2 less dimensions than a,
    a: a tensor to be padded
    reps: the number of repetitions of pad along second (timestep) axis

    pad: a broadcasted tensor
    """
    if a.shape.ndims - pad.shape.ndims == 2:
        pad, batch_tile = pad[None, None,:], tf.shape(a)[0]
    elif a.shape.ndims - pad.shape.ndims == 1:
        pad, batch_tile = pad[:,None], 1
    elif pad.shape.ndims == a.shape.ndims:
        batch_tile = 1
    else:
        raise ValueError("pad.ndim must be in [a.ndim, a.ndim-1, a.ndim-2]")
    pad = tf.tile(pad, [batch_tile, reps] + [1]*(a.shape.ndims - 2))
    return pad


def batch_shifted_fill(a, h, pad, r=0, right_pad=None, flatten=False):
    """

    a: array-like or tf.tensor
    h: int, history length
    pad: array-like or tf.tensor, padding value for a elements
    flatten: boolean, default=False
        whether to flatten histories. In case or a.dim >= 3 individual elements of a are arrays,
        therefore histories are sequences of arrays. By default they are not flattened.

    :returns
    answer_: array-like or tf.tensor
    """
    a = tf.convert_to_tensor(a)
    pad = tf.convert_to_tensor(pad)
    if r > 0:
        if right_pad is None:
            raise ValueError("right_pad cannot be None in case right padding is active (right_pad > 0)")
        right_pad = tf.convert_to_tensor(right_pad)
    pad = _broadcast_pad(pad, a, h-1)
    a_padded = tf.concat([pad, a], axis=1)
    if r > 0:
        right_pad = _broadcast_pad(right_pad, a, r)
        a_padded = tf.concat([a_padded, right_pad], axis=1)
    answer_shape = tf.concat([[tf.shape(a)[0], 0, h+r], tf.cast(a.shape[2:], tf.int32)], axis=0)
    i, answer = h-1+r, tf.zeros(answer_shape, dtype=a.dtype)
    cond = lambda i, ans, a, k: i < tf.shape(a_padded)[1]
    body = lambda i, ans, a, k: (
        i+1, tf.concat([ans, tf.expand_dims(a_padded[:,i-k+1:i+1], 1)], axis=1), a, k)
    ans_shape = [None, None, h+r] + a_padded.shape[2:].as_list()
    _, answer_, _, _ = tf.while_loop(
        cond, body, [i, answer, a_padded, h+r],
        shape_invariants=[tf.TensorShape([]), tf.TensorShape(ans_shape),
                          a_padded.shape, tf.TensorShape([])])
    if flatten and len(ans_shape) >= 4:
        outer_shape = tf.shape(answer_)
        elem_shape = [ans_shape[2] * ans_shape[3]] + ans_shape[4:]
        new_shape = tf.concat([outer_shape[:2], elem_shape], axis=0)
        answer_ = tf.reshape(answer_, new_shape)
    return answer_
