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
