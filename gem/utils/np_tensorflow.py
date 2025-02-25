import tensorflow as tf

def dot(left,right):
    return tf.matmul(tf.convert_to_tensor(left, dtype=tf.float32),tf.convert_to_tensor(right, dtype=tf.float32)).numpy()

def linalg_norm(matrix):
    return tf.norm(tf.convert_to_tensor(matrix, dtype=tf.float32)).numpy()

def mean(matrix):
    return tf.reduce_mean(tf.convert_to_tensor(matrix, dtype=tf.float32)).numpy()

def std(matrix):
    return tf.reduce_std(tf.convert_to_tensor(matrix, dtype=tf.float32)).numpy()

def tf_sum(matrix,axis=None):
    return tf.reduce_sum(tf.convert_to_tensor(matrix, dtype=tf.float32),axis=axis).numpy()

import numpy as np
np.linalg.norm = linalg_norm
np.mean = mean
np.std = std
np.sum = tf_sum