import tensorflow as tf

def dot(left,right):
    l = tf.convert_to_tensor(left, dtype=tf.float32)
    r = tf.convert_to_tensor(right, dtype=tf.float32)
    return tf.matmul(l,r)

def linalg_norm(matrix):
    m = tf.convert_to_tensor(matrix, dtype=tf.float32)
    return tf.norm(m)
    