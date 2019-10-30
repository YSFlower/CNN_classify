import tensorflow as tf

import os


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    g = tf.Graph()
    with g.as_default():
        sess = tf.compat.v1.Session()
        hello = tf.constant('Hello, TensorFlow!')
        print(sess.run(hello))

        # a = tf.constant(10)
        # b = tf.constant(32)
        # sess.run(a + b)
        # print(sess.run(a + b))
