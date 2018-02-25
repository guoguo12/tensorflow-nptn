import os
import time

import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

from models.nptn import two_layer_nptn

tf.app.flags.DEFINE_string('data_dir', 'MNIST_data/',
                           'where to load data from (or download data to)')
tf.app.flags.DEFINE_integer('batch_size', 32,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('train_steps', 30000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 1,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_boolean('use_seed', True,
                            'fix random seed to guarantee reproducibility')

FLAGS = tf.app.flags.FLAGS


def main():
    mnist = input_data.read_data_sets(FLAGS.data_dir, validation_size=0)

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 3])
    y = tf.placeholder(tf.int32, shape=[None])

    logits = two_layer_nptn(x, 10, 16, 4)
    loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                           logits=logits))
    opt = tf.train.AdamOptimizer().minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        avg_time = avg_loss = 0  # Averages over FLAGS.steps_per_display steps
        step = 0
        while step < FLAGS.train_steps:
            start_time = time.time()
            batch_x, batch_y = mnist.train.next_batch(FLAGS.batch_size)

            batch_x = batch_x.reshape((-1, 28, 28, 1))
            batch_x = np.concatenate((batch_x, batch_x, batch_x), axis=-1)

            cur_loss, _ = sess.run([loss, opt], {x: batch_x, y: batch_y})

            avg_time += (time.time() - start_time) / FLAGS.steps_per_display
            avg_loss += cur_loss / FLAGS.steps_per_display
            step += 1

            if step % FLAGS.steps_per_display == 0:
                print('step={}, loss={:.3f}, time={:.3f}'.format(
                    step, avg_loss, avg_time))
                avg_time = avg_loss = 0

if __name__ == '__main__':
    main()
