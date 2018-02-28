import os
import time

from keras.datasets import cifar10
import numpy as np
import sklearn.metrics
import tensorflow as tf

from models.nptn import two_layer_nptn

tf.app.flags.DEFINE_integer('batch_size', 150,
                            'batch size to use during training')
tf.app.flags.DEFINE_integer('train_steps', 49000,
                            'total minibatches to train')
tf.app.flags.DEFINE_integer('steps_per_display', 49,
                            'minibatches to train before printing loss')
tf.app.flags.DEFINE_string('train_dir', 'nptn_model',
                           'where to store the trained model')

FLAGS = tf.app.flags.FLAGS


def main():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize inputs
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    # Flatten outputs
    y_train = y_train.reshape((-1))
    y_test = y_test.reshape((-1))

    # Make validation set
    x_val, x_train = x_train[:1000], x_train[1000:]
    y_val, y_train = y_train[:1000], y_train[1000:]

    print(x_train.shape, 'train shape')
    print(x_val.shape, 'validation shape')
    print(x_test.shape, 'test shape')

    image_batch, label_batch = tf.train.shuffle_batch(
        [x_train, y_train],
        batch_size=FLAGS.batch_size,
        enqueue_many=True,
        capacity=50000,
        min_after_dequeue=10000)

    x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
    y = tf.placeholder(tf.int32, shape=[None])

    logits = two_layer_nptn(x,
                            num_classes=10,
                            out_channels=48,
                            num_transforms=1)
    loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y,
                                                           logits=logits))
    opt = tf.train.AdamOptimizer().minimize(loss)

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session() as sess:
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        sess.run(tf.global_variables_initializer())

        avg_time = avg_loss = 0  # Averages over FLAGS.steps_per_display steps
        step = 0
        while step < FLAGS.train_steps:
            start_time = time.time()

            batch_x, batch_y = sess.run([image_batch, label_batch])
            cur_loss, _ = sess.run([loss, opt], {x: batch_x, y: batch_y})

            avg_time += (time.time() - start_time) / FLAGS.steps_per_display
            avg_loss += cur_loss / FLAGS.steps_per_display
            step += 1

            if step % FLAGS.steps_per_display == 0:
                epoch = step * FLAGS.batch_size / x_train.shape[0]

                outputs = np.argmax(sess.run(logits, {x: x_val}), axis=-1)
                val_acc = sklearn.metrics.accuracy_score(y_val, outputs)

                print('step={}, epoch={:.2f}, loss={:.3f}, val={:.3f}, time={:.3f}'.format(
                    step, epoch, avg_loss, val_acc, avg_time), flush=True)
                avg_time = avg_loss = 0

        test_logits, test_loss = sess.run([logits, loss], {x: x_test, y: y_test})
        outputs = np.argmax(test_logits, axis=-1)
        test_acc = sklearn.metrics.accuracy_score(y_test, outputs)
        print('Test accuracy: {:.4f}'.format(test_acc))
        print('Test loss: {:.4f}'.format(test_loss))

        if not os.path.exists(FLAGS.train_dir):
            os.makedirs(FLAGS.train_dir)
        saver.save(sess, FLAGS.train_dir + '/ckpt', global_step=step)

        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':
    main()
