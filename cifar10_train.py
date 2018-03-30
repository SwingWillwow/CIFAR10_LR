# build-in
from datetime import datetime
import time
# third-party
import tensorflow as tf
import cifar10
import numpy as np

FLAGS = tf.app.flags.FLAGS
low_ranks = []

for i in range(1, 5):
    r = int(input('rank for %d layer' % i))
    low_ranks.append(r)

modelNumber = str(input('model number?'))

# define  global information
tf.app.flags.DEFINE_string('train_dir', 'cifar10_train/ABS/'+modelNumber,
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")


def train():
    with tf.Graph().as_default():
        # get global step
        global_step = tf.train.get_or_create_global_step()
        # get data through cpu
        with tf.device('/cpu:0'):
            images, labels = cifar10.distorted_inputs()
        logits = cifar10.inference(images=images, r=low_ranks)
        loss = cifar10.loss(logits=logits, labels=labels)
        # set train_op
        train_op = cifar10.train(loss, global_step)
        for v in tf.trainable_variables():
            print(v)

        class _LoggerHook(tf.train.SessionRunHook):
            """
            log session and runtime info
            """
            def begin(self):
                self._step = -1
                self._start_time = time.time()

            def before_run(self, run_context):
                self._step += 1
                return tf.train.SessionRunArgs(loss)

            def after_run(self, run_context, run_values):
                if self._step % FLAGS.log_frequency == 0:
                    current_time = time.time()
                    duration = current_time - self._start_time
                    self._start_time = current_time
                    loss_value = run_values.results
                    example_per_sec = FLAGS.log_frequency * FLAGS.batch_size / duration
                    sec_per_batch = float(duration / FLAGS.log_frequency)

                    format_str = ('%s: step %d, loss = %.6f (%.1f examples/sec;'
                                  '%.3f sec/batch)')
                    print(format_str % (datetime.now(), self._step, loss_value,
                                        example_per_sec, sec_per_batch))

        with tf.train.MonitoredTrainingSession(
                checkpoint_dir=FLAGS.train_dir,
                hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                       tf.train.NanTensorHook(loss),
                       _LoggerHook()],
                config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement)) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)


def main(argv = None):
    cifar10.maybe_download_and_extract()
    new_model = str(input('Train a new model?'))
    new_model = new_model == 'True'
    if tf.gfile.Exists(FLAGS.train_dir) and new_model:
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
        tf.gfile.MakeDirs(FLAGS.train_dir)
    train()


if __name__ == '__main__':
    tf.app.run()



