# build-in
from datetime import datetime
import math
import time
# third-party
import numpy as np
import tensorflow as tf
import cifar10

FLAGS = tf.app.flags.FLAGS

low_ranks = []

for i in range(1, 5):
    r = int(input('rank for %d layer' % i))
    low_ranks.append(r)

modelNumber = str(input('model number?'))
eval_time = int(input('eval time?'))
tf.app.flags.DEFINE_string('eval_dir', 'cifar10_eval/ABS/' + modelNumber,
                           """Directory where to write event logs.""")
tf.app.flags.DEFINE_string('eval_data', 'test',
                           """Either 'test' or 'train_eval'.""")
tf.app.flags.DEFINE_string('checkpoint_dir', 'cifar10_train/ABS/' + modelNumber,
                           """Directory where to read model checkpoints.""")
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5,
                            """How often to run the eval.""")
tf.app.flags.DEFINE_integer('num_examples', 10000,
                            """Number of examples to run.""")
tf.app.flags.DEFINE_boolean('run_once', True,
                            """Whether to run eval only once.""")


def eval_once(saver, summary_writer, truth_num, summary_op, logits):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found!')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True,
                                                 start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and not coord.should_stop():
                true_count += sess.run(truth_num)
                step += 1
            precision = true_count / total_sample_count
            print('%s: precision @1 = %.3f' % (datetime.now(), precision))
            print(true_count)
            print(total_sample_count)
            total_number = 0
            for v in tf.trainable_variables():
                if str(v.name).find('sparse') != -1:
                    tmp = int(np.prod(v.get_shape().as_list())*0.125)
                    total_number += tmp
                else:
                    total_number += int(np.prod(v.get_shape().as_list()))
            print('total parameter number :', total_number)
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)


def evaluate():
    with tf.Graph().as_default() as g:
        eval_data = FLAGS.eval_data == 'test'
        images, labels = cifar10.inputs(eval_data)

        logits = cifar10.inference(images,r=low_ranks)
        predict = tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels)
        truth_num = tf.reduce_sum(tf.cast(predict, tf.int32))
        saver = tf.train.Saver()
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)

        # while True:
        #     eval_once(saver, summary_writer, truth_num, summary_op, logits)
        #     if FLAGS.run_once:
        #         break
        #     time.sleep(FLAGS.eval_interval_secs)
        for j in range(eval_time):
            eval_once(saver, summary_writer, truth_num, summary_op, logits)


def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()


if __name__ == '__main__':
    tf.app.run()
