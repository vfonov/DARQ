# -*- coding: utf-8 -*-

# @author Vladimir S. FONOV
# @date 28/07/2019

from __future__ import absolute_import, division, print_function, unicode_literals
import argparse
from datetime import datetime  # for tensorboard
import os
import sys

import tensorflow as tf
# command line configuration
from tensorflow.python.platform import flags
# TPU enabled models from  https://github.com/tensorflow/tpu/
# add local copy of tpu module:
#from os.path import dirname
#sys.path.append(os.path.join(dirname(__file__),'tpu/models/'))

#import official.mobilenet.mobilenet_model as mobilenet_v1
from nets.resnet_v2 import resnet_v2_50

# using standard resnet
#import tensorflow.models.research.


# local
# from model import create_qc_model

# from tensorflow.contrib.framework.python.ops import arg_scope
# from tensorflow.contrib.training.python.training import evaluation

# slim tensorflow library
slim = tf.contrib.slim

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be the name used when "
    "creating the Cloud TPU. To find out hte name of TPU, either use command "
    "'gcloud compute tpus list --zone=<zone-name>', or use "
    "'ctpu status --details' if you have created Cloud TPU using 'ctpu up'.")
# Model specific parameters
tf.flags.DEFINE_string(
    "model_dir", default="model",
    help="This should be the path of GCS bucket which will be used as "
    "model_directory to export the checkpoints during training.")
# Model specific parameters
tf.flags.DEFINE_string(
    "training_data", default="deep_qc_data_shuffled_20190805_train.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "testing_data", default="deep_qc_data_shuffled_20190805_test.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_string(
    "validation_data", default="deep_qc_data_shuffled_20190805_val.tfrecord",
    help="This should be the path of GCS bucket with input data")
tf.flags.DEFINE_integer(
    "batch_size", default=12,
    help="This is the global batch size and not the per-shard batch.")
flags.DEFINE_integer(
    'num_cores', 1,
    'Number of shards (workers).')
tf.flags.DEFINE_integer(
    "train_epochs", default=100,
    help="Total number of training epochs")
tf.flags.DEFINE_integer(
    "eval_per_epoch", default=3,
    help="Total number of training steps per evaluation")
# tf.flags.DEFINE_integer(
#     "eval_steps", default=4,
#     help="Total number of evaluation steps. If `0`, evaluation "
#     "after training is skipped.")
tf.flags.DEFINE_integer(
    "n_samples", default=57848,
    help="Number of samples")
flags.DEFINE_float(
    'learning_rate', 1e-3, 'Initial learning rate')
tf.flags.DEFINE_integer(
    "learning_rate_decay_epochs", default=8, help="decay epochs")
flags.DEFINE_float(
    'learning_rate_decay', default=0.75, help="decay")
tf.flags.DEFINE_string(
    "optimizer", default="RMS",
    help="Training optimizer")
tf.flags.DEFINE_float(
    'depth_multiplier', default=1.0,
    help="mobilenet depth multiplier")
# tf.flags.DEFINE_bool(
#     "display_tensors", default=True,
#     help="display_tensors")

# TPU specific parameters.
tf.flags.DEFINE_bool(
    "use_tpu", default=False,
    help="True, if want to run the model on TPU. False, otherwise.")
tf.flags.DEFINE_bool(
    "moving_average", default=False,
    help="Use moving average")
# tf.flags.DEFINE_integer(
#     "iterations", default=500,
#     help="Number of iterations per TPU training loop.")
tf.flags.DEFINE_integer(
    "save_checkpoints_secs", default=600,
    help="Saving checkpoint freq")
tf.flags.DEFINE_integer(
    "save_summary_steps", default=10,
    help="Saving summary steps")
tf.flags.DEFINE_bool(
    "log_device_placement", default=False,
    help="log_device_placement")

#MULTI-GPU specific paramters
tf.flags.DEFINE_bool(
    "multigpu", default=False,
    help="Use all available GPUs")


FLAGS = tf.flags.FLAGS

# Constants dictating the learning rate schedule.
RMSPROP_DECAY = 0.9                # Decay term for RMSProp.
RMSPROP_MOMENTUM = 0.9             # Momentum in RMSProp.
RMSPROP_EPSILON = 1.0              # Epsilon term for RMSProp.

# Constants dictating moving average.
MOVING_AVERAGE_DECAY = 0.995

# Batchnorm moving mean/variance parameters
BATCH_NORM_DECAY = 0.996
BATCH_NORM_EPSILON = 1e-3


def create_inner_model(images, scope=None, is_training=True, reuse=False):
    with tf.variable_scope(scope, 'resnet', [images], reuse=reuse ) as _scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                                is_training=is_training):
            #net, _ = mobilenet_v1.mobilenet_v1_base(images, scope=_scope)
            net, _ = resnet_v2_50(images, scope=_scope, is_training=is_training, 
                                          global_pool=False,reuse=reuse)
    return net


def load_data(batch_size=None, filenames=None, training=True):
    """
    Create training dataset
    """
    if batch_size is None:
        batch_size = FLAGS.batch_size

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    raw_ds = tf.data.TFRecordDataset(filenames)

    def _parse_feature(i):
        # QC data
        feature_description = {
            'img1_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'img2_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'img3_jpeg': tf.io.FixedLenFeature([], tf.string, default_value=''),
            'qc':   tf.io.FixedLenFeature([], tf.int64,  default_value=0),
            'subj': tf.io.FixedLenFeature([], tf.int64,  default_value=0)
            #'_id':  tf.io.FixedLenFeature([], tf.string, default_value='')

        }
        # Parse the input tf.Example proto using the dictionary above.
        return tf.io.parse_single_example(i, feature_description)

    def _decode_jpeg(a):
        img1 = tf.cast(tf.image.decode_jpeg(
            a['img1_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0
        img2 = tf.cast(tf.image.decode_jpeg(
            a['img2_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0
        img3 = tf.cast(tf.image.decode_jpeg(
            a['img3_jpeg'], channels=1), dtype=tf.float32)/127.5-1.0
        # , 'subj':a['subj']
        return {'View1': img1, 'View2': img2, 'View3': img3}, {'qc': a['qc']}

    
    dataset = raw_ds.map(_parse_feature, num_parallel_calls=AUTOTUNE).map(_decode_jpeg, num_parallel_calls=AUTOTUNE)
    
    if training:
        # TODO: determine optimal buffer size, input should be already pre-shuffled
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size=6000))
        # dataset = dataset.shuffle(buffer_size=2000)
        # dataset = dataset.repeat()

    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

class LoadEMAHook(tf.train.SessionRunHook):
  def __init__(self, model_dir):
    super(LoadEMAHook, self).__init__()
    self._model_dir = model_dir

  def begin(self):
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY)
    variables_to_restore = ema.variables_to_restore()
    self._load_ema = tf.contrib.framework.assign_from_checkpoint_fn(
        tf.train.latest_checkpoint(self._model_dir), variables_to_restore)

  def after_create_session(self, sess, coord):
    tf.logging.info('Reloading EMA...')
    self._load_ema(sess)


def model_fn(features, labels, mode, params):
    """Mobilenet v1 model using Estimator API."""
    num_classes = 2
    batch_size = params['batch_size']

    training_active = (mode == tf.estimator.ModeKeys.TRAIN)
    eval_active = (mode == tf.estimator.ModeKeys.EVAL)
    predict_active = (mode == tf.estimator.ModeKeys.PREDICT)

    images = features

    images1 = tf.reshape(images['View1'], [batch_size, 224, 224, 1])
    images2 = tf.reshape(images['View2'], [batch_size, 224, 224, 1])
    images3 = tf.reshape(images['View3'], [batch_size, 224, 224, 1])
    labels  = tf.reshape(labels['qc'],    [batch_size])
    #ids     = features['id']

    # if eval_active:

    # pass input through the same network
    net1 = create_inner_model(images1, scope='InnerModel', is_training=training_active)
    net2 = create_inner_model(images2, scope='InnerModel', is_training=training_active, reuse=True)
    net3 = create_inner_model(images3, scope='InnerModel', is_training=training_active, reuse=True)

    with tf.variable_scope('addon',values=[net1]) as _scope:
        with slim.arg_scope([slim.batch_norm, slim.dropout],
                            is_training=training_active):
            print(net1)
            # concatenate along feature dimension  - 
            net = tf.concat( [net1, net2, net3], -1)
            #net = net1
            net = slim.conv2d(net, 2*512, [1, 1])
            net = slim.conv2d(net, 32, [1, 1])
            net = slim.conv2d(net, 32, [7,7], padding='VALID') # 7x7 -> 1x1 
            net = slim.conv2d(net, 32, [1,1])
            # flatten here?
            net = slim.dropout(net, 0.5)
            net = slim.conv2d(net, num_classes, [1,1])
            net_output = slim.flatten(net) # -> N,2
            #
            logits = slim.softmax( net_output )

    predictions = {
        'classes': tf.argmax(input=net_output, axis=1),
        'probabilities': logits
    }

    if predict_active:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            predictions=predictions,
            export_outputs={
                'classify': tf.estimator.export.PredictOutput(predictions)
            })

    one_hot_labels = tf.one_hot(labels, num_classes, dtype=tf.int32)

    tf.losses.softmax_cross_entropy(
        onehot_labels=one_hot_labels,
        logits=logits,
        weights=1.0,
        label_smoothing=0.0)

    loss = tf.losses.get_total_loss(add_regularization_losses=True)
    initial_learning_rate = FLAGS.learning_rate * FLAGS.batch_size / 256
    final_learning_rate = 1e-4 * initial_learning_rate

    train_op = None
    if training_active:
        batches_per_epoch = FLAGS.n_samples // FLAGS.batch_size
        global_step = tf.train.get_or_create_global_step()

        learning_rate = tf.train.exponential_decay(
            learning_rate=initial_learning_rate,
            global_step=global_step,
            decay_steps=FLAGS.learning_rate_decay_epochs * batches_per_epoch,
            decay_rate=FLAGS.learning_rate_decay,
            staircase=True)

        # Set a minimum boundary for the learning rate.
        learning_rate = tf.maximum(
            learning_rate,
            final_learning_rate,
            name='learning_rate')

        if FLAGS.optimizer == 'sgd':
            tf.logging.info('Using SGD optimizer')
            optimizer = tf.train.GradientDescentOptimizer(
                learning_rate=learning_rate)
        elif FLAGS.optimizer == 'momentum':
            tf.logging.info('Using Momentum optimizer')
            optimizer = tf.train.MomentumOptimizer(
                learning_rate=learning_rate, momentum=0.9)
        elif FLAGS.optimizer == 'RMS':
            tf.logging.info('Using RMS optimizer')
            optimizer = tf.train.RMSPropOptimizer(
                learning_rate,
                RMSPROP_DECAY,
                momentum=RMSPROP_MOMENTUM,
                epsilon=RMSPROP_EPSILON)
        elif FLAGS.optimizer == 'ADAM':
            tf.logging.info('Using ADAM optimizer')
            optimizer = tf.train.AdamOptimizer(
                learning_rate=learning_rate)
        else:
            tf.logging.fatal('Unknown optimizer:', FLAGS.optimizer)

        if FLAGS.use_tpu:
            optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss)
            # TODO: clip gradients
            gradients_norm = tf.linalg.global_norm(gradients,"gradients_norm")
            #ngradients, gradients_norm = tf.clip_by_global_norm(gradients, 2.0)
            train_op = optimizer.apply_gradients(gradients, global_step=global_step)

        if FLAGS.moving_average:
            ema = tf.train.ExponentialMovingAverage(
                decay=MOVING_AVERAGE_DECAY, num_updates=global_step)
                
            variables_to_average = (tf.trainable_variables() + tf.moving_average_variables())
            with tf.control_dependencies([train_op]), tf.name_scope('moving_average'):
                 train_op = ema.apply(variables_to_average)

    eval_metrics = None

    if eval_active:
        def metric_fn_ev(_labels, _predictions, _logits):
            return {
                'accuracy': tf.metrics.accuracy(_labels, tf.argmax(input=_predictions, axis=1)),
                'auc': tf.metrics.auc(labels, _logits[:, 1]),
                'tnr': tf.metrics.true_negatives_at_thresholds(_labels, _logits[:, 1], [0.5])
            }
        eval_metrics = (metric_fn_ev, [labels, net_output,logits])
    else: # do the same
        def metric_fn_tr(_labels, _predictions):
            return {
                'accuracy': tf.metrics.accuracy(_labels, tf.argmax(input=_predictions, axis=1)),
                #'auc': tf.metrics.auc(labels, predictions[:, 1])
            }
        eval_metrics = (metric_fn_tr, [labels, net_output])

    ############ DEBUG ##########
    #print_op = tf.print(ids)
    if not FLAGS.use_tpu:
        summary_writer = tf.contrib.summary.create_file_writer(
            os.path.join(params['model_dir'], 'debug' if training_active else "debug_e"), 
                name='debug' if training_active else "debug_e")

        with summary_writer.as_default():
            #qc_pass = tf.greater(labels, 0)
            #label_1 = tf.equal(labels, 1)

            # tf.summary.image("images1", images1)
            # tf.summary.image("images1_pass", tf.boolean_mask(images1, qc_pass))
            # tf.summary.image("images1_fail", tf.boolean_mask(images1, qc_fail))
            # tf.summary.image("images2_pass", tf.boolean_mask(images2, qc_pass))
            # tf.summary.image("images2_fail", tf.boolean_mask(images2, qc_fail))
            # tf.summary.image("images3_pass", tf.boolean_mask(images3, qc_pass))
            # tf.summary.image("images3_fail", tf.boolean_mask(images3, qc_fail))
            #tf.summary.histogram("gradients", gradients)
            if training_active:
                with tf.control_dependencies([gradients_norm, labels]): # print_ops
                    tf.summary.scalar("gradient norm",gradients_norm)
                    tf.summary.histogram( "labels",  labels )
                    # tf.summary.histogram( "logits0", logits[:,0] )
                    # tf.summary.histogram( "logits1", logits[:,1] )
            else:
                tf.summary.histogram( "Elabels",  labels )
                # tf.summary.histogram( "Elogits0", logits[:,0] )
                # tf.summary.histogram( "Elogits1", logits[:,1] )
    if not FLAGS.multigpu:
        return tf.estimator.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metrics=eval_metrics)
    else:
        return tf.estimator.EstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            eval_metric_ops={
                'accuracy': tf.metrics.accuracy(labels, tf.argmax(input=logits, axis=1)),
                'auc': tf.metrics.auc(labels, logits[:, 1]),
                'tnr': tf.metrics.true_negatives_at_thresholds(labels, logits[:, 1], [0.5])
            })


def main(argv):
    del argv  # Unused
    #tf.logging.set_verbosity('WARN')
    if FLAGS.use_tpu:
        assert FLAGS.model_dir.startswith("gs://"), ("'model_dir' should be a "
                                                     "GCS bucket path!")
        # Resolve TPU cluster and runconfig for this.
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
            FLAGS.tpu)
    else:
        tpu_cluster_resolver = None

    batch_size_per_shard = FLAGS.batch_size // FLAGS.num_cores
    batch_axis = 0

    steps_per_cycle = FLAGS.n_samples//FLAGS.batch_size//FLAGS.eval_per_epoch

    if FLAGS.multigpu:
        _strategy = tf.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            save_summary_steps=FLAGS.save_summary_steps,
            train_distribute=_strategy
            )
        inception_classifier = tf.estimator.Estimator(
            model_fn=model_fn,
            config=run_config,
            model_dir=FLAGS.model_dir,
            params={'batch_size':FLAGS.batch_size,'model_dir':FLAGS.model_dir})
    else:
        run_config = tf.estimator.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=FLAGS.save_checkpoints_secs,
            save_summary_steps=FLAGS.save_summary_steps,
            session_config=tf.ConfigProto(
                allow_soft_placement=True,
                log_device_placement=FLAGS.log_device_placement),
            tpu_config=tf.estimator.tpu.TPUConfig(
                iterations_per_loop=steps_per_cycle,
                per_host_input_for_training=True),
            )

        inception_classifier = tf.estimator.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            config=run_config,
            params={'model_dir': FLAGS.model_dir},
            train_batch_size=FLAGS.batch_size,
            eval_batch_size=FLAGS.batch_size
            ) # batch_axis=(batch_axis, 0)

    def _train_data(params):  # hack ?
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=[FLAGS.training_data],
            training=True)
        #images, labels = dataset.make_one_shot_iterator().get_next()
        return dataset
        #return images,labels

    def _eval_data(params):  # hack ?
        dataset = load_data(
            batch_size=params['batch_size'], 
            filenames=[FLAGS.validation_data],
            training=False)
        #images, labels = dataset.make_one_shot_iterator().get_next()
        return dataset
        #return images,labels

    if FLAGS.moving_average:
        eval_hooks = [LoadEMAHook(FLAGS.model_dir)]
    else:
        eval_hooks = []

    for cycle in range(FLAGS.train_epochs * FLAGS.eval_per_epoch):
        #tf.logging.info('Starting training cycle %d.' % cycle)
        inception_classifier.train(
            input_fn=_train_data,
            steps=steps_per_cycle)

        #tf.logging.info('Starting evaluation cycle %d .' % cycle)
        eval_results = inception_classifier.evaluate(
            input_fn=_eval_data,
            hooks=eval_hooks)

        tf.logging.info('Evaluation results: {}'.format(eval_results))


if __name__ == '__main__':
    # main()
    tf.app.run()
    # tf.compat.v1.app.run()
