"""
Code to train the generation model

"""
from data.input_pipeline_new import InputPipeline
from utils.utils import denormalize

from model.wegan_1_to_1 import WeGAN1to1

import os
import re
import tensorflow as tf

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

print('Go Go Power Rangers!!!')
#
# input flags
#
flags = tf.app.flags
flags.DEFINE_string('mode', 'predict_1to1', 'one of [predict, predict_1to1]')
flags.DEFINE_integer('num_epochs', 15, 'Number of epochs to train [15]')
flags.DEFINE_integer('batch_size', 64, 'Batch size [16]')
flags.DEFINE_integer('crop_size', 32, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('frame_count', 2, 'How long videos should be in frames [32]')
flags.DEFINE_integer('channels', 1, 'Number of weather variables [1]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')
flags.DEFINE_float('learning_rate', 0.001, 'Learning rate (alpha) for Adam [0.1]')
flags.DEFINE_float('beta1', 0.5, 'Beta parameter for Adam [0.5]')
flags.DEFINE_string('wvars', '11100', 'Define which weather variables are in use [T|CC|SH|SP|GEO] [11100]')

flags.DEFINE_string('dataset', '32x32', 'Size of a map [32x32 or 64x64]')
flags.DEFINE_string('action', 'train', 'Action of model [train, test, valid]')

flags.DEFINE_string('root_dir', '.',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('index_file', 'my-index-file.txt', 'Index file referencing all videos relative to root_dir')

flags.DEFINE_string('experiment_name', 'testytest_deleteme', 'Log directory')
flags.DEFINE_integer('output_every', 34, 'output loss to stdout every xx steps')
flags.DEFINE_integer('sample_every', 136*5, 'generate random samples from generator every xx steps')
flags.DEFINE_integer('save_model_every', 136*5, 'save complete model and parameters every xx steps')

flags.DEFINE_bool('recover_model', False, 'recover model')
flags.DEFINE_string('checkpoint', 'final-136001', 'recover model name')
flags.DEFINE_string('model_name', 'bigboi', 'checkpoint file if not latest one')
params = flags.FLAGS

#
# make sure all necessary directories are created
#
# experiment_dir = os.path.join('.', params.experiment_name)
path_dir = '/home/rafajdus/experiments'
experiment_dir = os.path.join(path_dir, params.experiment_name)
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
sample_dir = os.path.join(experiment_dir, 'samples')
log_dir = os.path.join(experiment_dir, 'logs')

print('PATHS TO FILES OF EXPERIMENT:')
print('Samples: ', sample_dir)
print('Checkpoints: ', checkpoint_dir)
print('Logs: ', log_dir)

for path in [experiment_dir, checkpoint_dir, sample_dir, log_dir]:
    if not os.path.exists(path):
        os.mkdir(path)

#
# set up input pipeline
#
data_set = InputPipeline(params.root_dir,
                         params.index_file,
                         action=params.action,
                         dataset=params.dataset,
                         batch_size=params.batch_size,
                         channels=params.channels,
                         wvars=params.wvars,
                         video_frames=params.frame_count,
                         reshape_size=params.crop_size)
values, times, meta = data_set.input_pipeline()
print("DATAPIPELINE DONE")

values_placeholder = tf.placeholder(values.dtype, values.shape)
time_placeholder = tf.placeholder(times.dtype, times.shape)

dataset = tf.data.Dataset.from_tensor_slices((values_placeholder, time_placeholder))

print(dataset.output_types)
print(values.shape, times.shape)
print(dataset.output_shapes)

dataset = dataset.shuffle(100000)
dataset = dataset.apply(tf.contrib.data.batch_and_drop_remainder(params.batch_size))
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()
#
# set up model
#
if params.mode == 'predict_1to1':
    model = WeGAN1to1(input_batch=next_element[0],
                      batch_size=params.batch_size,
                      frame_size=params.frame_count,
                      crop_size=params.crop_size,
                      channels=params.channels,
                      wvars=params.wvars,
                      learning_rate=params.learning_rate,
                      beta1=params.beta1,
                      critic_iterations=4)
else:
    raise Exception("unknown training mode")

print('Model setup DONE')

#
# Set up coordinator, session and thread queues
#

# Create a session for running operations in the Graph.
sess = tf.Session(config=config)
# Create a summary writer
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
# Initialize the variables (like the epoch counter).
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# Saver for model.
# saver = tf.train.Saver(tf.trainable_variables())
saver = tf.train.Saver()
#
# Recover Model
#
if params.recover_model:
    try:
        saver.restore(sess, os.path.join(checkpoint_dir, params.checkpoint))
        i = int(params.checkpoint.split('-')[-1]) + 1
    except:
        latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
        print(latest_cp)
        if latest_cp is not None:
            print("restore....")
            saver.restore(sess, latest_cp)
            i = int(re.findall('\d+', latest_cp)[-1]) + 1
        else:
            raise Exception("no checkpoint found to recover")
else:
    i = 0

#
# backup parameter configurations
#
with open(os.path.join(experiment_dir, 'hyperparams_{}.txt'.format(i)), 'w+') as f:
    f.write('general\n')
    f.write('crop_size: %d\n' % params.crop_size)
    f.write('frame_count: %d\n' % params.frame_count)
    f.write('batch_size: %d\n' % params.batch_size)
    f.write('z_dim: %d\n' % params.z_dim)
    f.write('\nlearning\n')
    f.write('learning_rate: %f\n' % params.learning_rate)
    f.write('beta1 (adam): %f\n' % params.beta1)
    f.close()

#
# TRAINING
#

# for var in tf.trainable_variables():
#    print(var)

kt = 0.0
lr = params.learning_rate
for e in range(params.num_epochs):
    print('Epoch:', e)
    sess.run(iterator.initializer, feed_dict={values_placeholder: values, time_placeholder: times})
    # print(sess.run(tf.trainable_variables()))
    while True:
        try:
            model.train(sess, i, summary_writer=summary_writer, log_summary=(i % params.output_every == 0),
                        sample_dir=sample_dir, generate_sample=(i % params.sample_every == 0), meta=meta)
            i += 1
        except tf.errors.OutOfRangeError:
            print('Steps:', i)
            # print('Backup model ..')
            # saver.save(sess, os.path.join(checkpoint_dir, 'cp'), global_step=i)
            break

print('Done training -- epoch limit reached')
# When done, write final checkpoint
saver.save(sess, os.path.join(checkpoint_dir, 'final'), global_step=i)
print('We donezo')

#
# Shut everything down
#
sess.close()
