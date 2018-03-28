"""
Code to train the generation model

"""
from __future__ import print_function

from data.input_pipeline import InputPipeline

from model.improved_video_gan import ImprovedVideoGAN
from model.improved_video_gan_future import ImprovedVideoGANFuture

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
flags.DEFINE_string('mode', 'predict', 'one of [generate, predict, bw2rgb, inpaint]')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train [15]')
flags.DEFINE_integer('batch_size', 64, 'Batch size [16]')
flags.DEFINE_integer('crop_size', 32, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('channels', 1, 'Number of weather variables [1]')
flags.DEFINE_integer('frame_count', 1, 'How long videos should be in frames [32]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')

flags.DEFINE_integer('read_threads', 8, 'Read threads [16]')

flags.DEFINE_float('learning_rate', 0.001, 'Learning rate (alpha) for Adam [0.1]')
flags.DEFINE_float('beta1', 0.5, 'Beta parameter for Adam [0.5]')

flags.DEFINE_string('root_dir', '.',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('index_file', 'my-index-file.txt', 'Index file referencing all videos relative to root_dir')

flags.DEFINE_string('experiment_name', 'temperature_32x32_1step_fin_v2', 'Log directory')
flags.DEFINE_integer('output_every', 25, 'output loss to stdout every xx steps')
flags.DEFINE_integer('sample_every', 200, 'generate random samples from generator every xx steps')
flags.DEFINE_integer('save_model_every', 200, 'save complete model and parameters every xx steps')

flags.DEFINE_bool('recover_model', False, 'recover model')
flags.DEFINE_string('model_name', 'small_v1', 'checkpoint file if not latest one')
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
                         params.read_threads,
                         params.batch_size,
                         params.channels,
                         num_epochs=params.num_epochs,
                         video_frames=params.frame_count,
                         reshape_size=params.crop_size)
batch, minn, maxx = data_set.input_pipeline()

print("DATAPIPELINE DONE")
#
# set up model
#

if params.mode == 'generate':
    model = ImprovedVideoGAN(batch,
                             batch_size=params.batch_size,
                             frame_size=params.frame_count,
                             crop_size=params.crop_size,
                             learning_rate=params.learning_rate,
                             z_dim=params.z_dim,
                             beta1=params.beta1)
elif params.mode == 'predict':
    model = ImprovedVideoGANFuture(input_batch=batch,
                                   batch_size=params.batch_size,
                                   frame_size=params.frame_count,
                                   crop_size=params.crop_size,
                                   channels=params.channels,
                                   minn=minn,
                                   maxx=maxx,
                                   learning_rate=params.learning_rate,
                                   beta1=params.beta1,
                                   critic_iterations=4)
else:
    raise Exception("unknown training mode")

print('Model setup DONE')

#
# Set up coordinator, session and thread queues
#

# Saver for model.
saver = tf.train.Saver()
# Coordinator for threads in queues etc.
coord = tf.train.Coordinator()
# Create a session for running operations in the Graph.
sess = tf.Session(config=config)
# Create a summary writer
summary_writer = tf.summary.FileWriter(log_dir, sess.graph)
# Initialize the variables (like the epoch counter).
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
# Start input enqueue threads.
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#
# Recover Model
#

if params.recover_model:
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
    f.write('beta1 (adam): %f\n' % params.beta1)  # TODO make beta parametrizable in BEGAN as well
    f.close()


#
# TRAINING
#

kt = 0.0
lr = params.learning_rate
try:
    while not coord.should_stop():
        model.train(sess, i, summary_writer=summary_writer, log_summary=(i % params.output_every == 0),
                    sample_dir=sample_dir, generate_sample=(i % params.sample_every == 0))
        if i % params.save_model_every == 0:
            print('Backup model ..')
            saver.save(sess, os.path.join(checkpoint_dir, 'cp'), global_step=i)
        i += 1

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop and write final checkpoint
    saver.save(sess, os.path.join(checkpoint_dir, 'final'), global_step=i)
    coord.request_stop()

#
# Shut everything down
#
coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
