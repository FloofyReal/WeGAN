from model.improved_video_gan_future import ImprovedVideoGANFuture
from model.improved_video_gan_future_1_to_1 import ImprovedVideoGANFutureOne

import os
import tensorflow as tf
import numpy as np
from data.input_pipeline_new import InputPipeline
from utils.utils import denormalize, save_image

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

print('What time it is? Its testing time!')
#
# input flags
#
flags = tf.app.flags
flags.DEFINE_string('mode', 'predict_1to1', 'Model name [predict or predict_1to1]')
flags.DEFINE_integer('batch_size', 1, 'Batch size [16]')
flags.DEFINE_integer('crop_size', 32, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('frame_count', 2, 'How long videos should be in frames [32]')
flags.DEFINE_integer('channels', 1, 'Number of weather variables [1]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')
flags.DEFINE_string('wvars', '10000' , 'Define which weather variables are in use [T|CC|SH|SP|GEO] [11100]')

flags.DEFINE_string('dataset', '32x32', 'Size of a map [32x32 or 64x64]')
flags.DEFINE_string('action', 'test', 'Action of model [train, test, valid]')

#EXACT CHECKPOINT TO SAMPLE FROM
flags.DEFINE_string('checkpoint', 'cp-final', 'checkpoint to recover')

flags.DEFINE_string('experiment_name', 'test', 'Log directory')
flags.DEFINE_string('root_dir', '.',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('index_file', 'my-index-file.txt', 'Index file referencing all videos relative to root_dir')
params = flags.FLAGS

#
# make sure all necessary directories are created
#
# experiment_dir = os.path.join('.', params.experiment_name)
path_dir = '/home/rafajdus/experiments'
experiment_dir = os.path.join(path_dir, params.experiment_name)
checkpoint_dir = os.path.join(experiment_dir, 'checkpoints')
sample_dir = os.path.join(experiment_dir, 'samples/test')
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

iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

if params.mode == 'predict':
    model = ImprovedVideoGANFuture(input_batch=next_element[0],
                                   batch_size=params.batch_size,
                                   frame_size=params.frame_count,
                                   crop_size=params.crop_size,
                                   channels=params.channels,
                                   wvars=params.wvars,
                                   learning_rate=0.1,
                                   beta1=0.1,
                                   critic_iterations=4)
elif params.mode == 'predict_1to1':
    model = ImprovedVideoGANFutureOne(input_batch=next_element[0],
                                   batch_size=params.batch_size,
                                   frame_size=params.frame_count,
                                   crop_size=params.crop_size,
                                   channels=params.channels,
                                   wvars=params.wvars,
                                   learning_rate=0.1,
                                   beta1=0.1,
                                   critic_iterations=4)
else:
    raise Exception("unknown training mode")

#
# Set up coordinator, session and thread queues
#
saver = tf.train.Saver(tf.trainable_variables())

sess = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

# LOAD PRE-TRAINED MODEL
# saver.restore(sess, os.path.join(checkpoint_dir,params.checkpoint))

latest_cp = tf.train.latest_checkpoint(checkpoint_dir)
print('Latest checkpoint:', latest_cp)
if latest_cp is not None:
    print("restoring....")
    saver.restore(sess, latest_cp)
else:
    raise Exception("no checkpoint found to recover")

sess.run(iterator.initializer, feed_dict={values_placeholder: values, time_placeholder: times})

weather_params = ['Temperature', 'Cloud_cover', 'Specific_humidity', 'Logarithm_of_surface_pressure', 'Geopotential']

global_rmse_temp = 0
global_rmse_cc = 0
global_rmse_sh = 0
global_rmse_sp = 0
global_rmse_geo = 0
global_rmse = [global_rmse_temp, global_rmse_cc, global_rmse_sh, global_rmse_sp, global_rmse_geo]
i = 1
global_diff = [0,0,0,0,0]
while True:
    try:
        rmse_all, diff = model.test(sess, i, sample_dir=sample_dir, meta=meta)
        for er, dif, k in zip(rmse_all, diff range(len(global_rmse))):
            global_rmse[k] += er
            global_diff[k] += dif
        i += 1
    except tf.errors.OutOfRangeError:
        for rmse, p, dif  in zip(global_rmse, weather_params, global_diff):
            print("Global RMSE of %s: %g" % (p, rmse/i))

            print('Saving global and mean diffs')
            save_image(dif, sample_dir, 'diff_global_%s' % p)
            save_image(dif/i, sample_dir, 'diff_mean_%s' % p)
        break


print('Testo donezo')
#
# Shut everything down
#
# Wait for threads to finish.
sess.close()
