import os
import tensorflow as tf
import numpy as np
from data.input_pipeline_new import InputPipeline
from utils.utils import denormalize_v2, write_image, sampleBatch

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

#
# input flags
#
flags = tf.app.flags
flags.DEFINE_integer('batch_size', 64, 'Batch size [16]')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train [15]')
flags.DEFINE_integer('crop_size', 32, 'Crop size to shrink videos [64]')
flags.DEFINE_integer('frame_count', 2, 'How long videos should be in frames [32]')
flags.DEFINE_integer('z_dim', 100, 'Dimensionality of hidden features [100]')
flags.DEFINE_integer('channels', 1, 'Number of weather variables [1]')
flags.DEFINE_integer('read_threads', 2, 'Read threads [16]')
flags.DEFINE_string('mode', 'predict_1to1', 'Model name [predict or predict_1to1]')
flags.DEFINE_string('action', 'train', 'Action of model [train, test, valid]')
flags.DEFINE_string('experiment_name', 'test', 'Log directory')
flags.DEFINE_string('checkpoint', 'cp-74600', 'checkpoint to recover')
flags.DEFINE_string('root_dir', '.',
                    'Directory containing all videos and the index file')
flags.DEFINE_string('index_file', 'my-index-file.txt', 'Index file referencing all videos relative to root_dir')
params = flags.FLAGS


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


data_set = InputPipeline(params.root_dir,
                         params.index_file,
                         read_threads=params.read_threads,
                         action=params.action,
                         batch_size=params.batch_size,
                         channels=params.channels,
                         num_epochs=params.num_epochs,
                         video_frames=params.frame_count,
                         reshape_size=params.crop_size)

# batch, minn, maxx = data_set.input_pipeline()
dataset = data_set.input_pipeline()


coord = tf.train.Coordinator()
sess = tf.Session(config=config)
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

dataset = dataset.batch(32)
iterator = dataset.make_initializable_iterator()
next_element = iterator.get_next()

# Compute for 100 epochs.
for i in range(10):
    print('Epoch:', i)
    sess.run(iterator.initializer)

    while True:
        try:
            print(sess.run(next_element))
        except tf.errors.OutOfRangeError:
            break



"""
for i in range(10):
    print(str(i*params.batch_size))

    print(sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                            labels_placeholder: labels})
    # print(sess.run(batch))
    # print(batch.shape)
    batch_z = np.random.normal(0.0, 1.0, size=[params.batch_size, params.z_dim]).astype(np.float32)
    feed_dict = {model.z_vec: batch_z}
    x = sess.run(model.videos_fake, feed_dict=feed_dict)
    x = denormalize_v2(x, minn, maxx)
    write_batch(x, sample_dir, 'test_', i, 16, 8)

i = 0
try:
    while not coord.should_stop():
        print('doing:', i)
        if i % params.save_model_every == 0:
            print('Backup model ..')
        i += 1

except tf.errors.OutOfRangeError:
    print('Done training -- epoch limit reached')
finally:
    # When done, ask the threads to stop and write final checkpoint
    print('Final checkpoint ..')
    coord.request_stop()

"""
#
# Shut everything down
#
coord.request_stop()
# Wait for threads to finish.
coord.join(threads)
sess.close()
