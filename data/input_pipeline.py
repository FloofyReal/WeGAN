"""
The input pipeline in this file takes care of loading video datasets.
Videos are stored as JPEG files of horizontally stacked frames.

The pipeline takes care of normalizing, cropping and making all videos
the same frame length.
Videos are randomized and put into batches in a multi-threaded fashion.
"""

import tensorflow as tf
import os
import pickle
import random
import numpy as np

class InputPipeline(object):
    def __init__(self, root_dir, index_file, read_threads, batch_size, channels=1,
                num_epochs=1, video_frames=32, reshape_size=32):
        """
        :param root_dir: root directory containing the index_file and all the videos
        :param index_file: list of video paths relative to root_dir
        :param read_threads: number of threads used for parallel reading
        :param batch_size: size of the batches to output
        :param num_epochs: number of epochs, use None to make infinite
        :param video_frames: number of frames every video should have in the end
                             if a video is shorter than this repeat the last frame
        :param reshape_size: videos frames are stored as 126x126 images, reshape them to
                             reduce the dimensionality
        """
        self.read_threads = read_threads
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.reshape_size = reshape_size
        self.channels = channels
        self.params = 'Temperature'

        print('DATAPIPELINE INIT')

        with open(os.path.join(root_dir, index_file)) as f:
            content = f.readlines()

        print('Path to dataset folder: ')
        print(content)
        content = [x.strip() for x in content]
        # shuffle=True origin
        self._filename_queue = tf.train.string_input_producer(content, shuffle=False, num_epochs=num_epochs)
        self._file_content = content

    def __init_dataset(self):
        """
        load weather data from saved pickle file
        based on chosen parameter - defined in init
        
        WARNING: WINDOWS SCHEME OF FILE LOADING
        TODO: Add Linux scheme (change \\ to / )
        """

        path_linux = self._file_content[0] + '/' + 'train_' + self.params + '.pkl'
        path = self._file_content[0] + '\\' + 'train_' + self.params + '.pkl'
        print('Path to loaded file: ', path)

        with open(path, 'rb') as f:
                data = pickle.load(f, encoding='latin1')
        return data

    def __normalize_v2(self, data):
        minn = np.amin(data)
        maxx = np.amax(data)
        
        normalized = (data - minn) / (maxx - minn)
        normalized -= 0.5
        return normalized, minn, maxx


    def __preprocess(self, data):
        """
        takes a image of horizontally stacked video frames and transforms
        it to a tensor of shape:
        [self.video_frames x self.reshape_size x self.reshape_size x self.channels]
        """
        # shape = tf.shape(data)
        # print(shape)
        normal, minn, maxx = self.__normalize_v2(data)
        normal.shape
        seq_list = []
        for x in range(len(normal)-self.video_frames):
            seq_tensor = tf.convert_to_tensor(normal[x:self.video_frames+x], np.float32)
            # print(seq_tensor.shape, self.reshape_size)
            seq_tensor = tf.reshape(seq_tensor, [self.video_frames, self.reshape_size, self.reshape_size, self.channels])
            seq_list.append(seq_tensor)
        print('Shape of 1 frame/state of weather', seq_tensor.shape)
        print('Num of all weather frames/states', len(seq_list))
        random.shuffle(seq_list)
        return seq_list, minn, maxx

    def input_pipeline(self):
        data = self.__init_dataset()
        seq_list, minn, maxx = self.__preprocess(data)
        video_batch = tf.train.batch([seq_list], batch_size=self.batch_size,
                                     # TODO(Bernhard): check if read_threads here actually speeds things up
                                     num_threads=self.read_threads, capacity=self.batch_size * 4, enqueue_many=True,
                                     shapes=[self.video_frames, self.reshape_size, self.reshape_size, self.channels])
        return video_batch, minn, maxx
