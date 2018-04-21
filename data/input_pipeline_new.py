"""
The input pipeline in this file takes care of loading weather dataset.
Videos are stored as pickle files of all weather states in time ordered sequence.

The pipeline takes care of normalizing, cropping and making all data
the same frame length.
Sequences of weather are randomized and return a tf.data.Dataset object to work with in main_train and main_sample.
"""

import tensorflow as tf
import os
import pickle
import random
import numpy as np

class InputPipeline(object):
    def __init__(self, root_dir, index_file, action, dataset, batch_size, channels=1,
                num_epochs=1, video_frames=2, reshape_size=32):
        """
        :param mode: action for data [train, test, valid]
        :param root_dir: root directory containing the index_file and all the videos
        :param index_file: list of video paths relative to root_dir
        :param batch_size: size of the batches to output
        :param num_epochs: number of epochs, use None to make infinite
        :param video_frames: number of frames every video should have in the end
                             if a video is shorter than this repeat the last frame
        :param reshape_size: videos frames are stored as 126x126 images, reshape them to
                             reduce the dimensionality
        """
        self.action = action
        self.datapath = dataset
        self.batch_size = batch_size
        self.video_frames = video_frames
        self.reshape_size = reshape_size
        self.channels = channels
        self.params = ['Temperature', 'Cloud_cover', 'Specific_humidity', 'Logarithm_of_surface_pressure', 'Geopotential']

        print('DATAPIPELINE INIT')

        with open(os.path.join(root_dir, index_file)) as f:
            content = f.readlines()

        print('Path to dataset folder: ')
        print(content)
        content = [x.strip() for x in content]

        self.file_content = content

    def __init_dataset(self):
        """
        load weather data from saved pickle file
        based on chosen parameter - defined in init
        """
        print('Content loaded:', self.file_content)

        data_all = []
        for i in range(self.channels):
            path_linux = self.file_content[0] + '/' + self.datapath + '/' + self.action + '_' + self.params[i] + '_' + '32x32' + '.pkl'
            print('Path to loaded file: ', path_linux)

            with open(path_linux, 'rb') as f:
                    data = pickle.load(f, encoding='bytes')
                    data_all.append(data)
        
        data_values_all = []
        for data in data_all:
            data_values = [i[0] for i in data]
            data_times = [i[1] for i in data]

            # print('Data[0] value:', data_values[0])
            # print('Data[1] value:', data_times[0])

            data_values = [i.reshape([1,1,self.reshape_size,self.reshape_size,1]) for i in data_values]
            data_values = np.concatenate(data_values, axis=0)
            
            data_values_all.append(data_values)

        data_values = np.concatenate(data_values_all, axis=4)

        seconds_in_day = 24*60*60
        data_times = [i.second + i.minute * 60 + i.hour * 3600 for i in data_times]

        sins = [np.sin(2*np.pi*secs/seconds_in_day) for secs in data_times]
        coss = [np.cos(2*np.pi*secs/seconds_in_day) for secs in data_times]

        data_times = np.stack([sins, coss], axis=1)
        data_times = data_times.astype(np.float32)

        """
        features_placeholder = tf.placeholder(data_values.dtype, data_values.shape)
        time_placeholder = tf.placeholder(data_times.dtype, data_times.shape)

        dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, time_placeholder))
        """
        del data
        del data_all
        print('u good to go')

        return tf.data.Dataset.from_tensor_slices((data_values, data_times))

    def __normalize_v2(self, data):
        minn = np.amin(data)
        maxx = np.amax(data)
        
        normalized = (data - minn) / (maxx - minn)
        normalized -= 0.5
        return normalized, minn, maxx

    def __preprocess(self, data):
        """
        output shape:
        [self.video_frames x self.reshape_size x self.reshape_size x self.channels]
        """
        # shape = tf.shape(data)
        print('Original data shape:', len(data), data[0].shape)
        normal, minn, maxx = self.__normalize_v2(data)
        print('Normalized data shape:', len(normal), normal[0].shape)
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
        dataset = self.__init_dataset()
        # return dataset, 0, 100
        return dataset
