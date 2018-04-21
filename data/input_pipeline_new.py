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
                wvars, video_frames=2, reshape_size=32):
        """
        :param mode: action for data [train, test, valid]
        :param root_dir: root directory containing the index_file and all the videos
        :param index_file: list of video paths relative to root_dir
        :param batch_size: size of the batches to output
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
        self.wvars = wvars
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

        wvars = str(self.wvars)

        test = 0
        for c in wvars:
            test =+ int(c)
        if not test == self.channels:
            print('U FUCKED UP M8')
            break

        # magic number 5 = number of max channels
        for i in range(5):
            if wvars[i] == '1':
                path_linux = self.file_content[0] + '/' + self.datapath + '/' + self.action + '_' + self.params[i] + '_' + '32x32' + '.pkl'
                print('Path to loaded file: ', path_linux)

                with open(path_linux, 'rb') as f:
                        data = pickle.load(f, encoding='bytes')
                        data_all.append(data)
        
        path_to_meta = self.file_content[0] + '/' + self.datapath + '/' + 'meta' + '_' + self.datapath + '.pkl'
        with open(path_to_meta,'rb') as f:
            self.meta = pickle.load(f)

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

        del data
        del data_all
        print('u good to go')

        return data_values, data_times

    def __normalize_v3(self, data, mean, std):
        return data-mean/std

    def __preprocess(self, data, times):
        """
        output shape (numpy array):
        [all_steps x self.video_frames x self.reshape_size x self.reshape_size x self.channels]
        """
        # shape = tf.shape(data)
        minmax = []
        normals = []
        print('Original data shape:', data.shape)
        for p,c in zip(self.params, self.wvars):
            if c == '1':
                normal = self.__normalize_v3(data[:,:,:,:,i], self.meta[p+'_mean'], self.meta[p+'_std'])
                normals.append(normal.reshape(-1,1,self.reshape_size,self.reshape_size,1))
        normals = normals.concatenate(normals, axis=4)
        print('Normalized data shape:', normals.shape)
        
        seq_list = []
        time_list = []
        for x in range(normals.shape[0]-self.video_frames):
            seq_tensor = tf.convert_to_tensor(normal[x:self.video_frames+x], np.float32)
            # print(seq_tensor.shape, self.reshape_size)
            seq_tensor = tf.reshape(seq_tensor, [1, self.video_frames, self.reshape_size, self.reshape_size, self.channels])
            seq_list.append(seq_tensor)
            time_tensor = tf.convert_to_tensor(times[x:self.video_frames+x], np.float32)
            time_tensor = tf.reshape(time_tensor, [1, self.video_frames, 2])
            time_list.append(time_tensor)

        print('Shape of 1 frame/state of weather', seq_tensor.shape)
        print('Num of all weather frames/states', len(seq_list))

        c = list(zip(seq_list, time_list))

        random.shuffle(c)

        seq_list, time_list = zip(*c)
        seq_list = np.concatenate(seq_list, axis=0)
        time_list = np.concatenate(time_list, axis=0)
        print('Final array shape:', seq_list.shape)
        print('Final time shape:', time_list.shape)
        return seq_list, time_list

    def input_pipeline(self):
        values, times = self.__init_dataset()
        values = self.__preprocess(values)
        # return dataset, 0, 100
        return values, times
