import pickle
import numpy as np

def split_dataset(data, valid=0.1, test=0.1):
    if test == 0 and valid == 0:
        return data
    
    x = dat
    size = len(x)
    
    # split
    train_features = x[:int(-(size * (test+valid)))]
    valid_features= x[int(-(size * (test+valid))): int(-size*test)]
    test_features = x[int(-size*test):]
    
    return train_features, valid_features, test_features

def split_time_dataset(data, test=0.1):
    x = data
    
    # split into 9 years of train and 1 year of test
    train_features = x[:-8760]
    test_features = x[-8760:]
    
    return train_features, test_features

def from64to32(data):
    print(data[0][0].shape)
    new = [[i[0][15:15+32,16:16+32],i[1]] for i in data]
    print(new[0][0].shape)
    return new

def save_dataset(train_data, valid_data, test_data, source_path, param, dirr):
    # saving dataset as parts
    
    if train_data:
        train_name = source_path + 'train_' + param + '_' + dirr + '.pkl'
        with open(train_name,'wb') as f:
            pickle.dump(train_data, f, pickle.HIGHEST_PROTOCOL)
            print('saved ' + train_name)
            
    if test_data:
        test_name = source_path + 'test_' + param + '_' + dirr + '.pkl'
        with open(test_name,'wb') as f:
            pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)
            print('saved ' + test_name)

    if valid_data:
        valid_name = source_path + 'valid_' + param + '_' + dirr + '.pkl'
        with open(valid_name,'wb') as f:
            pickle.dump(valid_data, f, pickle.HIGHEST_PROTOCOL)
            print('saved ' + valid_name)


##############################################################################
# MAIN CHANGABLE VARIABLES
dirr = '32x32'
# dirr = '64x64'

params = ['Temperature', 'Specific_humidity', 'Cloud_cover']
params_64 = ['Geopotential', 'Logarithm_of_surface_pressure']

# path to whole .pkl of data
source_path = '../../data_parsed/' + dirr + '/'

# reduced = True
reduced = False

# Supervised learning
# 80:10:10
# data_train, data_valid, data_test = split_dataset(data, valid=0.1, test=0.1)
# save_dataset(data_train, data_valid, data_test, source_path, params, dirr)

if not reduced:
    # Unsupervised learning
    for param in params:
        fin_file = param + '.pkl'
        
        print(fin_file)
        # loading dataset
        with open(source_path + fin_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        # no valid - 90:10
        
        # NO TIME
        # data_train, data_valid, data_test = split_dataset(data, valid=0, test=0.1)
        # save_dataset(data_train, data_valid, data_test, source_path, params, dirr)
        
        # WITH TIME
        data_train_time, data_test_time = split_time_dataset(data, test=0.1)
        print(len(data_train_time))
        print(data_train_time[0][1].ctime())
        
        print(len(data_test_time))
        print(data_test_time[0][1].ctime())
        save_dataset(data_train_time, [], data_test_time, source_path, param, dirr)

    if dirr == '64x64':
        for param in params_64:
            fin_file = param + '.pkl'

            print(fin_file)
            # loading dataset
            with open(source_path + fin_file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            # no valid - 90:10

            # NO TIME
            # data_train, data_valid, data_test = split_dataset(data, valid=0, test=0.1)
            # save_dataset(data_train, data_valid, data_test, source_path, params, dirr)

            # WITH TIME
            data_train_time, data_test_time = split_time_dataset(data, test=0.1)
            print(len(data_train_time))
            print(data_train_time[0][1].ctime())

            print(len(data_test_time))
            print(data_test_time[0][1].ctime())
            save_dataset(data_train_time, [], data_test_time, source_path, param, dirr)
else:
    # Unsupervised learning
    for param in params:
        fin_file = param + '.pkl'
        
        print(fin_file)
        # loading dataset
        with open(source_path + fin_file, 'rb') as f:
            data = pickle.load(f, encoding='bytes')

        data = from64to32(data)

        # no valid - 90:10
        
        # NO TIME
        # data_train, data_valid, data_test = split_dataset(data, valid=0, test=0.1)
        # save_dataset(data_train, data_valid, data_test, source_path, params, dirr)
        
        # WITH TIME
        data_train_time, data_test_time = split_time_dataset(data, test=0.1)
        print(len(data_train_time))
        print(data_train_time[0][1].ctime())
        
        print(len(data_test_time))
        print(data_test_time[0][1].ctime())
        save_dataset(data_train_time, [], data_test_time, source_path, param, dirr)

    if dirr == '64x64':
        for param in params_64:
            fin_file = param + '.pkl'

            print(fin_file)
            # loading dataset
            with open(source_path + fin_file, 'rb') as f:
                data = pickle.load(f, encoding='bytes')

            data = from64to32(data)
            # no valid - 90:10

            # NO TIME
            # data_train, data_valid, data_test = split_dataset(data, valid=0, test=0.1)
            # save_dataset(data_train, data_valid, data_test, source_path, params, dirr)

            # WITH TIME
            data_train_time, data_test_time = split_time_dataset(data, test=0.1)
            print(len(data_train_time))
            print(data_train_time[0][1].ctime())

            print(len(data_test_time))
            print(data_test_time[0][1].ctime())
            save_dataset(data_train_time, [], data_test_time, source_path, param, dirr)