import pickle
import numpy as np

def split_dataset(data, valid=0.1, test=0.1):
    x = data
    
    size = len(x)
    # split
    train_features = x[:int(-(size * (test+valid)))]
    valid_features= x[int(-(size * (test+valid))): int(-size*test)]
    test_features = x[int(-size*test):]
    
    return train_features, valid_features, test_features

dirr = '32x32'
# dirr = '64x64'

params='Temperature'

# path to whole .pkl of data
source_path = '../PARSED/' + dirr + '/'
fin_file = params + '.pkl'

# loading dataset
with open(source_path + fin_file, 'rb') as f:
    data = pickle.load(f, encoding='latin1')

# supervised learning - 80:10:10
# data_train, data_valid, data_test = split_dataset(data, valid=0.1, test=0.1)

# unsupervised learning - no valid - 90:10
data_train, data_valid, data_test = split_dataset(data, valid=0, test=0.1)

# saving dataset as parts
with open(source_path + 'train_temp_32.pkl','wb') as f:
    pickle.dump(data_train, f, pickle.HIGHEST_PROTOCOL)

with open(source_path + 'test_temp_32.pkl','wb') as f:
    pickle.dump(test_data, f, pickle.HIGHEST_PROTOCOL)

"""
with open('valid_temp_32.pkl','wb') as f:
    pickle.dump(valid_data, f, pickle.HIGHEST_PROTOCOL)
"""