import numpy as np
import pickle

name = '../../PARSED/32x32/Temperature.pkl'
name2 = '../../PARSED/32x32/Cloud_cover.pkl'
name3 = '../../PARSED/32x32/Specific_humidity.pkl'


with open(name,'rb') as f:
    w = pickle.load(f, encoding='bytes')
with open(name2,'rb') as f:
    y = pickle.load(f, encoding='bytes')
with open(name3,'rb') as f:
    z = pickle.load(f, encoding='bytes')

alll = [w,y,z]

data_all = []
for x in alll:
    data_values = [i[0] for i in x]
    data_times = [i[1] for i in x]

    test = data_values[0]

    data_values = [i.reshape([1,1,32,32,1]) for i in data_values]
    data_v = np.concatenate(data_values, axis=0)

    test2 = data_v[0]

    seconds_in_day = 24*60*60
    # data_times = [i.time() for i in data_times]
    data_times = [i.second + i.minute * 60 + i.hour * 3600 for i in data_times]

    sins = [np.sin(2*np.pi*secs/seconds_in_day) for secs in data_times]
    coss = [np.cos(2*np.pi*secs/seconds_in_day) for secs in data_times]

    data_t = np.stack([sins, coss], axis=1)
    # data_t = data_t.reshape([-1,2])

    data_all.append(data_v)

data_all = np.concatenate(data_all, axis=4)

values = data_all
times = data_times

def normalize_v3(data):
    mean = np.mean(data)
    std = np.std(data)
    
    norm = data - std / mean
    normalized -= 0.5
    return norm, mean, std

def normalize_v2(data):
    minn = np.amin(data)
    maxx = np.amax(data)
    
    normalized = (data - minn) / (maxx - minn)
    normalized -= 0.5
    return normalized, minn, maxx

def preprocess(data):
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

norm1, minn, maxx = normalize_v2(values[:,:,:,:,0])
norm1_w, minn, maxx = normalize_v2([i[0] for i in w])
norm2, minn, maxx = normalize_v2(values[:,:,:,:,1])
norm3, minn, maxx = normalize_v2(values[:,:,:,:,2])

print(values.shape)
print(norm1[0:2].shape)
print(norm2[1:3].shape)
print(norm3[2:4].shape)

print(values[0,:,:,:,0])
print(w[0][0])
print(norm1[0])
print(norm1_w[0])

sq = []
for i in range(10):
   sq.append(values[i:i+2].reshape([1,2,32,32,3]))
sq = np.concatenate(sq, axis=0)
print(sq.shape)
print(sq[0].shape)

print(data_values[0], data_values[0].shape, np.min(data_values), np.max(data_values))
mmm = np.mean(data_values)
std = np.std(data_values)
print(mmm, std)
norm2 = ((data_values-mmm)/std)
print(np.min(norm2), np.max(norm2))

temp = values[:,:,:,:,0]
cc = values[:,:,:,:,1]
hum = values[:,:,:,:,2]
  
print('Temp')
mmm = np.mean(temp)
std = np.std(temp)
print('mean:', mmm,'std:', std)
norm2 = ((temp-mmm)/(std))
print('min:',np.min(norm2),'max:', np.max(norm2))

print('CC')
mmm = np.mean(cc)
std = np.std(cc)
print('mean:', mmm,'std:', std)
norm2 = ((cc-mmm)/(std))
print('min:',np.min(norm2),'max:', np.max(norm2))

print('Humidity')
mmm = np.mean(hum)
std = np.std(hum)
print('mean:', mmm,'std:', std)
norm2 = ((hum-mmm)/(std))
print('min:',np.min(norm2),'max:', np.max(norm2))

seq_list = [norm[x:2+x] for x in range(values.shape[0]-2)]
print(len(seq_list))