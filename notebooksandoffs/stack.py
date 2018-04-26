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

print(len(w))
print(len(y))
print(len(z))

print(w[0])

print(w[0][0].shape)
print(y[0][0].shape)
print(z[0][0].shape)

# print(x[0][1].ctime())

alll = [w,y,z]

data_all = []
for x in alll:
    data_values = [i[0] for i in x]
    data_times = [i[1] for i in x]

    print(len(data_values))
    print(len(data_times))

    test = data_values[0]

    data_values = [i.reshape([1,1,32,32,1]) for i in data_values]
    print(data_values[0].shape)
    data_v = np.concatenate(data_values, axis=0)
    print(data_v.shape)

    test2 = data_v[0]

    print(test)
    print(test2)
    print(test.shape)
    print(test2.shape)

    seconds_in_day = 24*60*60
    # data_times = [i.time() for i in data_times]
    data_times = [i.second + i.minute * 60 + i.hour * 3600 for i in data_times]

    sins = [np.sin(2*np.pi*secs/seconds_in_day) for secs in data_times]
    coss = [np.cos(2*np.pi*secs/seconds_in_day) for secs in data_times]

    data_t = np.stack([sins, coss], axis=1)
    # data_t = data_t.reshape([-1,2])

    print(data_t.shape)
    print(data_t[0])
    print(data_t[1])

    data_all.append(data_v)

data_all = np.concatenate(data_all, axis=4)
print(data_all.shape)

x = input('Waiting')