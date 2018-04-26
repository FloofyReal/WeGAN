import numpy as np
import pickle
from collections import defaultdict

name = '../../data_parsed/32x32/Temperature.pkl'
name2 = '../../data_parsed/32x32/Cloud_cover.pkl'
name3 = '../../data_parsed/32x32/Specific_humidity.pkl'

# names_32 = [name, name2, name3]
names_32 = [name]

name = '../../data_parsed/64x64/Temperature.pkl'
name2 = '../../data_parsed/64x64/Cloud_cover.pkl'
name3 = '../../data_parsed/64x64/Specific_humidity.pkl'

# names_64 = [name, name2, name3]
names_64 = [name]

data_32 = []
for name in names_32:
    with open(name,'rb') as f:
        x = pickle.load(f, encoding='bytes')
        data_32.append(x)
print('loaded 32x32 data')

data_64 = []
for name in names_64:
    with open(name,'rb') as f:
        x = pickle.load(f, encoding='bytes')
        data_64.append(x)
print('loaded 64x64 data')

hitmap = defaultdict(int)

for param32,param64 in zip(data_32,data_64):
    print('doing new param')
    for i in range(2000):
        init_64 = param64[17544+(i*10)][0]
        init_32 = param32[i][0]

        mins = []
        posit = []
        for i in range(32):
            for j in range(32):
                # print(i,j)
                small = init_64[j:j+32,i:i+32]
                # print(small.shape)
                diff = small - init_32
                minn = np.mean(np.abs(diff))
                # print(minn)
                mins.append(minn)
                posit.append((i,j))
        m = 10
        for el,pos in zip(mins,posit):
            if el < m:
                m = el
                pos_best = pos
        name = 'i' + str(pos_best[0]) + '_j' + str(pos_best[1])
        hitmap[name] += 1
    # print(hitmap)

# print(hitmap)
d_view = [ (v,k) for k,v in hitmap.items() ]
d_view.sort(reverse=True) # natively sort tuples by first element
top = 5
for v,k in d_view:
    print(top)
    if top < 0:
        break
    top -= 1
    print("%s: %d" % (k,v))
    print('frame[j:j+32,i:i=32]')
