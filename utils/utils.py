"""
utility functions
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import os
import pickle
import numpy as np
from PIL import Image
from mpl_toolkits.basemap import Basemap


def saveGIFBatch(batch, directory, name=''):
    """
    save gif frames into pickle
    """
    fin_name = directory + '/future_' + name + '.pkl'
    print('save gif as: %s', fin_name)
    with open(fin_name, 'wb') as f:
        pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)


def get_all_files(source_path, filename):
    fullpath = source_path + filename
    f = []
    for (dirpath, dirnames, filenames) in walk(fullpath):
        f.extend(filenames)
        break
    return f


def denormalize_v2(data, minn, maxx):
    diff = maxx - minn
    
    data += 0.5
    denormalized = (data * diff) + minn
    return denormalized


def sampleBatch(samples, batch_size, minn, maxx):
    return denormalize_v2(samples, minn, maxx)


def write_image(batch, sample_dir, name):
    fin_name = sample_dir + '/image_' + name + '.pkl'
    print('save image as: %s', fin_name)
    with open(fin_name, 'wb') as f:
        pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)