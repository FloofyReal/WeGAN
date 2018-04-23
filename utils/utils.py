"""
utility functions
"""

import tensorflow as tf
import os
import pickle
import numpy as np


def get_all_files(source_path, filename):
    fullpath = source_path + filename
    f = []
    for (dirpath, dirnames, filenames) in walk(fullpath):
        f.extend(filenames)
        break
    return f

def f_denormalize(data, mean, std):
    denormal = (data*std)+mean
    return denormal

def denormalize(data, wvars, reshape_size, frame_count, meta)
    """
    Returns list of denormalized data.
    """
    params = ['Temperature', 'Cloud_cover', 'Specific_humidity', 'Logarithm_of_surface_pressure', 'Geopotential']
    denormals = []
    for p,c in zip(params, wvars):
        if c == '1':
            denormal = f_denormalize(data[:,:,:,:,i], meta[p+'_mean'], meta[p+'_std'])
            denormals.append(denormal.reshape(-1,frame_count,reshape_size,reshape_size,1))
    return denormals


def save_image(batch, sample_dir, name='def'):
    """
    Save an input file [image, batch, numpy array] into output pickle.
    """
    fin_name = sample_dir + '/sample_' + name + '.pkl'
    print('save image as: %s', fin_name)
    with open(fin_name, 'wb') as f:
        pickle.dump(batch, f, pickle.HIGHEST_PROTOCOL)
