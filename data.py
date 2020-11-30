# -*- coding: utf-8 -*-
"""
read complex data and create dataset of tf.data
it use to complex classifier network

for it can avoid RAM not enough problem
we only read image path
then, using tf.data.dataset map function to read batch size image

input path: amplitude and phase image path
input label: label index

return: tf.data.dataset

refer repo-projects: 
https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/preprocessing/image_dataset.py#L34-L206
https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/preprocessing/dataset_utils.py

@author: Jacky Gao
@date: 2020/11/30
"""

import os
import glob
import random
import tensorflow as tf


def get_data_list(directory_Am, directory_Ph, shuffle=False, seed=1):
    def get_image_list(directory_Am, directory_Ph):
        return glob.glob(directory_Am), glob.glob(directory_Ph)
    
    def get_label_list(data_path):
        lbl_name = []
        labels = []
        for _path in data_path:
            _filename = os.path.basename(_path)    # 'Background_000000_R000_Z00_A.png'  
            _lbl_name = _filename[:_filename.find('_')]    # 'Background'
            
            if _lbl_name not in lbl_name:
                lbl_name.append(_lbl_name)
            
            _lbl_idx = lbl_name.index(_lbl_name)
            labels.append(_lbl_idx)
        return labels
    
    def shuffle_data_sequence(lists_A, lists_P, lists_L, seed):
        shuffle_idx = list(range(len(lists_A)))
        for i in range(seed):
            random.shuffle(shuffle_idx)
        lists_A = [lists_A[idx] for idx in shuffle_idx]
        lists_P = [lists_P[idx] for idx in shuffle_idx]
        lists_L = [lists_L[idx] for idx in shuffle_idx]
        return lists_A, lists_P, lists_L
    
    
    lists_A, lists_P = get_image_list(directory_Am, directory_Ph)
    lists_L = get_label_list(lists_A)
    
    if shuffle:
        return shuffle_data_sequence(lists_A, lists_P, lists_L, seed)
    return lists_A, lists_P, lists_L


def get_dataset(data_Am, data_Ph, data_Lb,
                batch_size, image_height, image_width, num_channels=3,
                shuffle=False, seed=None,
                one_hot=False, num_classes=0):
    
    def load_img(Am, Ph):
        # Read amplitude file
        XA = tf.io.read_file(Am)
        XA = tf.io.decode_image(
            XA, channels=num_channels, expand_animations=False)
        XA = tf.image.resize(XA, (image_height,image_width))
        XA = tf.dtypes.cast(XA, tf.float32)
        XA.set_shape((image_height, image_width, num_channels))
        
        # Read phase file
        XP = tf.io.read_file(Ph)
        XP = tf.io.decode_image(
            XP, channels=num_channels, expand_animations=False)
        XP = tf.image.resize(XP, (image_height,image_width))
        XP = tf.dtypes.cast(XP, tf.float32)
        XP.set_shape((image_height, image_width, num_channels))
        return {'XA': XA, 'XP': XP}
    
    def load_lbl(Lb, one_hot, num_classes):
        YC = Lb
        if one_hot:
            YC = tf.one_hot(Lb, num_classes)
        return {'YC': YC}
    
    if num_classes <= len(set(data_Lb)):
        num_classes = len(set(data_Lb))
    
    data_Im = (data_Am, data_Ph)
    ds_Im = tf.data.Dataset.from_tensor_slices(data_Im)
    ds_Im = ds_Im.map(lambda Am, Ph: load_img(Am, Ph))
    
    ds_Lb = tf.data.Dataset.from_tensor_slices(data_Lb)
    ds_Lb = ds_Lb.map(lambda x: load_lbl(x, one_hot, num_classes))
    
    dataset = tf.data.Dataset.zip((ds_Im, ds_Lb))
    dataset = dataset.batch(batch_size)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size*8, seed=seed)
    return dataset
    
    