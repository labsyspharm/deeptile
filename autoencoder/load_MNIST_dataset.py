import numpy as np
import os
import typing

import tensorflow as tf

def load(batch_size: int) -> typing.Dict[str, typing.Any]:
    # MNIST is small, so for performance consideration load all to memory
    mnist_folderpath = '/n/scratch2/hungyiwu/MNIST'
    train_images = np.load(os.path.join(mnist_folderpath, 'train_X.npy'))
    test_images = np.load(os.path.join(mnist_folderpath, 'test_X.npy'))
    # reshape for the channel dimension, change type to float32
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    test_images = test_images.reshape(test_images.shape[0], 28, 28, 1).astype('float32')
    # get normalizer
    train_flat = train_images.reshape((-1, train_images.shape[-1]))
    test_flat = test_images.reshape((-1, test_images.shape[-1]))
    all_flat = np.concatenate([train_flat, test_flat], axis=0)
    mean = all_flat.mean(axis=0)
    std = all_flat.std(axis=0)
    # normalization
    train_images -= mean
    train_images /= std
    test_images -= mean
    test_images /= std
    # convert to tf.Dataset class
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
            .shuffle(train_images.shape[0])\
            .batch(batch_size)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)\
            .shuffle(test_images.shape[0])\
            .batch(batch_size)
    data_dict = {
            'train_dataset':train_dataset,
            'test_dataset':test_dataset,
            'train_batch_count':np.ceil(train_images.shape[0]/batch_size).astype(int),
            'test_batch_count':np.ceil(test_images.shape[0]/batch_size).astype(int),
            'data_shape':train_images.shape[1:],
            }
    return data_dict

