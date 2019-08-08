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
    # normalize intensity to the range [0., 1.]
    train_images /= 255.
    test_images /= 255.
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
            'train_batch_count':train_images.shape[0]//batch_size,
            'test_batch_count':test_images.shape[0]//batch_size,
            'data_shape':train_images.shape[1:],
            }
    return data_dict

