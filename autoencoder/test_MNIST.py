import numpy as np
import os
import shutil
import typing
import time
import tensorflow as tf

import CVAE

if __name__ == '__main__':
    ts_start = time.time()
    # download MNIST dataset as numpy.ndarray
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
    BATCH_SIZE = 100
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images)\
            .shuffle(train_images.shape[0])\
            .batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices(test_images)\
            .shuffle(test_images.shape[0])\
            .batch(BATCH_SIZE)
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start))
    # setup module and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(latent_dim=50, input_shape=train_images.shape[1::])
    optimizer = tf.keras.optimizers.Adam()
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # test run
    total_epoch = 100
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # train loop
        ts_start = time.time()
        train_elbo = np.zeros(train_images.shape[0]//BATCH_SIZE)
        for index, train_x in enumerate(train_dataset):
            batch_loss = CVAE.compute_apply_gradients(cvae_model, train_x, optimizer)
            train_elbo[index] = batch_loss.numpy()
        train_elbo = -train_elbo.mean()
        # progress report
        test_elbo = np.zeros(test_images.shape[0]//BATCH_SIZE)
        for index, test_x in enumerate(test_dataset):
            batch_loss = CVAE.compute_loss(cvae_model, test_x)
            test_elbo[index] = batch_loss.numpy()
        test_elbo = -test_elbo.mean()
        ts_end = time.time()
        print('Epoch: {}/{} done, Train ELBO: {:.3f} Test ELBO: {:.3f}, Runtime: {:.3f} sec.'\
                .format(epoch, total_epoch, train_elbo, test_elbo, ts_end-ts_start))
    print('Done.')
