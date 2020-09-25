import numpy as np
import time
import argparse
import os

import h5py
import skimage.transform
import tensorflow as tf
import tqdm

import ACAE

def get_generator(x):
    def generator():
        for index in range(x.shape[0]):
            yield (x[index, ...],)
    return generator

if __name__ == '__main__':
    # parameters
    BATCH_SIZE = 10
    LATENT_DIM = 20
    LABEL_NUM = 10
    TOTAL_EPOCH = int(1e2)
    LEARNING_RATE = 1e-6
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    # data
    MNIST_hdf5_filepath = '/n/scratch2/hungyiwu/MNIST/MNIST.hdf5'
    if not os.path.isfile(MNIST_hdf5_filepath):
        print('MNIST hdf5 file not found, generating it...')
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        with h5py.File(MNIST_hdf5_filepath, 'w') as outfile:
            dump_step = 100
            (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
            outfile.create_dataset('x_train', shape=(x_train.shape[0], 300, 300, 3),
                    dtype=np.float32, chunks=True)
            for index in tqdm.tqdm(
                    iterable=np.array_split(np.arange(x_train.shape[0]).astype(int), dump_step),
                    desc='dump x_train',
                    disable=not args.verbose,
                    ):
                tile = x_train[index, ...]
                tile = skimage.transform.resize(tile, (tile.shape[0], 300, 300))
                tile = tile.astype(np.float32) / 255
                tile = np.stack([tile]*3, axis=-1)
                outfile['x_train'][index, ...] = tile
            del x_train
            outfile.create_dataset('x_test', shape=(x_test.shape[0], 300, 300, 3),
                    dtype=np.float32, chunks=True)
            for index in tqdm.tqdm(
                    iterable=np.array_split(np.arange(x_test.shape[0]).astype(int), dump_step),
                    desc='dump x_test',
                    disable=not args.verbose,
                    ):
                tile = x_test[index, ...]
                tile = skimage.transform.resize(tile, (tile.shape[0], 300, 300))
                tile = tile.astype(np.float32) / 255
                tile = np.stack([tile]*3, axis=-1)
                outfile['x_test'][index, ...] = tile
            del x_test
            outfile.create_dataset('y_train', data=y_train, chunks=True)
            outfile.create_dataset('y_test', data=y_test, chunks=True)
            del y_train, y_test
        print('Done MNIST hdf5 preprocessing.')
    infile = h5py.File(MNIST_hdf5_filepath, 'r')
    train_dataset = tf.data.Dataset.from_generator(
            generator=get_generator(infile['x_train']),
            output_types=(np.float32,))\
                    .batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_generator(
            generator=get_generator(infile['x_test']),
            output_types=(np.float32,))\
                    .batch(BATCH_SIZE)
    train_batch_count = np.ceil(infile['x_train'].shape[0] / BATCH_SIZE).astype(int)
    test_batch_count = np.ceil(infile['x_test'].shape[0] / BATCH_SIZE).astype(int)
    # model and optimizer
    model = ACAE.ACAE(
            latent_dim=LATENT_DIM,
            label_num=LABEL_NUM,
            feature_shape=infile['x_train'][0, ...].shape,
            )
    opt_dict = {'reconstruction':tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            'discrimination':tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            'confusion':tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE),
            }
    # loop
    for epoch in range(TOTAL_EPOCH):
        ts_start = time.time()
        train_loss_dict = {}
        test_loss_dict = {}
        # train
        for (batch_x,) in tqdm.tqdm(
                iterable=train_dataset,
                desc='train',
                total=train_batch_count,
                disable=not args.verbose,
                ):
            # loss
            loss_dict = model.compute_loss(batch_x)
            # gradient
            for key in loss_dict:
                print(key)
                print(loss_dict[key])
                if key == 'reconstruction':
                    var_list = model.inference_net.trainable_variables\
                            + model.generation_net.trainable_variables
                elif key == 'discrimination':
                    var_list = model.discriminant_net.trainable_variables
                elif key == 'confusion':
                    var_list = model.inference_net.trainable_variables
                opt_dict[key].minimize(
                        loss=lambda: tf.reduce_mean(loss_dict[key]),
                        var_list=var_list,
                        )
            # record
            if len(train_loss_dict):
                for key in loss_dict:
                    train_loss_dict[key] += loss_dict[key].numpy().mean()
            else:
                for key in loss_dict:
                    train_loss_dict[key] = loss_dict[key].numpy().mean()
        # average
        for key in train_loss_dict:
            train_loss_dict[key] /= train_batch_count
        # test
        for (batch_x,) in tqdm.tqdm(
                iterable=test_dataset,
                desc='test',
                total=test_batch_count,
                disable=not args.verbose,
                ):
            loss_dict = model.compute_loss(batch_x)
            # record
            if len(test_loss_dict):
                for key in loss_dict:
                    test_loss_dict[key] += loss_dict[key].numpy().mean()
            else:
                for key in loss_dict:
                    test_loss_dict[key] = loss_dict[key].numpy().mean()
        # average
        for key in test_loss_dict:
            test_loss_dict[key] /= test_batch_count
        ts_end = time.time()
        # report
        print('=' * 10)
        print('epoch {}, runtime {:.3f} sec'.format(epoch, ts_end-ts_start))
        print('train')
        for phase_name in ['reconstruction', 'discrimination', 'confusion']:
            print('- {} loss {:.3E}'.format(phase_name, train_loss_dict[phase_name]))
        print('test')
        for phase_name in ['reconstruction', 'discrimination', 'confusion']:
            print('- {} loss {:.3E}'.format(phase_name, test_loss_dict[phase_name]))

