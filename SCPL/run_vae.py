import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import tqdm

from autoencoder_model import CVAE

def data_generator(data_folderpath, index):
    name_list = os.listdir(data_folderpath)
    for name in [name_list[i] for i in index]:
        data_filepath = os.path.join(data_folderpath, name)
        tile = np.load(data_filepath)
        yield (tile,)

if __name__ == '__main__':
    # paths
    data_folderpath = '/n/scratch2/hungyiwu/project_deeptile/data/tile_15x15'

    # model params
    latent_dim = 5
    batch_size = 16
    num_epoch = 5
    learning_rate = 1e-5
    train_fraction = 0.8
    verbosity = 0

    # derived params
    example_filepath = os.path.join(data_folderpath, os.listdir(data_folderpath)[0])
    tile_shape = np.load(example_filepath).shape

    num_cell = len(os.listdir(data_folderpath))
    all_index = np.arange(num_cell, dtype=int)
    np.random.shuffle(all_index)

    num_train = int(num_cell * train_fraction)
    train_index = all_index[0:num_train]
    test_index = all_index[num_train:]

    # build dataset, model, optimizer
    tile_dtype = np.float32
    train_data = tf.data.Dataset.from_generator(
            generator=lambda: data_generator(data_folderpath, train_index),
            output_types=(tile_dtype,),
            output_shapes=(tile_shape,),
            ).batch(batch_size)
    test_data = tf.data.Dataset.from_generator(
            generator=lambda: data_generator(data_folderpath, test_index),
            output_types=(tile_dtype,),
            output_shapes=(tile_shape,),
            ).batch(batch_size)

    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model = CVAE(latent_dim=latent_dim, input_shape=tile_shape, optimizer=optimizer)

    # utility function
    batch_count_dict = dict(train=None, test=None)
    def train():
        ts_start = time.time()
        epoch_loss = 0
        batch_count = 0
        for (X,) in tqdm.tqdm(iterable=train_data, desc='train',
                total=batch_count_dict['train'], disable=not verbosity):
            batch_loss = model.compute_apply_gradients(X)
            epoch_loss += tf.reduce_mean(batch_loss).numpy()
            batch_count += 1
        if batch_count_dict['train'] is None:
            batch_count_dict['train'] = batch_count
        ts_end = time.time()
        return dict(loss=epoch_loss/batch_count, runtime=ts_end-ts_start)
    def test():
        ts_start = time.time()
        epoch_loss = 0
        batch_count = 0
        for (X,) in tqdm.tqdm(iterable=test_data, desc='test',
                total=batch_count_dict['test'], disable=not verbosity):
            batch_loss = model.compute_loss(X)
            epoch_loss += tf.reduce_mean(batch_loss).numpy()
            batch_count += 1
        if batch_count_dict['test'] is None:
            batch_count_dict['test'] = batch_count
        ts_end = time.time()
        return dict(loss=epoch_loss/batch_count, runtime=ts_end-ts_start)

    # main loop
    test_output = test()
    print('before training', flush=True)
    print('test loss: {:.3f}, runtime {:.0f} sec,'.format(
        test_output['loss'], test_output['runtime']), flush=True)

    for epoch_index in range(num_epoch):
        train_output = train()
        test_output = test()
        print('epoch: {}/{}'.format(epoch_index+1, num_epoch), flush=True)
        print('train loss: {:.3f}, runtime {:.0f} sec,'.format(
            train_output['loss'], train_output['runtime']), flush=True)
        print('test loss: {:.3f}, runtime {:.0f} sec,'.format(
            test_output['loss'], test_output['runtime']), flush=True)
