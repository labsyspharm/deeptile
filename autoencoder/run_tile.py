import numpy as np
import os
import shutil
import typing
import time
import tensorflow as tf

import CVAE

def tile_generator(
        folderpath: str,
        batch_size: int,
        index_array: np.ndarray=None,
        shuffle: bool,
        ) -> typing.Tuple[tf.Tensor]:
    # get full list
    tile_filepath_array = np.array([os.path.join(folderpath, name)\
            for name in os.listdir(folderpath) if name.endswith('.npy')])
    if index_array is None:
        index_array = np.arange(tile_filepath_array.shape[0])
    if shuffle:
        np.random.shuffle(index_array)
    X_list = []
    for tile_filepath in tile_filepath_array[index_array]:
        X = np.load(tile_filepath)
        # add first dimension for tensorflow
        X = X[np.newaxis, ...]
        X_list.append(X)
        if len(X_list) >= batch_size:
            X_batch = np.vstack(X_list)
            # modify type for tensorflow
            X_batch = tf.convert_to_tensor(
                    value=X_batch,
                    dtype=tf.float32,
                    )
            X_list = []
            # [WARNING] should I normalize it?
            yield (X_batch,)

if __name__ == '__main__':
    ts_start = time.time()
    # paths
    tile_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/tiles_cleanChannel'
    # train-test split
    input_count = len([name for name in os.listdir(tile_folderpath)\
            if name.endswith('.npy')])
    full_index = np.arange(input_count, dtype=int)
    np.random.shuffle(full_index)
    train_fraction = 0.7
    train_count = int(input_count*train_fraction)
    train_index = full_index[:train_count]
    test_index = full_index[train_count:]
    # get generators
    BATCH_SIZE = 100
    train_generator = tile_generator(
            folderpath=tile_folderpath,
            batch_size=BATCH_SIZE,
            index_array=train_index,
            shuffle=True,
            )
    test_generator = tile_generator(
            folderpath=tile_folderpath,
            batch_size=BATCH_SIZE,
            index_array=test_index,
            shuffle=True,
            )
    # convert to tf.Dataset class
    example_input = next(tile_generator(
            folderpath=tile_folderpath,
            batch_size=BATCH_SIZE,
            index_array=None,
            shuffle=False,
            ))
    train_dataset = tf.data.Dataset.from_generator(
            generator=train_generator,
            output_types=[type(x) for x in example_input],
            output_shapes=[x.shape for x in example_input],
            )
    ts_end = time.time()
    print('Prepare dataset took {:.3f} sec.'.format(ts_end-ts_start))
    # setup module and optimizer
    ts_start = time.time()
    cvae_model = CVAE.CVAE(latent_dim=50, input_shape=example_input.shape)
    optimizer = tf.keras.optimizers.Adam()
    ts_end = time.time()
    print('Prepare model took {:.3f} sec.'.format(ts_end-ts_start))
    # test run
    total_epoch = 100
    print('Start epoch loop ({} epochs in total).'.format(total_epoch))
    for epoch in range(1, total_epoch + 1):
        # train loop
        ts_start = time.time()
        train_elbo = np.zeros(input_count//BATCH_SIZE)
        for index, train_x in enumerate(train_dataset):
            batch_loss = CVAE.compute_apply_gradients(cvae_model, train_x, optimizer)
            train_elbo[index] = batch_loss.numpy()
        train_elbo = -train_elbo.mean()
        # progress report
        test_elbo = np.zeros(input_count//BATCH_SIZE)
        for index, test_x in enumerate(test_dataset):
            batch_loss = CVAE.compute_loss(cvae_model, test_x)
            test_elbo[index] = batch_loss.numpy()
        test_elbo = -test_elbo.mean()
        ts_end = time.time()
        print('Epoch: {}/{} done, Train ELBO: {:.3f} Test ELBO: {:.3f}, Runtime: {:.3f} sec.'\
                .format(epoch, total_epoch, train_elbo, test_elbo, ts_end-ts_start))
    print('Done.')
