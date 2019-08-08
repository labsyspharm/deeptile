import numpy as np
import os
import typing

import tensorflow as tf

def tile_generator(
        batch_size: int,
        tile_filepath_list: typing.List[str],
        tile_shape: typing.Tuple[int, int, int],
        tile_normalizer: np.ndarray=None,
        ) -> typing.Tuple[np.ndarray]:
    # X_batch = np.zeros((batch_size,)+tile_shape)
    X_batch = np.zeros((batch_size,)+tile_shape)
    for index, tile_filepath in enumerate(tile_filepath_list):
        remainder = index % batch_size
        X = np.load(tile_filepath).astype(np.float32)
        X_batch[remainder, ...] = X[..., [0]]
        if remainder == batch_size-1:
            if tile_normalizer is not None:
                X_batch -= tile_normalizer[:, 0]
                X_batch /= tile_normalizer[:, 1]
            yield (X_batch,)

def load(
        batch_size: int, 
        train_fraction: float=0.007, 
        test_fraction: float=0.003,
        ) -> typing.Dict[str, typing.Any]:
    # paths
    tile_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/'\
            'tiles_cleanChannel'
    tile_normalizer_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'output/tile_normalizer.npy'
    # train-test split
    tile_filepath_list = [os.path.join(tile_folderpath, name)\
            for name in os.listdir(tile_folderpath)\
            if name.endswith('.npy')]
    total_tile_count = len(tile_filepath_list)
    train_count = int(total_tile_count*train_fraction)
    test_count = int(total_tile_count*test_fraction)
    train_test_array = np.random.choice(
            a=tile_filepath_list,
            size=train_count+test_count,
            replace=False,
            )
    train_filepath_array = train_test_array[:train_count]
    test_filepath_array = train_test_array[train_count:]
    # get generators
    tile_shape = np.load(train_filepath_array[0])[..., [0]].shape
    tile_normalizer = np.load(tile_normalizer_filepath)
    train_generator_callable = lambda: tile_generator(
            batch_size=batch_size,
            tile_filepath_list=train_filepath_array,
            tile_shape=tile_shape,
            tile_normalizer=tile_normalizer,
            )
    test_generator_callable = lambda: tile_generator(
            batch_size=batch_size,
            tile_filepath_list=test_filepath_array,
            tile_shape=tile_shape,
            tile_normalizer=tile_normalizer,
            )
    # convert to tf.Dataset class
    output_types = (np.float32,)
    output_shapes = ((batch_size,)+tile_shape,)
    train_dataset = tf.data.Dataset.from_generator(
            generator=train_generator_callable,
            output_types=output_types,
            output_shapes=output_shapes,
            )
    test_dataset = tf.data.Dataset.from_generator(
            generator=test_generator_callable,
            output_types=output_types,
            output_shapes=output_shapes,
            )
    data_dict = {
            'train_dataset':train_dataset,
            'test_dataset':test_dataset,
            'train_batch_count':train_count//batch_size,
            'test_batch_count':test_count//batch_size,
            'data_shape':tile_shape,
            }
    return data_dict
