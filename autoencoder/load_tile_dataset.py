import numpy as np
import os
import typing

import tifffile
import tensorflow as tf

def tile_generator(
        batch_size: int,
        tile_filepath_list: typing.List[str],
        tile_shape: typing.Tuple[int, int, int],
        channel_bounds: np.ndarray,
        ) -> typing.Tuple[tf.Tensor]:
    X_batch = np.zeros((batch_size,)+tile_shape)
    for index, tile_filepath in enumerate(tile_filepath_list):
        remainder = index % batch_size
        X_batch[remainder, ...] = np.load(tile_filepath).astype(np.float32)
        if remainder == batch_size-1:
            # normalize
            X_batch -= channel_bounds[:, 0]
            X_batch /= (channel_bounds[:, 1]-channel_bounds[:, 0])
            X_batch = tf.convert_to_tensor(
                    value=X_batch,
                    dtype=tf.float32,
                    )
            yield (X_batch,)

def get_channel_bounds(
        tile_filepath_list: typing.List[str],
        ) -> np.ndarray:
    # paths
    input_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data'
    image_filename = '26531POST.ome.tif'
    channel_filename = 'channel_info.csv'
    image_filepath = os.path.join(input_folderpath, image_filename)
    channel_filepath = os.path.join(input_folderpath, channel_filename)
    # get channel index
    with open(channel_filepath, 'r') as ch:
        channel_name_list = ch.readlines()[0].split(',')
    biomarker_channels = [
            i for i, ch in enumerate(channel_name_list) if\
                    not ch.startswith('Hoechst')\
                    and ch not in ['A488', 'A555', 'A647']
                    ]
    first_hoechst_channel = [
            i for i, ch in enumerate(channel_name_list) if\
                    ch.startswith('Hoechst')
                    ]
    original_channel_list = first_hoechst_channel+biomarker_channels
    channel_list = [(t, o) for t, o in enumerate(original_channel_list)]
    # load WSI (Whole Slide Image)
    channel_bounds = np.zeros((len(channel_list), 2))
    with tifffile.TiffFile(image_filepath) as tif:
        for tile_channel, original_channel in channel_list:
            wsi = tif.asarray(series=0, key=original_channel)
            channel_bounds[tile_channel, 0] = wsi.min() # global min per channel
            channel_bounds[tile_channel, 1] = wsi.max() # global max per channel
    return channel_bounds

def load(
        batch_size: int, 
        train_fraction: float=0.007, 
        test_fraction: float=0.003,
        ) -> typing.Dict[str, typing.Any]:
    # paths
    tile_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/tiles_cleanChannel'
    # train-test split
    tile_filepath_list = [os.path.join(tile_folderpath, name) for name in os.listdir(tile_folderpath)\
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
    tile_shape = np.load(train_filepath_array[0]).shape
    channel_bounds = get_channel_bounds(tile_filepath_list=tile_filepath_list)
    train_generator_callable = lambda: tile_generator(
            batch_size=batch_size,
            tile_filepath_list=train_filepath_array,
            tile_shape=tile_shape,
            channel_bounds=channel_bounds,
            )
    test_generator_callable = lambda: tile_generator(
            batch_size=batch_size,
            tile_filepath_list=test_filepath_array,
            tile_shape=tile_shape,
            channel_bounds=channel_bounds,
            )
    # convert to tf.Dataset class
    train_dataset = tf.data.Dataset.from_generator(
            generator=train_generator_callable,
            output_types=(tf.Tensor,),
            output_shapes=(tf.TensorShape((BATCH_SIZE,)+tile_shape),),
            )
    test_dataset = tf.data.Dataset.from_generator(
            generator=test_generator_callable,
            output_types=(tf.Tensor,),
            output_shapes=(tf.TensorShape((BATCH_SIZE,)+tile_shape),),
            )
    data_dict = {
            'train_dataset':train_dataset,
            'test_dataset':test_dataset,
            'train_batch_count':train_count//batch_size,
            'test_batch_count':test_count//batch_size,
            'data_shape':tile_shape,
            }
    return data_dict
