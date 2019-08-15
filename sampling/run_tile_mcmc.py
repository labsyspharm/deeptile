import numpy as np
import pandas as pd
import time
import argparse
import os
import copy
import typing

import tqdm
import tensorflow as tf

'''
# turn on memory growth so GPU memory allocation becomes as-needed
# for cases when training takes too much memory.
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
'''

import deeptile_model
import deeptile_dataset
import deeptile_sampling

if __name__ == '__main__':
    # define tile shape
    # note: shape is based on image coordinate
    displacement_cells = 10 # width/2, in unit of cells
    cell_size = 10 # in unit of micro-meter
    pixel_size = 0.65 # in unit of micro-meter
    tile_width = 2*int(displacement_cells * cell_size / pixel_size) # in unit of pixel
    tile_shape = (tile_width, tile_width)
    # paths
    image_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'input_data/26531POST.ome.tif'
    channel_info_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'input_data/channel_info.csv'
    workspace_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/'\
            'output/workspace'
    record_filepath = './training_history.csv'
    # target ROI obtained from PathViewer on OMERO server
    ROI = {
        'image_x':23969,
        'image_y':9398,
        'image_width':5932, # delta-y
        'image_height':5170, # delta-x
    }
    ROI_x_low = int(ROI['image_x']-ROI['image_height']/2)
    ROI_x_high = int(ROI['image_x']+ROI['image_height']/2)
    ROI_y_low = int(ROI['image_y']-ROI['image_width']/2)
    ROI_y_high = int(ROI['image_y']+ROI['image_width']/2)
    ROI_support_range = [(ROI_x_low, ROI_x_high), (ROI_y_low, ROI_y_high)]
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    verbose = args.verbose
    # data
    loader = deeptile_dataset.tile_loader(
            workspace_folderpath=workspace_folderpath,
            warm_start=True,
            image_filepath=image_filepath,
            channel_filepath=channel_info_filepath,
            )
    # model
    cvae_model = deeptile_model.CVAE(
            latent_dim=20, 
            input_shape=tile_shape+(loader.count_channel,),
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
            )
    # stepping function with normal prior
    def __random_step(current_point):
        next_x = np.random.normal(loc=0, scale=tile_shape[0])
        next_y = np.random.normal(loc=0, scale=tile_shape[1])
        next_point = (current_point[0]+int(next_x), current_point[1]+int(next_y))
        next_tile = loader.get_tile(
                tile_shape=tile_shape,
                center=next_point,
                need_validate=True,
                )
        return next_point, next_tile
    def random_step(current_point):
        next_point, next_tile = __random_step(current_point)
        while next_tile is None:
            next_point, next_tile = __random_step(current_point)
        next_tile = next_tile[np.newaxis, ...]
        return next_point, next_tile
    # MCMC loop
    batch_size = 100
    ts_start = time.time()
    total_step = int(1e3)
    current_point = (ROI['image_x'], ROI['image_y'])
    record = []
    for step in tqdm.tqdm(
            iterable=range(total_step),
            desc='MCMC',
            disable=not verbose):
        # phase 1: train on current point
        current_tile = loader.get_tile(
                tile_shape=tile_shape,
                center=current_point,
                need_validate=True,
                )
        current_tile = current_tile[np.newaxis, ...]
        cvae_model.compute_apply_gradients(current_tile)
        # phase 2: pick next point based on prior
        next_point, next_tile = random_step(current_point)
        # phase 3: compare loss and make step decision
        current_loss = cvae_model.compute_loss(current_tile).numpy()[0]
        next_loss = cvae_model.compute_loss(next_tile).numpy()[0]
        ratio = min(1, next_loss/current_loss)
        if np.random.rand() < ratio:
            current_point = next_point
        if step % batch_size == 0:
            ts_end = time.time()
            print('step {} current_point {} current_loss {} runtime {:.3f}/step'.format(
                step, current_point, current_loss, (ts_end-ts_start)/batch_size))
            ts_start = time.time()
    print('Done.')

