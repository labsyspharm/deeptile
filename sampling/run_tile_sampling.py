import numpy as np
import pandas as pd
import time
import argparse
import os
import copy

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
    # note: input is based on image coordinate
    ROI = {
        'image_x':23969,
        'image_y':9398,
        'image_width':5932,
        'image_height':5170,
    }
    ROI_x_low = int(ROI['image_x']-ROI['image_width']/2+tile_shape[0]/2)
    ROI_x_high = int(ROI['image_x']+ROI['image_width']/2-tile_shape[0]/2)
    ROI_y_low = int(ROI['image_y']-ROI['image_height']/2+tile_shape[1]/2)
    ROI_y_high = int(ROI['image_y']+ROI['image_height']/2-tile_shape[1]/2)
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
            optimizer=tf.keras.optimizers.Adam(learning_rate=1e-6),
            )
    # training parameters
    batch_size = 10
    # define grid
    x_linspace = np.linspace(
            start=ROI_support_range[0][0], 
            stop=ROI_support_range[0][1], 
            num=int(ROI['image_width']/tile_shape[0]*2),
            ).astype(int)
    y_linspace = np.linspace(
            start=ROI_support_range[1][0], 
            stop=ROI_support_range[1][1], 
            num=int(ROI['image_height']/tile_shape[1]*2),
            ).astype(int)
    x_mesh, y_mesh = np.meshgrid(x_linspace, y_linspace)
    survey_grid = [(x,y) for x, y in zip(x_mesh.flatten(), y_mesh.flatten())]
    # keep only those within image
    survey_grid = loader.within_image(
            tile_shape=tile_shape,
            center_list=survey_grid,
            )
    survey_grid_array = np.zeros((len(survey_grid), 2))
    for index, (x,y) in enumerate(survey_grid):
        survey_grid_array[index, 0] = x
        survey_grid_array[index, 1] = y
    # convert to dataset
    grid_params = {
            'tile_shape':tile_shape,
            'center_list':survey_grid,
            'batch_size':batch_size,
            }
    grid_dataset = loader.get_dataset(**grid_params)
    grid_batch_count = np.ceil(len(survey_grid)/batch_size).astype(int)
    # setup training, evaluation, sampling loop
    train_sample = copy.deepcopy(survey_grid)
    total_cycle = 3
    record = []
    for cycle in range(total_cycle):
        ts_start = time.time()
        # phase 1: train on sample
        train_sample_count = len(train_sample)
        sample_params = {
                'tile_shape':tile_shape,
                'center_list':train_sample,
                'batch_size':batch_size,
                }
        sample_dataset = loader.get_dataset(**sample_params)
        sample_batch_count = np.ceil(len(train_sample)/batch_size).astype(int)
        for batch_tile in tqdm.tqdm(
                iterable=sample_dataset,
                desc='train',
                total=sample_batch_count,
                disable=not verbose):
            cvae_model.compute_apply_gradients(batch_tile)
        # phase 2: survey on grid
        loss_list = []
        for batch_tile in tqdm.tqdm(
                iterable=grid_dataset,
                desc='survey',
                total=grid_batch_count,
                disable=not verbose):
            loss = cvae_model.compute_loss(batch_tile)
            loss_list.append(loss.numpy().flatten())
        prob = np.hstack(loss_list)
        prob /= prob.sum() # note that regression loss >= 0
        # phase 3: generate next training sample
        if verbose:
            print('Multivariate inverse transform sampling...')
        train_sample_array = deeptile_sampling.multivariate_inverse_transform_sampling(
            data_X=survey_grid_array,
            data_prob=prob,
            sample_size=len(survey_grid),
            support_range=ROI_support_range,
            grid_count=100,
            )
        train_sample = []
        for index in range(train_sample_array.shape[0]):
            x = int(train_sample_array[index, 0])
            y = int(train_sample_array[index, 1])
            train_sample.append((x, y))
        train_sample = loader.within_image(
                tile_shape=tile_shape,
                center_list=train_sample,
                )
        ts_end = time.time()
        # report progress
        mean_loss = np.mean(np.hstack(loss_list))
        runtime = ts_end-ts_start
        print('cycle {} done, {} samples, grid loss {:.3f}, runtime {:.3f} sec.'.format(
            cycle, train_sample_count, mean_loss, runtime))
        # record history
        record.append([cycle, mean_loss, runtime])
    # save record to disk
    df = pd.DataFrame.from_records(record, columns=['cycle', 'loss', 'runtime (sec)'])
    df.to_csv(record_filepath)
    print('Done.')

