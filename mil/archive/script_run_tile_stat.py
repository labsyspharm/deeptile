import numpy as np
import pandas as pd
import os
import argparse
import yaml

import tensorflow as tf
import tqdm

import deeptile_dataset
import deeptile_model
import deeptile_stat

if __name__ == '__main__':
    # parameter defined within the scope of model preparation
    CLUSTER_NUM = 5
    # load configuration
    workspace_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/workspace'
    config_filepath = os.path.join(workspace_folderpath, 'default_config.yaml')
    with open(config_filepath, 'r') as yaml_file:
        config_dict = yaml.safe_load(yaml_file)
    # paths
    model_filepath = os.path.join(workspace_folderpath, 'CVAE_model.hdf5'),
    record_filepath = os.path.join(workspace_folderpath, 'analysis_result.csv')
    # parse arguments
    parser = argparse.ArgumentParser(description='Get verbosity.')
    parser.add_argument('--verbose', action='store_true', # default is False
            help='Turn on tqdm progress bar.')
    args = parser.parse_args()
    verbose = args.verbose
    if verbose: print('logistics done')
    # sliding window
    center_list = []
    for x in range(
            start=config_dict['support_range'][0][0],
            stop=config_dict['support_range'][0][1]):
        for y in range(
                start=config_dict['support_range'][1][0],
                stop=config_dict['support_range'][1][1]):
            point = (int(x+config_dict['tile_shape'][0]/2), 
                    int(y+config_dict['tile_shape'][1]/2))
            if deeptile_dataset.within_range(
                    point=point,
                    tile_shape=config_dict['tile_shape'],
                    support_range=config_dict['support_range']):
                center_list.append(point)
    # load model
    model = tf.keras.models.load_model(model_filepath)
    # clustering
    cluster_label = deeptile_stat.clustering(
            sample_loader=loader,
            embedding_model=model,
            center_list=center_list,
            tile_shape=config_dict['tile_shape'],
            n_cluster=CLUSTER_NUM,
            )
    # data
    loader = deeptile_dataset.tile_loader(
            workspace_folderpath=workspace_folderpath,
            warm_start=True,
            image_filepath=config_dict['image_filepath'],
            channel_filepath=config_dict['channel_info_filepath'],
            )
    # loop
    record = []
    for tile_index, tile in tqdm.tqdm(
            iterable=enumerate(loader.generate_tiles(
                center_list=center_list,
                tile_shape=config_dict['tile_shape'],
                )),
            desc='main',
            disable=not verbose,
            ):
        significance = deeptile_stat.permutation_significance(
                embedding_model=model,
                tile=tile,
                )
        uniqueness = deeptile_stat.relative_uniqueness(
                embedding_model=model,
                sample_loader=loader,
                center_list=center_list,
                tile_shape=config_dict['tile_shape'],
                target_tile_index=tile_index,
                )
        center = center_list[tile_index]
        record.append([center[0], center[1], significance, uniqueness, cluster_label[tile_index]])
    # save result
    record_df = pd.DataFrame.from_records(record, 
            columns=['x', 'y', 'significance', 'uniqueness', 'cluster_label'])
    record_df.to_csv(record_filepath, index=False)
    print('Done.')
