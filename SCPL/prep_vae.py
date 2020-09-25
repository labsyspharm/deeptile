import os
import sys
import shutil

import numpy as np
import pandas as pd

if __name__ == '__main__':
    # paths
    data_folderpath = '/n/scratch2/hungyiwu/project_deeptile/data/'
    checktile_filepath = os.path.join(data_folderpath, 'check_tile_result',
            'checktile_45x45.csv')
    cellrecord_filepath = os.path.join(data_folderpath, 'cell_record.csv')
    imagerecord_filepath = os.path.join(data_folderpath, 'image_record.csv')
    scaledimage_folderpath = os.path.join(data_folderpath, 'scaled_images')
    output_folderpath = os.path.join(data_folderpath, 'tile_45x45')

    # filter criteria
#    cell_in_tile = 0.99
#    neighbor_in_tile = 0.5

    # load data
    checktile_df = pd.read_csv(checktile_filepath)
#    mask = (checktile_df['frac_cell_in_tile'] > cell_in_tile)\
#            & (checktile_df['frac_neighbor_in_tile'] < neighbor_in_tile)
#    checktile_df = checktile_df.loc[mask]
    num_cell = checktile_df.shape[0]
    num_sample = int(5e4)
    index_sample = np.random.choice(range(num_cell), size=num_sample, replace=False)
    checktile_df = checktile_df.iloc[index_sample]
    checktile_gb = checktile_df.groupby('key_id')

    image_record = pd.read_csv(imagerecord_filepath)
    filepath_dict = {key_id: os.path.join(scaledimage_folderpath, key+'.npy')\
            for key_id, key in zip(image_record['key_id'], image_record['key'])}

    cell_record = pd.read_csv(cellrecord_filepath)

    # store
    tile_shape = (45, 45)
    num_group = len(checktile_gb.groups)
    num_split = 16
    job_id = int(sys.argv[1])
    group_index = np.array_split(range(num_group), num_split)
    job_groups = [g for i, g in enumerate(checktile_gb.groups) if i in group_index[job_id]]
    for key_id in job_groups:
        df_1 = checktile_gb.get_group(key_id)[['cell_id']]
        df_2 = cell_record.loc[cell_record['key_id'] == key_id,
                ['cell_id', 'x_median', 'y_median']]
        df = df_1.merge(df_2, on='cell_id', how='left')
        coords = np.zeros((df.shape[0], 4), dtype=int)
        centroid = df[['x_median', 'y_median']].values
        coords[:, 0] = (centroid[:, 0] - tile_shape[0]/2).astype(int)
        coords[:, 1] = coords[:, 0] + tile_shape[0]
        coords[:, 2] = (centroid[:, 1] - tile_shape[1]/2).astype(int)
        coords[:, 3] = coords[:, 2] + tile_shape[1]
        cell_id = df['cell_id'].values

        image_filepath = filepath_dict[key_id]
        image = np.load(image_filepath)

        for i in range(coords.shape[0]):
            tile = image[coords[i, 0]:coords[i, 1], coords[i, 2]:coords[i, 3], :]
            output_filepath = os.path.join(output_folderpath,
                    '{}_{}.npy'.format(key_id, cell_id[i]))
            np.save(output_filepath, tile)
    print('all done.', flush=True)
