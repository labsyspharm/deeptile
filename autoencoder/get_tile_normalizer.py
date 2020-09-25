import numpy as np
import os
import typing
import multiprocessing
import argparse

import tifffile
import tqdm

def get_channel_normalizer(
        job: typing.Dict[str, typing.Any],
        ) -> typing.Dict[str, typing.Any]:
    # load WSI (Whole Slide Image)
    with tifffile.TiffFile(job['image_filepath']) as tif:
        wsi = tif.asarray(series=0, key=job['original_channel'])
        result_dict = {
                'tile_channel':job['tile_channel'],
                'mean':wsi.mean(),
                'std':wsi.std(),
                }
    return result_dict

if __name__ == '__main__':
    # get CPU count
    parser = argparse.ArgumentParser(description='Get CPU count.')
    parser.add_argument('-n', type=int)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    cpu_count = args.n
    verbose = args.verbose
    # paths
    input_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data'
    output_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output'
    image_filename = '26531POST.ome.tif'
    channel_filename = 'channel_info.csv'
    normalizer_filename = 'tile_normalizer.npy'
    image_filepath = os.path.join(input_folderpath, image_filename)
    channel_filepath = os.path.join(input_folderpath, channel_filename)
    normalizer_filepath = os.path.join(output_folderpath, normalizer_filename)
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
    # multiprocessing
    job_list = [{
        'image_filepath':image_filepath,
        'original_channel':o,
        'tile_channel':t,
        } for (t, o) in enumerate(original_channel_list)]
    normalizer = np.zeros((len(job_list), 2))
    worker_pool = multiprocessing.Pool(processes=cpu_count)
    for result in tqdm.tqdm(worker_pool.imap_unordered(
            func=get_channel_normalizer,
            iterable=job_list,
            ), total=len(job_list), disable=not verbose):
        normalizer[result['tile_channel'], 0] = result['mean']
        normalizer[result['tile_channel'], 1] = result['std']
    # save to disk
    np.save(normalizer_filepath, normalizer)
    print('Done.')

