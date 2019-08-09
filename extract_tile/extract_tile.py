import os
import numpy as np
import csv
import shutil
import typing
import multiprocessing

import tifffile

# define ROI
def within_ROI(
        xy: typing.Tuple[int, int], 
        target_ROI: typing.Dict[str, int],
        ) -> bool:
    return (xy[0] >= target_ROI['x']-target_ROI['width']/2)\
        and (xy[0] <= target_ROI['x']+target_ROI['width']/2)\
        and (xy[1] >= target_ROI['y']-target_ROI['height']/2)\
        and (xy[1] <= target_ROI['y']+target_ROI['height']/2)

# parse histoCAT output .csv file
def get_cell_list(
        csv_filepath: str,
        target_ROI: typing.Dict[str, int],
        ) -> typing.List[typing.Dict[str, int]]:
    cell_list = []
    with open(csv_filepath, newline='') as csvfile:
        line_reader = csv.reader(csvfile, delimiter=',')
        header = next(line_reader)
        cellid_pos = header.index('CellId')
        x_pos = header.index('X_position')
        y_pos = header.index('Y_position')
        for row in line_reader:
            current_cell_x = int(float(row[x_pos]))
            current_cell_y = int(float(row[y_pos]))
            if within_ROI((current_cell_x, current_cell_y), target_ROI):
                cell_dict = {
                    'CellId':int(row[cellid_pos]),
                    'X_position':current_cell_x,
                    'Y_position':current_cell_y,
                }
                cell_list.append(cell_dict)
    return cell_list

# slice and save tile to disk
def save_tile(
        job: typing.Dict[str, typing.Any],
        ) -> None:
    # unpack input
    image_filepath = job['image_filepath']
    output_folderpath = job['output_folderpath']
    tile_channel, original_channel = job['channel']
    cell_list = job['cell_list']
    tile_width = job['tile_width']
    # create sub-directory
    channel_folderpath = os.path.join(output_folderpath, 'tile_channel_{}'.format(tile_channel))
    if os.path.isdir(channel_folderpath):
        shutil.rmtree(channel_folderpath)
    os.mkdir(channel_folderpath)
    # open image
    with tifffile.TiffFile(image_filepath) as tif:
        # wsi = Whole Slide Image
        wsi = tif.asarray(series=0, key=original_channel)
        # normalization by channel
        wsi -= wsi.mean()
        wsi /= wsi.std()
        # loop through cells
        for cell in cell_list:
            tile = wsi[
                    cell['Y_position']-tile_width//2:cell['Y_position']+tile_width//2,
                    cell['X_position']-tile_width//2:cell['X_position']+tile_width//2]
            # save to disk
            tile_filepath = os.path.join(channel_folderpath, 'cell_{}.npy'.format(cell['CellId']))
            np.save(tile_filepath, tile)
    return

# tile job generator
def tile_job_generator(
        image_filepath: str,
        output_folderpath: str,
        channel_list: typing.List[typing.Tuple[int, int]],
        cell_list: typing.List[typing.Dict[str, int]],
        tile_width: int,
        ) -> typing.Dict[str, typing.Any]:
    for channel in channel_list:
        yield {
                'image_filepath':image_filepath,
                'output_folderpath':output_folderpath,
                'channel':channel,
                'cell_list':cell_list,
                'tile_width':tile_width,
                }

# assemble tiles for each cell
def assemble_cell(
        job: typing.Dict[str, typing.Any],
        ) -> None:
    # unpack input
    output_folderpath = job['output_folderpath']
    channel_list = job['channel_list']
    cell = job['cell']
    # loop over channels
    tile_list = []
    for tile_channel, _ in channel_list:
        tile_path = os.path.join(
                output_folderpath,
                'tile_channel_{}'.format(tile_channel),
                'cell_{}.npy'.format(cell['CellId']),
                )
        tile = np.load(tile_path)
        tile_list.append(tile)
    # save assembled cell to disk
    tile_stack = np.stack(tile_list, axis=-1)
    tile_stack_filepath = os.path.join(output_folderpath, 'cell_{}.npy'.format(cell['CellId']))
    np.save(tile_stack_filepath, tile_stack)
    return

# cell job generator
def cell_job_generator(
        output_folderpath: str,
        channel_list: typing.List[typing.Tuple[int, int]],
        cell_list: typing.List[typing.Dict[str, int]],
        ) -> typing.Dict[str, typing.Any]:
    for cell in cell_list:
        yield {
                'output_folderpath':output_folderpath,
                'channel_list':channel_list,
                'cell':cell,
                }

# main function
if __name__ == '__main__':
    # paths and names
    input_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data'
    output_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output'
    image_filename = '26531POST.ome.tif'
    histoCAT_filename = '26531POST.csv'
    channel_filename = 'channel_info.csv'
    tile_foldername = 'tiles_normalized'
    image_filepath = os.path.join(input_folderpath, image_filename)
    histoCAT_filepath = os.path.join(input_folderpath, histoCAT_filename)
    channel_filepath = os.path.join(input_folderpath, channel_filename)
    tile_folderpath = os.path.join(output_folderpath, tile_foldername)
    # create output folder
    if os.path.isdir(tile_folderpath):
        shutil.rmtree(tile_folderpath)
    os.mkdir(tile_folderpath)
    # target ROI obtained from PathViewer on OMERO server
    target_ROI = {
        'x':23969,
        'y':9398,
        'width':5932,
        'height':5170,
    }
    # define lesion tile size
    displacement_cells = 10 # width/2, in unit of cells
    cell_size = 10 # in unit of micro-meter
    pixel_size = 0.65 # in unit of micro-meter
    tile_width = 2*int(displacement_cells * cell_size / pixel_size) # in unit of pixel
    # print metadata of image
    with tifffile.TiffFile(image_filepath) as tif:
        # preview file size
        print('Loading image metadata...')
        arr = tif.asarray(series=0, key=0)
        arr_size_gb = arr.nbytes/1024**3
        count_page = len(tif.series[0].pages)
        print('file path: {}\n'
                'successfully loaded series 0 slice 0\n'
                'numpy.array shape={}, size {:.2f} GB\n'
                '{} arrays, {:.2f} GB in total'\
                .format(image_filepath,\
                    str(arr.shape), arr_size_gb,\
                    count_page, count_page*arr_size_gb))
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
    # load single cell positions and filter with ROI
    print('Loading and filtering single cell positions...')
    cell_list = get_cell_list(
            csv_filepath=histoCAT_filepath,
            target_ROI=target_ROI,
            )
    print('{} cells within ROI found.'.format(len(cell_list)))
    # slice and save tiles
    print('Saving tiles to folder {}'.format(tile_folderpath))
    worker_pool = multiprocessing.Pool(20)
    # one job per channel
    job_iterator = worker_pool.imap_unordered(
            save_tile,
            tile_job_generator(
                image_filepath=image_filepath,
                output_folderpath=tile_folderpath,
                channel_list=channel_list,
                cell_list=cell_list,
                tile_width=tile_width,
                )
            )
    for _ in job_iterator:
        pass
    print('Assembling tiles...')
    # assemble tiles for each cell
    job_iterator = worker_pool.imap_unordered(
            assemble_cell,
            cell_job_generator(
                output_folderpath=tile_folderpath,
                channel_list=channel_list,
                cell_list=cell_list,
                )
            )
    for _ in job_iterator:
        pass
    # clean-up
    for tile_channel, _ in channel_list:
        tile_channel_folderpath = os.path.join(
                tile_folderpath, 
                'tile_channel_{}'.format(tile_channel),
                )
        shutil.rmtree(tile_channel_folderpath)
    print('Done.')
