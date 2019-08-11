import os
import numpy as np
import csv
import shutil
import typing
import multiprocessing

import tifffile
import tensorflow as tf

class tile_extractor(object):
    def __init__(
            self,
            image_filepath: str,
            channel_info_filepath: str,
            tile_width: int, # square tile for now
            cpu_count: int,
        ) -> None:
        super().__init__()
        # populate attributes
        self.tif = tifffile.TiffFile(image_filepath)
        self.image_shape = self.tif.asarray(series=0, key=0).T.shape
        self.half_tile_width = tile_width//2
        self.worker_pool = multiprocessing.Pool(processes=cpu_count)
        # get channel index
        with open(channel_info_filepath, 'r') as ch:
            channel_name_list = ch.readlines()[0].split(',')
        self.original_channel_list = []
        first_hoechst = False
        for original_channel, channel_name in enumerate(channel_name_list):
            if not first_hoechst and channel_name.startswith('Hoechst'):
                self.original_channel_list.append(original_channel)
                first_hoechst = True
            elif not channel_name.startswith('Hoechst') and channel_name not in ['A488', 'A555', 'A647']:
                self.original_channel_list.append(original_channel)
        # data shape for initializing model
        self.data_shape = (self.half_tile_width*2, self.half_tile_width*2, len(self.original_channel_list))
        return

    def within_image(self, pixel_coordinate: typing.Tuple[int, int]) -> bool:
        x_pos, y_pos = pixel_coordinate
        criteria = [
                x_pos-self.half_width >= 0,
                x_pos+self.half_width < self.image_shape[0],
                y_pos-self.half_width >= 0,
                y_pos+self.half_width < self.image_shape[1],
                ]
        return all(criteria)

    def slice_job(
            self,
            sample_X: np.ndarray,
            output_folderpath: str,
            ) -> typing.Dict[str, typing.Any]:
        for tile_channel, original_channel in enumerate(self.original_channel_list):
            yield {'tile_channel':tile_channel,
                    'original_channel':original_channel,
                    'sample_X':sample_X,
                    'output_folderpath':output_folderpath,
                    }

    def slice_process(
            self,
            job: typing.Dict[str, typing.Any],
            ) -> None:
        # create sub-directory
        channel_folderpath = os.path.join(
                job['output_folderpath'], 
                'channel_{}'.format(job['tile_channel']))
        if os.path.isdir(channel_folderpath):
            shutil.rmtree(channel_folderpath)
        os.mkdir(channel_folderpath)
        # wsi = Whole Slide Image
        wsi = self.tif.asarray(series=0, key=job['original_channel']).T
        # normalization by channel
        wsi -= wsi.mean()
        wsi /= wsi.std()
        # loop through cells
        for index in range(job['sample_X'].shape[0]):
            x_pos = job['sample_X'][index, 0]
            y_pos = job['sample_X'][index, 1]
                if self.within_image((x_pos, y_pos))
                    tile = wsi[
                            x_pos-self.half_width:x_pos+self.half_width,
                            y_pos-self.half_width:y_pos+self.half_width]
                    # save to disk
                    tile_filepath = os.path.join(channel_folderpath, 'tile_{}.npy'.format(index))
                    np.save(tile_filepath, tile)
        return

    def assemble_job(
            self,
            output_folderpath: str,
            tile_count: int,
            ) -> typing.Dict[str, typing.Any]:
        for tile_index in range(tile_count):
            yield {
                    'output_folderpath':output_folderpath,
                    'tile_index':tile_index,
                    }

    def assemble_process(
            self,
            job: typing.Dict[str, typing.Any],
            ) -> None:
        # loop over channels
        tile_list = []
        for tile_channel in range(len(self.original_channel_list)):
            tile_path = os.path.join(
                    job['output_folderpath'],
                    'channel_{}'.format(tile_channel),
                    'tile_{}.npy'.format(job['tile_index']),
                    )
            tile = np.load(tile_path)
            tile_list.append(tile)
        # save assembled cell to disk
        tile_stack = np.stack(tile_list, axis=-1)
        tile_stack_filepath = os.path.join(
                job['output_folderpath'], 
                'tile_{}.npy'.format(job['tile_index']))
        np.save(tile_stack_filepath, tile_stack)
        return

    def extract(
            self,
            sample_X: np.ndarray,
            output_folderpath: str,
            ) -> None:
        # create output folder
        if os.path.isdir(output_folderpath):
            shutil.rmtree(output_folderpath)
        os.mkdir(output_folderpath)
        # validate samples
        sample_X_valid = []
        for index in range(sample_X.shape[0]):
            x_pos = sample_X[index, 0]
            y_pos = sample_X[index, 1]
            if self.within_image((x_pos, y_pos)):
                sample_X_valid.append(index)
        sample_X_valid = sample_X[np.array(sample_X_valid), :]
        # slicing loop
        for _ in self.worker_pool.imap_unordered(
                func=self.slice_process,
                iterable=self.slice_job(
                    sample_X=sample_X_valid,
                    output_folderpath=output_folderpath,
                    ),
                ):
            pass
        # assembling loop
        for _ in self.worker_pool.imap_unordered(
                func=self.assemble_process,
                iterable=self.assemble_job(
                    output_folderpath=otuput_folderpath,
                    tile_count=sample_X_valid.shape[0],
                    ),
                ):
            pass
        return

class tile_loader(object):
    def __init__(
            self,
            tile_folderpath: str,
            data_shape: typing.Tuple[int, int, int],
            ) -> None:
        self.tile_folderpath = tile_folderpath
        self.data_shape = data_shape
        return

    def generator(self) -> np.ndarray:
        tile_filepath_list = []
        for name in os.listdir(self.tile_folderpath):
            if name.endswith('.npy'):
                tile_filepath_list.append(os.path.join(self.tile_folderpath, name))
        for tile_filepath in tile_filepath_list:
            yield np.load(tile_filepath).astype(np.float32)

    def get_dataset(self, batch_size: int) -> tf.Dataset:
        return tf.data.Dataset.from_generator(
                generator=self.generator,
                output_types=np.float32,
                output_shapes=self.data_shape,
                ).batch(batch_size)

