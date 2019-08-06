import tifffile
import numpy as np
import os
import shutil

@profile
def tifffile_fn(
        image_filepath: str,
        output_folderpath: str) -> None:
    temp_filepath = os.path.join(output_folderpath, 'test.npy')
    for ch in range(10):
        with tifffile.TiffFile(image_filepath) as tif:
            wsi = tif.asarray(series=0, key=ch)
            for j in range(100000):
                x = wsi[j:j+200, j:j+200]
                np.save(temp_filepath, x)
    return

if __name__ == '__main__':
    image_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data/26531POST.ome.tif'
    output_folderpath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/output/temp'
    if os.path.isdir(output_folderpath):
        shutil.rmtree(output_folderpath)
    os.mkdir(output_folderpath)
    tifffile_fn(
            image_filepath=image_filepath,
            output_folderpath=output_folderpath,
            )
    print('Done.')
