import skimage.external.tifffile
import tifffile
import time

def skimage_fn(image_filepath: str, key_list: list) -> None:
    with skimage.external.tifffile.TiffFile(image_filepath) as tif:
        wsi = tif.asarray(series=0, key=key_list)
    return

def tifffile_fn(image_filepath: str, key_list: list) -> None:
    with tifffile.TiffFile(image_filepath) as tif:
        wsi = tif.asarray(series=0, key=key_list)
    return

if __name__ == '__main__':
    image_filepath = '/n/scratch2/hungyiwu/deeptile_data/26531POST/input_data/26531POST.ome.tif'
    test_fn = {
            'skimage':skimage_fn,
            'tifffile':tifffile_fn,
            }
    key_len = {
            '1':[0],
            '3':list(range(3)),
            }
    for fn_name in test_fn:
        for key in key_len:
            ts_start = time.time()
            test_fn[fn_name](
                    image_filepath=image_filepath,
                    key_list=key_len[key],
                    )
            ts_end = time.time()
            print('fn: {}, #channel: {}, runtime: {:.3f}'.format(fn_name, key, ts_end-ts_start))
    print('Done.')
