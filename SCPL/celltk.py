import numpy as np

def iter_cell(cellid_array):
    '''
    Iterate over (x,y) coordinates given a cell ID mask.
    '''
    # flatten
    cellid = cellid_array.flatten()
    pixelid = np.arange(cellid.shape[0], dtype=int)
    # by convention, background has value of zero
    mask = cellid > 0
    cellid, pixelid = cellid[mask], pixelid[mask]
    # sort by cell ID
    sortkey = np.argsort(cellid)
    cellid, pixelid = cellid[sortkey], pixelid[sortkey]
    unique_cellid, count_cellid = np.unique(cellid, return_counts=True)
    cut_point = np.cumsum(count_cellid)
    sequence = zip(unique_cellid, np.split(pixelid, cut_point))
    # store
    cellcoord = []
    for ci, pi in sequence:
        xi, yi = np.unravel_index(pi, cellid_array.shape)
        yield ci, xi, yi

def percentile_scale(
    image_in: np.ndarray,
    p_low: float=1.,
    p_high: float=99.):
    '''
    Scale an image using a percentile approach.
    
    reference: https://stackoverflow.com/questions/9744255/instagram-lux-effect/9761841#9761841
    
    Args:
        image_in (np.ndarray): image to be adjusted. 2D image with single color channel.
        p_low (float): percentile of the lower bound. Intensities whose percentiles lower than
            this number will be scaled to zero. Default to 1.
        p_high (float): percentile of the upper bound. Intensities whose percentiles higher than
            this number will be scaled to one. Default to 99.
            
    Returns:
        new_image (np.ndarray): a copy of image_in but scaled to [0, 1]. 
    '''
    # cast the input image to float type
    image = image_in.astype(float)
    # calculate the percentiles
    I_low = np.percentile(image, p_low)
    I_high = np.percentile(image, p_high)
    # formulate the systems of equations
    m = np.array([[I_low, 1], [I_high, 1]])
    y = np.array([0, 1])
    # solve the systems of equations
    x = np.linalg.solve(m, y)
    # unpack the coefficients
    contrast_coef = x[0]
    intensity_coef = x[1]
    # scale the image according to the coefficients determined
    new_image = contrast_coef * image + intensity_coef
    new_image[new_image > 1] = 1
    new_image[new_image < 0] = 0
    return new_image
