import numpy as np
import typing

import scipy.interpolate

def univariate_inverse_transform_sampling(
        CDF: typing.Callable[[np.ndarray], np.ndarray],
        support_range: typing.Tuple[float, float],
        grid_count: int,
        sample_size: int,
        ) -> np.ndarray:
    # evaluate CDF at grid
    grid = np.linspace(start=support_range[0], stop=support_range[1], num=grid_count)
    grid_cum_prob = CDF(grid)
    # in case of NaN
    keep_index = np.isfinite(grid_cum_prob)
    if keep_index.sum() == 0:
        return np.full(sample_size, float('nan'))
    grid = grid[keep_index]
    grid_cum_prob = grid_cum_prob[keep_index]
    # uniform sample in range [0,1)
    uniform_sample = np.random.rand(sample_size)
    uniform_sample *= (grid_cum_prob.max()-grid_cum_prob.min())
    uniform_sample += grid_cum_prob.min()
    # use scipy.interpolate.interp1d to generate the inversed CDF function
    CDF_inversed = scipy.interpolate.interp1d(grid_cum_prob, grid)
    # convert uniform sample to samples on support domain by the inversed CDF
    sample_X = CDF_inversed(uniform_sample)
    return sample_X

def marginal_conditional_CDF(
        data_X: np.ndarray,
        data_prob: np.ndarray,
        target_axis: int,
        given_axis: typing.List[int],
        ) -> typing.Callable[[np.ndarray, np.ndarray], np.ndarray]:
    '''
    Get marginal conditional CDF from data points.
    Any axis other than target_axis and given_axis will be marginalized.
    Use np.cumsum for marginalization, scipy.interpolate.interp1d for conditioning.
    prob = \int_{y_low}^{y_0} PDF(x_0,y) dy
    '''
    # normalize value to get PDF
    marginal_prob = data_prob.copy() / data_prob.sum()
    # marginalization
    all_axis = set(np.arange(data_X.shape[1], dtype=int))
    axis_map = sorted(list(given_axis)+[target_axis])
    marginal_axis = all_axis.difference(set(axis_map))
    for axis in marginal_axis:
        # CDF goes with ascending axis value
        sort_key = np.argsort(data_X[:, axis])
        marginal_prob = marginal_prob[sort_key]
        # cumulative sum
        marginal_prob = np.cumsum(marginal_prob)
        # resume original ordering
        marginal_prob = marginal_prob[sort_key]
    # construct conditional CDF
    def CDF(xi: np.ndarray, given_X: np.ndarray=None):
        '''
        given_X is np.ndarray with shape (len(given_axis_list),)
        xi is np.ndarray with shape (-1,)
        '''
        # construct input
        if given_axis:
            x_input = np.zeros((xi.shape[0], given_X.shape[0]+1))
            given_X_axis_head = 0
            for xi_axis, original_axis in enumerate(axis_map):
                if original_axis == target_axis:
                    x_input[:, xi_axis] = xi
                else:
                    x_input[:, xi_axis] = given_X[given_X_axis_head]
                    given_X_axis_head += 1
        else:
            x_input = xi
        # multivariate interpolation from unstructured data
        prob = scipy.interpolate.griddata(
                points=data_X[:, axis_map],
                values=marginal_prob,
                xi=x_input,
                method='nearest',
                )
        return prob
    return CDF

def multivariate_inverse_transform_sampling(
        data_X: np.ndarray, 
        data_prob: np.ndarray,
        sample_size: int,
        support_range: typing.List[typing.Tuple[float, float]],
        grid_count: int,
        ) -> typing.List[typing.List[int]]:
    '''
    Inverse transform sampling:
    1. data -> PDF -> CDF -> inversed CDF
    2. generate uniform samples of the probability
    3. feed probability to inversed CDF to get samples of the coordinate

    This doesn't work for multivariate CDF
    because now one probability corresponds to multiple coordinates
    (degeneracy, or contours on the multidimentional CDF)

    Algorithm for multivariate:
    (from https://tinyurl.com/y45wth7r)
    1. get marginal CDF of x. ie.
    def CDF_1(x_0):
        prob = \int_{y_low}^{y_up} \int_{x_low}^{x_0} PDF(x,y) dx dy
        return prob
    2. do inverse transform sampling on this CDF and get samples of x.
    3. get marginal CDF of y given x. ie.
    def CDF_2(y_0, x_0) return 
       prob = \int_{y_low}^{y_0} PDF(x_0,y) dy
       return prob
    4. for each sample x obtained in step 2, plug in the sample value
       and get a CDF of y **for this sample point**.
    5. do inverse transform sampling on each CDF and get
       **one sample value of y** for this sample point.
    6. Done.
    '''
    axes = np.arange(data_X.shape[1], dtype=int)
    given_axis_list = []
    sample_X = np.zeros((sample_size, data_X.shape[1]))
    for axis in axes:
        mcCDF = marginal_conditional_CDF(
                data_X=data_X,
                data_prob=data_prob,
                target_axis=axis,
                given_axis=given_axis_list,
                )
        for sample_index in range(sample_size):
            given_X = sample_X[sample_index, given_axis_list]
            try:
                '''
                scipy.interpolate.NearestNDInterpolator uses cKDTree by default
                I manually switched to KDTree (Python implementation) and got the same error
                After digging into KDTree code I found there's a parameter called distance_upper_bound
                This parameter defaults to np.inf but somehow in the algorithm design it decreases
                Therefore there can be cases of "missing neighbor" even with "nearest neighbor algorithm"
                In this case the returned neighbor index is N when argument xi has shape (N, D)
                This is the reason why I saw this error

                    IndexError: index N is out of bounds for axis 0 with size N

                sometimes.
                '''
                sample_X[sample_index, axis] = univariate_inverse_transform_sampling(
                        CDF=lambda xi: mcCDF(xi, given_X),
                        support_range=support_range[axis],
                        grid_count=grid_count,
                        sample_size=1,
                        )
            except:
                sample_X[sample_index, axis] = float('nan')
        given_axis_list.append(axis)
    return sample_X

