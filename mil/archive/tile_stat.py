import numpy as np
import scipy.stats
import typing

import sklearn.mixture

import deeptile_dataset

def shuffle_along_axis(a, axis):
    '''
    From StackOverflow https://tinyurl.com/yys26oqe
    '''
    idx = np.random.rand(*a.shape).argsort(axis=axis)
    return np.take_along_axis(a,idx,axis=axis)

def clustering(
        sample_loader,
        embedding_model,
        center_list: typing.List[typing.Tuple[int, int]],
        tile_shape: typing.Tuple[int, int],
        n_cluster: int,
        ):
    result = []
    for index, center in enumerate(center_list):
        tile = sample_loader.get_tile(
                center=center,
                tile_shape=tile_shape,
                )
        mean, logvar = embedding_model.encode(tile[np.newaxis, ...])
        embedding = embedding_model.reparameterize(mean, logvar)
        result.append(embedding)
    result = np.vstack(result)
    model = sklearn.mixture.GaussianMixture(n_components=n_cluster)
    cluster_label = model.fit_predict(result)
    return cluster_label

def permutation_significance(embedding_model, tile, total_sample=1000):
    result = []
    for _ in range(total_sample):
        sample = shuffle_along_axis(tile, axis=0) # x-axis
        sample = shuffle_along_axis(sample, axis=1) # y-axis
        mean, logvar = embedding_model.encode(sample[np.newaxis, ...])
        embedding = embedding_model.reparameterize(mean, logvar)
        result.append(embedding)
    result = np.vstack(result)
    # multivariate p-value
    # assuming each embedding dimension follows Gaussian distribution
    mvn = scipy.stats.multivariate_normal(
            mean=result.mean(axis=0),
            cov=np.cov(result),
            )
    mean, logvar = embedding_model.encode(tile[np.newaxis, ...])
    tile_embedding = embedding_model.reparameterize(mean, logvar)
    return mvn.logpdf(tile_embedding)

def relative_uniqueness(
        embedding_model,
        sample_loader,
        center_list: typing.List[typing.Tuple[int, int]],
        tile_shape=typing.Tuple[int, int],
        target_tile_index=int,
        ):
    target_tile = sample_loader.get_tile(
            center=center_list[target_tile_index],
            tile_shape=tile_shape,
            )
    mean, logvar = embedding_model.encode(target_tile[np.newaxis, ...])
    target_embedding = embedding_model.reparameterize(mean, logvar)
    distance = 0.
    for index, center in enumerate(center_list):
        tile = sample_loader.get_tile(
                center=center,
                tile_shape=tile_shape,
                )
        mean, logvar = embedding_model.encode(tile[np.newaxis, ...])
        embedding = embedding_model.reparameterize(mean, logvar)
        distance += np.linalg.norm(target_embedding-embedding, ord=2)
    return distance/len(center_list)

