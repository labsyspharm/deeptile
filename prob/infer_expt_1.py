import numpy as np
from scipy import special, optimize, spatial
import matplotlib.pyplot as plt

from sklearn import mixture
from sklearn import preprocessing

from sklearn.decomposition import PCA
from MulticoreTSNE import MulticoreTSNE as TSNE
from umap import UMAP

def stratefied_sampling(index, label, prob, size):
    sample = np.zeros(size).astype(index.dtype)
    label_count = (prob * size).astype(int)
    if label_count.sum() != size:
        label_count[-1] += (size-label_count.sum())
    progress_index = 0
    for label_i in range(prob.shape[0]):
        if label_count[label_i] > 0:
            index_i = index[label == label_i]
            index_i_sample = np.random.choice(index_i, size=label_count[label_i], replace=True)
            sample[progress_index:progress_index+label_count[label_i]] = index_i_sample
            progress_index += label_count[label_i]
    return sample

def generate_data(mean, cov, state, size, noise_level=0):
    n_components = state.shape[0]
    dim = mean.shape[1]
    state_logit = np.log(state+np.finfo(float).eps)
    state_logit_sample = state_logit + np.random.randn(*(state.shape)) * noise_level
    state_sample = special.softmax(state_logit_sample)
    mean_sample = mean + np.random.randn(*(mean.shape)) * noise_level
    cov_sample = cov + np.random.randn(*(cov.shape)) * noise_level
    count = (state_sample * size).astype(int)
    data = np.zeros((size, dim))
    front = 0
    for i in range(n_components):
        mean_i = mean_sample[i, ...]
        cov_i = cov_sample[i, ...]
        data[front:front+count[i], :] = np.random.multivariate_normal(
                mean=mean_i, cov=cov_i, size=count[i])
        front += count[i]
    return data

def treatment_ops(v0, treatment_operator):
    v0_logit = np.log(v0+np.finfo(float).eps)
    v1_logit = np.dot(v0_logit, treatment_operator)
    v1 = special.softmax(v1_logit)
    return v1

def loss_fn(v0, v1):
    dot_product = np.multiply(v0, v1).sum(axis=1)
    v0_length = np.sqrt(np.multiply(v0, v0).sum(axis=1))
    v1_length = np.sqrt(np.multiply(v1, v1).sum(axis=1))
    cos_distance = 1-np.divide(np.divide(dot_product, v0_length), v1_length)
    return cos_distance.mean()

def simulation_fn(
        gmm_mean, # mean of Gaussian mixture model, (n_components, dim)
        gmm_cov, # covariance of Gaussian mixture model, (n_components, dim, dim)
        state_0, # initial state in format of Categorical PMF, (n_components,)
        state_1, # final state in format of Categorical PMF, (n_components,)
        n_components=2,
        n_tiles=100,
        n_case=100,
        noise_level=1e-4):
    # data
    index_dict = {}
    dim = gmm_mean.shape[1]
    data = np.zeros((n_case * n_tiles * 2, dim)) # 2 states
    front = 0
    for i in range(n_case):
        data_0 = generate_data(mean=gmm_mean, cov=gmm_cov,
                state=state_0, size=n_tiles, noise_level=noise_level)
        data_1 = generate_data(mean=gmm_mean, cov=gmm_cov,
                state=state_1, size=n_tiles, noise_level=noise_level)
        index_0 = np.arange(front, front+n_tiles).astype(int)
        index_1 = np.arange(front+n_tiles, front+2*n_tiles).astype(int)
        index_dict[i] = {'state_0': index_0, 'state_1': index_1}
        data[index_0, :] = data_0
        data[index_1, :] = data_1
        front += 2 * n_tiles

    # model
    model = mixture.GaussianMixture(n_components=n_components)
    model.fit(data)
    sortkey = np.argsort(model.means_[:, 0])

    # estimate states
    v0 = np.zeros((n_case, n_components))
    for case_ID in index_dict:
        index = index_dict[case_ID]['state_0']
        y_pred = model.predict_proba(data[index, ...])
        y_pred = y_pred[:, sortkey]
        v0[case_ID, :] = y_pred.mean(axis=0)
    v1 = np.zeros((n_case, n_components))
    for case_ID in index_dict:
        index = index_dict[case_ID]['state_1']
        y_pred = model.predict_proba(data[index, ...])
        y_pred = y_pred[:, sortkey]
        v1[case_ID, :] = y_pred.mean(axis=0)

    # estimate linear transformation
    # v0: (n_case, n_components)
    # v1: (n_case, n_components)
    # solve A: (n_components, n_components)
    # v0 @ A = v1
    def target_fn(A_flat):
        A = A_flat.reshape((n_components, n_components))
        v1_pred = treatment_ops(v0, A)
        loss = loss_fn(v1, v1_pred)
        return loss
    minimization_result = optimize.minimize(fun=target_fn, x0=np.eye(n_components).flatten())

    # report
    result_dict = {
            'state_0_pred': v0,
            'state_1_pred': v1,
            'gmm': model,
            'sortkey': sortkey,
            'minimization_result': minimization_result,
            }
    metric_dict = {
            'state_0_cosine_distance':spatial.distance.cosine(state_0,
                result_dict['state_0_pred'].mean(axis=0)),
            'state_1_cosine_distance':spatial.distance.cosine(state_1,
                result_dict['state_1_pred'].mean(axis=0)),
            'minimization_loss': result_dict['minimization_result'].fun,
            'gmm_convergence': result_dict['gmm'].converged_,
            'gmm_lower_bound': result_dict['gmm'].lower_bound_,
            'gmm_mean_cosine_distance': spatial.distance.cosine(gmm_mean.flatten(),
                result_dict['gmm'].means_[result_dict['sortkey'], ...].flatten()),
            'gmm_cov_cosine_distance': spatial.distance.cosine(gmm_cov.flatten(),
                result_dict['gmm'].covariances_[result_dict['sortkey'], ...].flatten()),
            }

    return result_dict, metric_dict

if __name__ == '__main__':
    # load and preprocess MNIST embeddings from VAE
    z = np.load('./result/mnist_z.npy')
    y = np.load('./result/mnist_y.npy')
    z = preprocessing.scale(z)

    # ground truth: mean and covariance of the Gaussian mixture model
    true_mean = np.stack([
        z[y == 0, :].mean(axis=0),
        z[y == 1, :].mean(axis=0),
        ], axis=0) # (n_components, dim)
    true_cov = np.stack([
        np.cov(z[y == 0, :], rowvar=False),
        np.cov(z[y == 1, :], rowvar=False),
        ], axis=0) # (n_components, dim, dim)
    sortkey = np.argsort(true_mean[:, 0]) # for comparison
    true_mean = true_mean[sortkey, ...]
    true_cov = true_cov[sortkey, ...]

    # parameters to vary and monitor
    params = {
            'n_components': 2,
            'n_tiles': 100,
            'n_case': 10,
            'noise_level': 1e-4,
            }

    # run simulation
    for case in ['balanced', 'biased']:
        print('case {}'.format(case))
        for p in np.arange(0, 1.1, 0.1):
            state_0 = np.array([p, 1-p])

            if case == 'balanced':
                state_1 = 1-state_0
            elif case == 'biased':
                state_1 = state_0

            result_dict, metric_dict = simulation_fn(
                    gmm_mean=true_mean,
                    gmm_cov=true_cov,
                    state_0=state_0,
                    state_1=state_1,
                    **params,
                    )
            print('p: {:.3f}, state 0 loss: {:.3E}, state 1 loss: {:.3E}'.format(
                p, metric_dict['state_0_cosine_distance'], metric_dict['state_1_cosine_distance']))
            print(result_dict['gmm'].weights_[result_dict['sortkey']])

