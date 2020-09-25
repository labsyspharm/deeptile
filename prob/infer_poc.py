import numpy as np
from scipy import special, optimize
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

def generate_data(mean, cov, state, size, noise_level=1e-2):
    n_components = state.shape[0]
    dim = mean.shape[1]
    state_logit = np.log(state)
    state_logit_sample = np.random.randn(*(state.shape)) * noise_level
    state_sample = special.softmax(state_logit_sample)
    mean_sample = mean + np.random.randn(*(mean.shape)) * noise_level
    cov_sample = cov + np.random.randn(*(cov.shape)) * noise_level
    count = (state * size).astype(int)
    data = np.zeros((size, dim))
    front = 0
    for i in range(n_components):
        mean_i = mean_sample[i, ...]
        cov_i = cov_sample[i, ...]
        data[front:front+count[i], :] = np.random.multivariate_normal(
                mean=mean_i, cov=cov_i, size=count[i])
        front += count[i]
    np.random.shuffle(data)
    return data

def treatment_ops(v0, treatment_operator):
    v0_logit = np.log(v0)
    v1_logit = np.dot(v0_logit, treatment_operator)
    v1 = special.softmax(v1_logit)
    return v1

def loss_fn(v0, v1):
    dot_product = np.multiply(v0, v1).sum(axis=1)
    v0_length = np.sqrt(np.multiply(v0, v0).sum(axis=1))
    v1_length = np.sqrt(np.multiply(v1, v1).sum(axis=1))
    cos_distance = 1-np.divide(np.divide(dot_product, v0_length), v1_length)
    return cos_distance.mean()

if __name__ == '__main__':
    # setup
    n_components = 2 # heuristic number
    n_tiles = int(1e2)
    n_case = 100
    noise_level = 1e-4
    z_obs = np.load('./result/mnist_z.npy')
    y_obs = np.load('./result/mnist_y.npy')

    # preprocessing
    z_obs = preprocessing.scale(z_obs)
    dim = z_obs.shape[1]

    # ground truth
    treatment_operator = np.array([[1., 0.], [0., 2.]])
    state_0 = np.ones(2) * 0.5
    state_1 = treatment_ops(state_0, treatment_operator)
    anchor_mean = np.stack([
        z_obs[y_obs == 0, :].mean(axis=0),
        z_obs[y_obs == 1, :].mean(axis=0),
        ], axis=0) # (n_components, dim)
    anchor_cov = np.stack([
        np.cov(z_obs[y_obs == 0, :], rowvar=False),
        np.cov(z_obs[y_obs == 1, :], rowvar=False),
        ], axis=0) # (n_components, dim, dim)
    anchor_mean.sort(axis=0) # for comparison
    anchor_cov.sort(axis=0)
    params_true = {
            'treatment_operator': treatment_operator,
            'state_0': state_0,
            'state_1': state_1,
            'anchor_mean': anchor_mean,
            'anchor_cov': anchor_cov,
            }

    # data
    data_dict = {}
    pooled_data_array = np.zeros((n_case * n_tiles * 2, dim)) # 2 states
    front = 0
    for i in range(n_case):
        data_state_0 = generate_data(mean=anchor_mean, cov=anchor_cov,
                state=state_0, size=n_tiles, noise_level=noise_level)
        data_state_1 = generate_data(mean=anchor_mean, cov=anchor_cov,
                state=state_1, size=n_tiles, noise_level=noise_level)
        data_dict[i] = [data_state_0, data_state_1]
        pooled_data_array[front:front+n_tiles, :] = data_state_0
        front += n_tiles
        pooled_data_array[front:front+n_tiles, :] = data_state_1
        front += n_tiles

    # model
    model = mixture.GaussianMixture(n_components=n_components)
    model.fit(pooled_data_array)
    model.means_.sort(axis=0)
    model.covariances_.sort(axis=0)
    params_estimate = {
            'anchor_mean': model.means_,
            'anchor_cov': model.covariances_,
            }

    # estimate states
    v0 = np.zeros((n_case, n_components))
    for case_ID in data_dict:
        y_pred = model.predict(data_dict[case_ID][0])
        y_onehot = np.zeros((y_pred.shape[0], n_components))
        y_onehot[np.arange(y_onehot.shape[0]).astype(int), y_pred] = 1
        v0[case_ID, :] = y_onehot.mean(axis=0)
    v1 = np.zeros((n_case, n_components))
    for case_ID in data_dict:
        y_pred = model.predict(data_dict[case_ID][1])
        y_onehot = np.zeros((y_pred.shape[0], n_components))
        y_onehot[np.arange(y_onehot.shape[0]).astype(int), y_pred] = 1
        v1[case_ID, :] = y_onehot.mean(axis=0)

    # estimate linear transformation
    # v0: (n_case, n_components)
    # v1: (n_case, n_components)
    # solve A: (n_components, n_components)
    # A @ v0 = v1
    def target_fn(A_flat):
        A = A_flat.reshape((n_components, n_components))
        v1_pred = treatment_ops(v0, A)
        loss = loss_fn(v1, v1_pred)
        return loss
    result = optimize.minimize(fun=target_fn, x0=np.eye(n_components).flatten())
    params_estimate['treatment_operator'] = result.x.reshape((n_components,n_components))

    # check
    print('true treatment operator')
    print(params_true['treatment_operator'])
    print('estimated treatment operator')
    print(params_estimate['treatment_operator'])
    print('true state 0:', params_true['state_0'])
    state_1_estimate = treatment_ops(params_true['state_0'], params_estimate['treatment_operator'])
    print('estimated state 1:', state_1_estimate)
    print('true state 1:', params_true['state_1'])
    print('loss', result.fun)
    center = np.arange(dim)
    width = 0.3
    fig, ax = plt.subplots(ncols=n_components, nrows=1, figsize=(10,5))
    for i in range(n_components):
        ax[i].bar(center-width/2, params_true['anchor_mean'][i, :], width=width)
        ax[i].bar(center+width/2, params_estimate['anchor_mean'][i, :], width=width)
        ax[i].legend(['true', 'estimate'])
        ax[i].set_xlabel('dimension')
        ax[i].set_ylabel('weight')
        ax[i].set_title('component {}'.format(i+1))

    fig.tight_layout()
    plt.show()
