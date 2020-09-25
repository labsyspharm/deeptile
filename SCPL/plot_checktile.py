import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tqdm

from scipy import stats

if __name__ == '__main__':
    # path
    data_folderpath = '/n/scratch2/hungyiwu/project_deeptile/data/'
    width_list = list(range(5, 32, 2))
    fs = 18

    fig, axes = plt.subplots(ncols=2, nrows=7, sharex=True, sharey=True,
            figsize=(8,15))
    ax_list = axes.T.flatten()
    for i in tqdm.tqdm(range(len(width_list))):
        width = width_list[i]
        filepath = os.path.join(data_folderpath, 'checktile_{}x{}.csv'.format(width, width))
        df = pd.read_csv(filepath)
        subsample = np.random.choice(range(df.shape[0]), size=int(1e4), replace=True)
        x = df['frac_cell_in_tile'].values[subsample]
        y = df['frac_neighbor_in_tile'].values[subsample]
        xy = np.stack([x,y], axis=-1).T
        d = stats.gaussian_kde(xy)(xy)
        sortkey = np.argsort(d)
        x, y, d = x[sortkey], y[sortkey], d[sortkey]
        ax_list[i].scatter(x, y, c=d, s=1, cmap='coolwarm')
        ax_list[i].set_title('{}x{}'.format(width, width), fontsize=fs)
    # arange figure
    axes[0, 0].set_xlim([-0.1, 1.1])
    axes[0, 0].set_ylim([-0.1, 1.1])
    for ax in axes[6, :]:
        ax.set_xlabel('cell area included', fontsize=fs)
    axes[3, 0].set_ylabel('neighbors in tile', fontsize=fs)
    ticks = [0, 0.5, 1]
    ticklabels = [format(i, '.1f') for i in ticks]
    for ax in axes.flatten():
        ax.set_xticks(ticks)
        ax.set_xticklabels(ticklabels, fontsize=fs)
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels, fontsize=fs)

    fig.tight_layout()
    output_filepath = os.path.join(data_folderpath, 'check_tile.png')
    plt.savefig(output_filepath)
    plt.close()
