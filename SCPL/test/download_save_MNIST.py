import numpy as np
import os
import shutil
from tensorflow.keras.datasets import mnist

if __name__ == '__main__':
    (train_X, train_Y), (test_X, test_Y) = mnist.load_data()
    folderpath = './MNIST'
    if os.path.isdir(folderpath):
        shutil.rmtree(folderpath)
    os.mkdir(folderpath)
    np.save(os.path.join(folderpath, 'train_X.npy'), train_X)
    np.save(os.path.join(folderpath, 'train_Y.npy'), train_Y)
    np.save(os.path.join(folderpath, 'test_X.npy'), test_X)
    np.save(os.path.join(folderpath, 'test_Y.npy'), test_Y)
    print('Done.')
