import numpy as np
import pickle
import argparse
import os

import CAEP
import tensorflow as tf

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype(np.float32)/255
    x_test = x_test.astype(np.float32)/255
    # parse arguments
    parser = argparse.ArgumentParser(description='get job id')
    parser.add_argument('--jobid', help='Slurm job id.', default='default')
    args = parser.parse_args()
    folderpath = 'joboutput_{}'.format(args.jobid)
    with open(os.path.join(folderpath, 'model.pkl'), 'rb') as infile:
        model_dict = pickle.load(infile)
    model = CAEP.CAEP(
        latent_dim=model_dict['LATENT_DIM'],
        feature_shape=model_dict['feature_shape'],
        )
    model.inference_net.set_weights(model_dict['inference_net_weights'])
    model.generation_net.set_weights(model_dict['generation_net_weights'])
    for pre_digit in model_dict['PRE_DIGIT']:
        for post_digit in model_dict['POST_DIGIT']:
            all_index = np.arange(y_test.shape[0]).astype(int)[y_test == pre_digit]
            index = np.random.choice(all_index, replace=False, size=10)
            x_pre = x_test[index, ...]
            x_pre = np.stack([x_pre]*3, axis=-1)
            z_pre = model.inference_net(x_pre)
            delta_z = np.ones(z_pre.shape) * model_dict['prior_mean']
            z_post = z_pre + delta_z
            x_post = model.generation_net(z_post)
            pre_name = 'prediction_{}.npy'.format(pre_digit)
            post_name = 'prediction_{}_{}.npy'.format(pre_digit, post_digit)
            np.save(os.path.join(folderpath, pre_name), x_pre)
            np.save(os.path.join(folderpath, post_name), x_post)
