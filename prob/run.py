import numpy as np
import pandas as pd
import time
import pickle
import itertools

import tensorflow as tf

import VCAE

# convenient abbreviation
tfk = tf.keras

if __name__ == '__main__':
    # parameters
    BATCH_SIZE = 64
    LATENT_DIM = 20
    TOTAL_EPOCH = 20
    LEARNING_RATE = 1e-4
    # data
    def prep(x):
        return np.stack([x.astype(np.float32)/255]*3, axis=-1)
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = prep(x_train)
    x_test = prep(x_test)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,))\
            .batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,))\
            .batch(BATCH_SIZE)
    train_batch_count = np.ceil(x_train.shape[0] / BATCH_SIZE).astype(int)
    test_batch_count = np.ceil(x_test.shape[0] / BATCH_SIZE).astype(int)
    # model and optimizer
    opt = tfk.optimizers.Adam(learning_rate=LEARNING_RATE)
    model = VCAE.VCAE(
            latent_dim=LATENT_DIM,
            feature_shape=x_train[0, ...].shape,
            MNIST=True,
            )
    # evaluation
    def evaluate():
        train_loss = np.zeros(train_batch_count)
        test_loss = np.zeros(test_batch_count)
        for index, (batch_x,) in enumerate(train_dataset):
            loss = tf.math.reduce_mean(tfk.losses.MSE(batch_x, model.vae(batch_x)))
            train_loss[index] = loss.numpy()
        for index, (batch_x,) in enumerate(test_dataset):
            loss = tf.math.reduce_mean(tfk.losses.MSE(batch_x, model.vae(batch_x)))
            test_loss[index] = loss.numpy()
        return train_loss.mean(), test_loss.mean()
    record = []
    # preview
    ts_start = time.time()
    train_loss, test_loss = evaluate()
    ts_end = time.time()
    result = [0, ts_end-ts_start, train_loss, test_loss]
    record.append(result)
    print('epoch {}, runtime {:.3f} sec, train loss {:.3E}, test loss {:.3E}'.format(*result), flush=True)
    # train
    for epoch in range(1, TOTAL_EPOCH+1):
        ts_start = time.time()
        for (batch_x,) in train_dataset:
            loss = lambda: tf.math.reduce_mean(tfk.losses.MSE(batch_x, model.vae(batch_x)))
            opt.minimize(loss=loss, var_list=model.vae.trainable_variables)
        train_loss, test_loss = evaluate()
        ts_end = time.time()
        result = [epoch, ts_end-ts_start, train_loss, test_loss]
        record.append(result)
        print('epoch {}, runtime {:.3f} sec, train loss {:.3E}, test loss {:.3E}'.format(*result), flush=True)
    # save
    df = pd.DataFrame.from_records(record, columns=['epoch', 'runtime(sec)', 'train_loss', 'test_loss'])
    df.to_csv('train_history.csv', index=False)
    z_pred = np.zeros((x_train.shape[0]+x_test.shape[0], LATENT_DIM))
    progress_index = 0
    for index, (batch_x,) in enumerate(itertools.chain(train_dataset, test_dataset)):
        batch_z = model.inference_net(batch_x).sample().numpy()
        z_pred[progress_index:progress_index+batch_z.shape[0], :] = batch_z
        progress_index += batch_z.shape[0]
    np.save('mnist_z.npy', z_pred)
    np.save('mnist_y.npy', np.concatenate([y_train, y_test], axis=0))
    weight_dict = {
            'inference_net': model.inference_net.get_weights(),
            'generation_net': model.generation_net.get_weights(),
            'vae': model.vae.get_weights(),
            }
    with open('vae_weight.pkl', 'wb') as outfile:
        pickle.dump(weight_dict, outfile)
