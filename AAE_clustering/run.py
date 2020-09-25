import numpy as np
import pandas as pd
import time
import itertools

import tensorflow as tf
import tqdm

import AAE

# convenient abbreviation
tfk = tf.keras

def preprocess_data(x):
    return np.stack([x.astype(np.float32)/255]*3, axis=-1)

if __name__ == '__main__':
    # parameters
    BATCH_SIZE = 32
    LATENT_DIM = 20
    NUM_LABEL = 10
    TOTAL_EPOCH = 20
    LEARNING_RATE = 1e-5
    # data
    print('preparing data...')
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = preprocess_data(x_train)
    x_test = preprocess_data(x_test)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train,))\
            .batch(BATCH_SIZE)\
            .shuffle(x_train.shape[0])
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test,))\
            .batch(BATCH_SIZE)\
            .shuffle(x_test.shape[0])
    train_batch_count = np.ceil(x_train.shape[0] / BATCH_SIZE).astype(int)
    test_batch_count = np.ceil(x_test.shape[0] / BATCH_SIZE).astype(int)
    feature_shape = x_train[0, ...].shape
    # model and optimizer
    print('preparing model...')
    context_list = ['reconstruction', 'discrimination', 'confusion']
    opt_dict = {context: tfk.optimizers.SGD(learning_rate=LEARNING_RATE)\
            for context in context_list}
    model = AAE.AAE(
            latent_dim=LATENT_DIM,
            num_label=NUM_LABEL,
            feature_shape=feature_shape,
            MNIST=True,
            )
    # evaluation
    def evaluate():
        train_loss = np.zeros(train_batch_count)
        test_loss = np.zeros(test_batch_count)
        for index, (batch_x,) in enumerate(train_dataset):
                r_loss, d_loss, c_loss = model.compute_loss(batch_x, context='evaluation')
                train_loss[index] = r_loss.numpy().mean()\
                        + d_loss.numpy().mean()\
                        + c_loss.numpy().mean()
        for index, (batch_x,) in enumerate(test_dataset):
                r_loss, d_loss, c_loss = model.compute_loss(batch_x, context='evaluation')
                test_loss[index] = r_loss.numpy().mean()\
                        + d_loss.numpy().mean()\
                        + c_loss.numpy().mean()
        return train_loss.mean(), test_loss.mean()
    record = []
    # preview
    print('running preview...')
    ts_start = time.time()
    train_loss, test_loss = evaluate()
    ts_end = time.time()
    result = [0, ts_end-ts_start, train_loss, test_loss]
    record.append(result)
    print('epoch {}, runtime {:.3f} sec, train loss {:.3E}, test loss {:.3E}'.format(*result), flush=True)
    # train
    print('started training')
    for epoch in range(1, TOTAL_EPOCH+1):
        ts_start = time.time()
        for (batch_x,) in train_dataset:
            for context in context_list:
                model.compute_apply_gradients(batch_x, context=context, optimizer_dict=opt_dict)
        train_loss, test_loss = evaluate()
        ts_end = time.time()
        result = [epoch, ts_end-ts_start, train_loss, test_loss]
        record.append(result)
        print('epoch {}, runtime {:.3f} sec, train loss {:.3E}, test loss {:.3E}'.format(*result), flush=True)
    # save
    print('saving result...')
    df = pd.DataFrame.from_records(record, columns=['epoch', 'runtime(sec)', 'train_loss', 'test_loss'])
    df.to_csv('train_history.csv', index=False)

