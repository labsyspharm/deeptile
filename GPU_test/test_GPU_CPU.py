import tensorflow as tf
import time

def run(device_ID: str='default', dim: int=100, n: int=100) -> None:
    if device_ID == 'default':
        ts_start = time.time()
        a = tf.ones(shape=(dim, dim))
        b = tf.ones(shape=(dim, dim))
        time_load = time.time()-ts_start
        ts_start = time.time()
        for _ in range(n):
            c = tf.matmul(a, b)
        time_run = (time.time()-ts_start)/n
    else:
        with tf.device(device_ID):
            ts_start = time.time()
            a = tf.ones(shape=(dim, dim))
            b = tf.ones(shape=(dim, dim))
            time_load = time.time()-ts_start
            ts_start = time.time()
            for _ in range(n):
                c = tf.matmul(a, b)
            time_run = (time.time()-ts_start)/n
    print('device ID: {}, dimension: {}, n: {}'.format(device_ID, dim, n),
            'load time: {:.1f} msec, run time: {:.1f} usec'.format(time_load*1e3, time_run*1e6))
    return

if __name__ == '__main__':
    print('GPU available?:', tf.test.is_gpu_available())
    run(dim=500, n=int(1e4))
    print('Done.')
