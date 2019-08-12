import numpy as np
import typing

import tensorflow as tf
import tensorflow_probability as tfp

# convolutional variational autoencoder
class CVAE(tf.keras.Model):
    # initiation
    def __init__(self, 
            latent_dim: int,
            input_shape: typing.Tuple[int, int, int],
            optimizer: tf.optimizers.Optimizer,
            ) -> None:
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super().__init__()
        # set optimizer
        self.optimizer = optimizer
        # model architecture
        self.latent_dim = latent_dim
        self.data_shape = input_shape
        conv1_filter = 32
        conv1_kernel_size = 3
        conv1_strides = (1, 1)
        conv2_filter = 64
        conv2_kernel_size = 3
        conv2_strides = (1, 1)
        embedding_shape = (
            int(input_shape[0]/conv1_strides[0]/conv2_strides[0]),
            int(input_shape[1]/conv1_strides[1]/conv2_strides[1]),
            int(conv2_filter/2), # mean and variance
            )
        reconstruct_filter = input_shape[2]
        reconstruct_kernel_size = 3
        reconstruct_strides = (1, 1)
        # define encoder architecture
        self.encoder = tf.keras.Sequential([
            # input layer
            tf.keras.layers.InputLayer(input_shape=input_shape),
            # convolution layer(s)
            tf.keras.layers.Conv2D(
                filters=conv1_filter,
                kernel_size=conv1_kernel_size,
                strides=conv1_strides,
                activation='relu',
                ),
            tf.keras.layers.Conv2D(
                filters=conv2_filter,
                kernel_size=conv2_kernel_size,
                strides=conv2_strides,
                activation='relu',
                ),
            # flatten for fully-connected layer(s)
            tf.keras.layers.Flatten(),
            # No activation
            # output dimension is laten_dim (mean) + laten_dim (logvar)
            tf.keras.layers.Dense(units=latent_dim + latent_dim),
            ])
        # define decoder architecture
        self.decoder = tf.keras.Sequential([
            # input layer
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            # WARNING: where is the variational part?
            tf.keras.layers.Dense(
                units=embedding_shape[0]*embedding_shape[1]*embedding_shape[2],
                activation=tf.nn.relu),
            # reshape for convolution
            tf.keras.layers.Reshape(target_shape=embedding_shape),
            # convolution layer(s)
            tf.keras.layers.Conv2DTranspose(
                filters=conv2_filter,
                kernel_size=conv2_kernel_size,
                strides=conv2_strides,
                padding='same',
                activation='relu',
                ),
            tf.keras.layers.Conv2DTranspose(
                filters=conv1_filter,
                kernel_size=conv1_kernel_size,
                strides=conv1_strides,
                padding='same',
                activation='relu',
                ),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=reconstruct_filter,
                kernel_size=reconstruct_kernel_size,
                strides=reconstruct_strides,
                padding='same',
                ),
            ])

    def sample(self, n: int=100):
        eps = tf.random.normal(shape=(n, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x: np.ndarray):
        if x.shape[1::] != self.data_shape:
            raise ValueError(
                    'Model was trained on shape {} but input has shape {}'.format(
                        self.data_shape, x.shape))
        mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=tf.shape(mean))
        return eps * tf.exp(logvar * .5) + mean

    def decode(self, z):
        x_pred = self.decoder(z)
        return x_pred

    def compute_loss(self, x):
        # reconstruction in the data space
        # MSE assumes Gaussian noise
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_pred = self.decode(z)
        reconstruction_loss = tf.math.reduce_mean(tf.keras.losses.MSE(x, x_pred))
        # K-L Divergence in the latent space
        def KLD(y_true, y_pred):
            return y_true*(tf.math.log(y_true)-tf.math.log(y_pred))
        def pdf(sample, mu, sigma):
            prob = tfp.distributions.MultivariateNormalDiag(
                    loc=mu, scale_diag=sigma).prob(sample)
            # for numerical stability
            prob = prob.numpy()
            prob[prob < np.finfo(np.float32).eps] = np.finfo(np.float32).eps
            return prob
        p_data = pdf(z, mean, tf.math.exp(logvar * .5))
        p_target = pdf(z, tf.zeros(shape=mean.shape), tf.ones(logvar.shape))
        latent_loss = tf.math.reduce_mean(KLD(p_target, p_data))
        return reconstruction_loss+latent_loss

    def compute_apply_gradients(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

