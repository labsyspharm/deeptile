import numpy as np
import typing
import tensorflow as tf

# convolutional variational autoencoder
class CVAE(tf.keras.Model):
    # initiation
    def __init__(self, 
            latent_dim: int,
            input_shape: typing.Tuple[int, int, int],
            ) -> None:
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super(CVAE, self).__init__()
        # populate attributes
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

    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

# utility functions below

def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

def compute_loss(model, x):
    mean, logvar = model.encode(x)
    z = model.reparameterize(mean, logvar)
    x_logit = model.decode(z)
    # I need to understand the cross entropy term below someday
    # also this line took too much memory so 20G is not enough for one batch
    cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
    logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
    logpz = log_normal_pdf(z, 0., 0.)
    logqz_x = log_normal_pdf(z, mean, logvar)
    loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
    return loss

def compute_apply_gradients(model, x, optimizer):
    with tf.GradientTape() as tape:
        loss = compute_loss(model, x)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss.numpy()

