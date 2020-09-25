import numpy as np
import typing

import tensorflow as tf
import tensorflow_probability as tfp

# convolutional autoencoder with variational constraint at
# "delta latent vector" (multivariate normal)
# temporary code name: Convolutional AutoEncoder for Perturbation
# blueprint: https://tinyurl.com/yyk5ln3s
class CAEP(tf.keras.Model):
    def __init__(self,
            latent_dim: int, # latent dimension
            feature_shape: typing.Tuple[int, int, int],
            ):
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super().__init__()
        # set attributes
        self.latent_dim = latent_dim
        self.feature_shape = feature_shape
        self.prior_mean = 6./np.sqrt(self.latent_dim)
        self.prior_std = 1
        reconstruct_filter = feature_shape[2]
        reconstruct_kernel_size = (3, 3)
        reconstruct_strides = (1, 1)
        # download VGG19 first two convolution layers
        vgg_model = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        # TODO: pseudo-coloring: 1x1 conv layers, trainable, 30->10->3
        # define convolutional layers
        forward_layers = []
        backward_layers = []
        x_shrink_factor = 1
        y_shrink_factor = 1
        for layer in vgg_model.layers[1:7]: # skip original input layer
            if isinstance(layer, tf.keras.layers.Conv2D):
                new_forward_layer = tf.keras.layers.Conv2D(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation=layer.activation,
                        )
                new_backward_layer = tf.keras.layers.Conv2DTranspose(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation=layer.activation,
                        )
            elif isinstance(layer, tf.keras.layers.MaxPool2D):
                new_forward_layer = tf.keras.layers.MaxPool2D(
                        pool_size=layer.pool_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        )
                new_backward_layer = tf.keras.layers.UpSampling2D(
                        size=layer.pool_size,
                        )
            else:
                raise ValueError('unrecognized layer in VGG19 {}'.format(type(layer)))
            forward_layers.append(new_forward_layer)
            backward_layers.insert(0, new_backward_layer)
            x_shrink_factor *= layer.strides[0]
            y_shrink_factor *= layer.strides[1]
        # define inference, generation, discriminant networks
        deconv_shape = (
            int(feature_shape[0]/x_shrink_factor),
            int(feature_shape[1]/y_shrink_factor),
            int(forward_layers[-2].filters),
            )
        # define inference, generation networks
        self.inference_net = tf.keras.Sequential(
                [tf.keras.layers.InputLayer(input_shape=feature_shape)]\
                    + forward_layers\
                    + [tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(units=latent_dim)]) # no activation
        self.generation_net = tf.keras.Sequential(
                [tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
                    tf.keras.layers.Dense(
                        units=deconv_shape[0]*deconv_shape[1]*deconv_shape[2],
                        activation=tf.keras.activations.relu),
                    tf.keras.layers.Reshape(target_shape=deconv_shape)]\
                            + backward_layers\
            + [tf.keras.layers.Conv2DTranspose( # final deconvolution layer
                filters=reconstruct_filter,
                kernel_size=reconstruct_kernel_size,
                strides=reconstruct_strides,
                padding='same',
                activation=tf.keras.activations.sigmoid,
                )])
        # set weight and none-trainable
        # note: new model layers count skips InputLayer
        for index in range(1, 1+len(forward_layers)):
            if isinstance(vgg_model.layers[index], tf.keras.layers.Conv2D):
                self.inference_net.layers[index-1].set_weights(vgg_model.layers[index].get_weights())
                self.inference_net.layers[index-1].trainable = False

    def compute_loss(
            self, 
            x_0, # tile before perturbation, data space, shape=(N, X, Y, Channel)
            x_1, # tile after perturbation, data space, shape=(M, X, Y, Channel)
            ):
        z_0_pred = self.inference_net(x_0)
        z_1_pred = self.inference_net(x_1)
        x_0_pred = self.generation_net(z_0_pred)
        x_1_pred = self.generation_net(z_1_pred)
        # reconstruction loss, N+M tiles
        x_0_loss = tf.keras.losses.MSE(x_0, x_0_pred)
        x_1_loss = tf.keras.losses.MSE(x_1, x_1_pred)
        reconstruction_loss = tf.math.reduce_sum(x_0_loss + x_1_loss)
        # prior belief:
        # (z_1_pred - z_0_pred) ~ MVN(loc=6/np.sqrt(latent_dim), scale=?)
        # calculate K-L divergence and then back-propagation
        # N*M tile-pairs
        # use broadcasting for efficient pair-wise subtraction
        # idea from https://tinyurl.com/y22rnav6
        z_0_pred = tf.expand_dims(input=z_0_pred, axis=0)
        z_1_pred = tf.expand_dims(input=z_1_pred, axis=1)
        delta_z_pred = tf.reshape(tensor=z_1_pred-z_0_pred, shape=(-1, self.latent_dim))
        dist_pred = tfp.distributions.MultivariateNormalDiag(
                loc=tf.math.reduce_mean(input_tensor=delta_z_pred, axis=0), 
                scale_diag=tf.math.reduce_std(input_tensor=delta_z_pred, axis=0),
                )
        dist_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.ones(shape=(self.latent_dim,)) * self.prior_mean,
                scale_diag=tf.ones(shape=(self.latent_dim,)) * self.prior_std,
                )
        latent_loss = tfp.distributions.kl_divergence(dist_pred, dist_prior)
        return reconstruction_loss+latent_loss

    def compute_apply_gradients(
            self, 
            x_0, # tile before perturbation, data space, shape=(N, X, Y, Channel)
            x_1, # tile after perturbation, data space, shape=(M, X, Y, Channel)
            optimizer, # tf.keras.optimizers
            ):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x_0, x_1)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

