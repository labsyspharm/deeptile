import numpy as np
import typing
import time

import tensorflow as tf
import tensorflow_probability as tfp

# convenient abbreviations
tfk = tf.keras
tfkl = tf.keras.layers
tfd = tfp.distributions
tfpl = tfp.layers

# adversarial convolutional autoencoder
class VCAE(object):
    def __init__(self, 
            latent_dim: int, # latent dimension of style vector
            feature_shape: typing.Tuple[int, int, int], # x-y-channel
            MNIST: bool=False, # test flag
            ):
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super().__init__()
        # set attributes
        self.latent_dim = latent_dim
        self.feature_shape = feature_shape
        # download VGG19 first two convolution layers
        vgg_model = tfk.applications.vgg19.VGG19(include_top=False, weights='imagenet')
        # TODO: pseudo-coloring: 1x1 conv layers, trainable, 30->10->3
        # define convolutional layers
        forward_layers = []
        backward_layers = []
        x_shrink_factor = 1
        y_shrink_factor = 1
        if MNIST:
            layers = vgg_model.layers[1:7]
        else:
            layers = vgg_model.layers[1:]
        for layer in layers: # skip original input layer
            if isinstance(layer, tfkl.Conv2D):
                new_forward_layer = tfkl.Conv2D(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation=layer.activation,
                        )
                new_backward_layer = tfkl.Conv2DTranspose(
                        filters=layer.filters,
                        kernel_size=layer.kernel_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        activation=layer.activation,
                        )
            elif isinstance(layer, tfkl.MaxPool2D):
                new_forward_layer = tfkl.MaxPool2D(
                        pool_size=layer.pool_size,
                        strides=layer.strides,
                        padding=layer.padding,
                        )
                new_backward_layer = tfkl.UpSampling2D(
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
        self.prior = tfd.Independent(tfd.Normal(loc=tf.zeros(latent_dim), scale=1), reinterpreted_batch_ndims=1)
        self.inference_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=feature_shape)]\
                    + forward_layers\
                    + [tfkl.Flatten()]\
                    + [tfkl.Dense(
                        units=tfpl.MultivariateNormalTriL.params_size(latent_dim), activation=None)]\
                    + [tfpl.MultivariateNormalTriL(latent_dim,
                        activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior, weight=1.0))]\
                                )
        self.generation_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=(latent_dim,))]\
                    + [tfkl.Dense(
                        units=deconv_shape[0]*deconv_shape[1]*deconv_shape[2],
                        activation=tfk.activations.relu)]\
                    + [tfkl.Reshape(target_shape=deconv_shape)]\
                    + backward_layers\
                    + [tfkl.Conv2D( # final layer
                        filters=feature_shape[2],
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='same',
                        activation=tfk.activations.sigmoid)]\
                                )
        # set weight and none-trainable
        # note: new model layers count skips InputLayer
        for index in range(1, 1+len(forward_layers)):
            if isinstance(vgg_model.layers[index], tfkl.Conv2D):
                self.inference_net.layers[index-1].set_weights(
                        vgg_model.layers[index].get_weights())
                self.inference_net.layers[index-1].trainable = False
        # compile model
        self.vae = tfk.Model(
                inputs=self.inference_net.inputs,
                outputs=self.generation_net(self.inference_net.outputs),
                )

