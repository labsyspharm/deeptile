import numpy as np
import typing

import tensorflow as tf
import tensorflow_probability as tfp

# adversarial convolutional autoencoder
class ACAE(tf.keras.Model):
    '''
    Reference: https://arxiv.org/abs/1511.05644
    Use architecture in Fig. 3 as starting point
    Use pre-trained convolution layers (ex. VGG)
        Check here: https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/keras/applications
    We only have one label per WSI, not per tile,
        so will need some modification on the discriminant part.
    One (out of tons) of AAE implemented in Tensorflow on Github: 
        https://github.com/Naresh1318/Adversarial_Autoencoder
    '''
    def __init__(self, 
            latent_dim: int, # latent dimension of style vector
            label_num: int, # number of tile categories
            feature_shape: typing.Tuple[int, int, int], # x-y-channel
            prior_mean: np.float32=None, # MVN prior mean
            prior_std: np.float32=None, # MVN prior std
            ):
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super().__init__()
        # set attributes
        self.latent_dim = latent_dim
        self.label_num = label_num
        self.feature_shape = feature_shape
        if prior_std is None:
            self.prior_std = 1
        if prior_mean is None:
            self.prior_mean = 6 * self.prior_std / np.sqrt(2)
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
                        tf.keras.layers.Dense(
                            units=latent_dim+label_num,
                            )],
                    )
        self.generation_net = tf.keras.Sequential(
                [tf.keras.layers.InputLayer(input_shape=(latent_dim+label_num,)),
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
        self.discriminant_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(
                units=2, # binary
                activation=tf.nn.softmax),
            ])
        # set weight and none-trainable
        # note: new model layers count skips InputLayer
        for index in range(1, 1+len(forward_layers)):
            if isinstance(vgg_model.layers[index], tf.keras.layers.Conv2D):
                self.inference_net.layers[index-1].set_weights(vgg_model.layers[index].get_weights())
                self.inference_net.layers[index-1].trainable = False

    def infer(self, x):
        vector_pred_raw = self.inference_net(x)
        z_pred, y_pred_logit = tf.split(vector_pred_raw, 
                [self.latent_dim, self.label_num], axis=1)
        return z_pred, y_pred_logit

    def generate(self, z_pred, y_pred_logit):
        vector_pred = tf.concat([z_pred, y_pred_logit], axis=1)
        return self.generation_net(vector_pred)

    def compute_prior(self, y_pred_logit):
        y_pred = tf.one_hot(tf.math.argmax(y_pred_logit, axis=1),
                depth=y_pred_logit.shape[1])
        # need to handle different dimension Y category ==> D dimension
        dist_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.math.multiply(y_pred, self.prior_mean),
                scale_diag=tf.math.multiply(tf.ones(y_pred.shape), self.prior_std),
                )
        z_prior = dist_prior.sample()
        return z_prior

    def compute_loss(self, x):
        z_pred, y_pred_logit = self.infer(x)
        x_pred = self.generate(z_pred, y_pred_logit)
        z_prior = self.compute_prior(y_pred_logit)
        print('z_pred shape', z_pred.shape)
        print('z_prior shape', z_prior.shape)
        label_pred = self.discriminant_net(z_pred)
        label_prior = self.discriminant_net(z_prior)
        # reconstruction
        reconstruction_loss = tf.reduce_sum(
                tf.reduce_sum(
                    tf.reduce_sum(
                        tf.square(x - x_pred),
                        axis=3),
                    axis=2),
                axis=1)
        # discrimination
        label_prior_true = tf.concat(
                [tf.ones(label_prior.shape[0], 1), 
                    tf.zeros(label_prior.shape[0], 1)], axis=1)
        label_pred_true = tf.concat(
                [tf.zeros(label_prior.shape[0], 1), 
                    tf.ones(label_prior.shape[0], 1)], axis=1)
        discrimination_loss_prior = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prior_true, logits=label_prior))
        discrimination_loss_pred = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_pred_true, logits=label_pred))
        discrimination_loss = discrimination_loss_prior + discrimination_loss_pred
        # confusion
        confusion_loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=label_prior_true, logits=label_pred))
        return {
                'reconstruction':reconstruction_loss,
                'discrimination':discrimination_loss,
                'confusion':confusion_loss,
                }

