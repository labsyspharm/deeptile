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
            latent_dim: int, # latent dimension
            feature_shape: typing.Tuple[int, int, int], # x-y-channel
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
        # Bayesian prior for the state vectors
        self.state_vector_0 = np.ones(latent_dim).astype(np.float32) / latent_dim
        self.state_vector_1 = np.ones(latent_dim).astype(np.float32) / latent_dim
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
                            units=latent_dim,
                            activation=tf.keras.activations.softmax,
                            )],
                    )
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
        self.perturbation_net = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(
                units=latent_dim,
                activation=tf.keras.activations.relu,
                ),
            tf.keras.layers.Dense(
                units=latent_dim,
                activation=tf.keras.activations.softmax,
                ),
            ])
        # set weight and none-trainable
        # note: new model layers count skips InputLayer
        for index in range(1, 1+len(forward_layers)):
            if isinstance(vgg_model.layers[index], tf.keras.layers.Conv2D):
                self.inference_net.layers[index-1].set_weights(vgg_model.layers[index].get_weights())
                self.inference_net.layers[index-1].trainable = False

    def compute_state(self, dataset):
        vi_prior_mean = 0
        vi_prior_std = 1
        reconstruction_loss = 0
        z_list = []
        for (x,) in dataset:
            # reconstruction
            z_pred = self.inference_net(x)
            x_pred = self.generation_net(z_pred)
            reconstruction_loss += tf.math.reduce_sum(tf.keras.losses.MSE(x, x_pred))
            z_list.append(z_pred)
        z_pred = tf.concat(z_list, axis=0)
        z_pred_mean = tf.math.reduce_mean(z_pred, axis=0)
        z_pred_std = tf.math.reduce_std(z_pred, axis=0)
        count_tile = z_pred.shape[0]
        reconstruction_loss = tf.math.divide(reconstruction_loss, count_tile)
        state = tf.math.reduce_sum(z_pred, axis=0)
        state = tf.math.divide(state, tf.math.reduce_sum(state))
        # variational regularization
        dist_pred = tfp.distributions.MultivariateNormalDiag(
                loc=z_pred_mean, 
                scale_diag=z_pred_std,
                )
        dist_prior = tfp.distributions.MultivariateNormalDiag(
                loc=tf.math.multiply(tf.ones(z_pred_mean.shape), vi_prior_mean),
                scale_diag=tf.math.multiply(tf.ones(z_pred_std.shape), vi_prior_std),
                )
        variational_regularization_loss = tfp.distributions.kl_divergence(dist_pred, dist_prior)
        embedding_loss = reconstruction_loss + variational_regularization_loss
        return state, embedding_loss, count_tile

    def compute_loss(self, dataset_list_0, dataset_list_1):
        agg_embedding_loss = 0
        agg_count_tile = 0
        # before perturbation
        state_list = []
        for dataset in dataset_list_0:
            state, embedding_loss, count_tile = self.compute_state(dataset)
            state_list.append(state)
            agg_embedding_loss = tf.math.add(agg_embedding_loss, embedding_loss)
            agg_count_tile = tf.math.add(agg_embedding_loss, embedding_loss)
        state_0 = tf.concat(state_list, axis=0)
        # after perturbation
        state_list = []
        for dataset in dataset_list_1:
            state, embedding_loss, count_tile = self.compute_state(dataset)
            state_list.append(state)
            agg_embedding_loss = tf.math.add(agg_embedding_loss, embedding_loss)
            agg_count_tile = tf.math.add(agg_embedding_loss, embedding_loss)
        state_1 = tf.concat(state_list, axis=0)
        # compute perturbation loss
        state_0_logit = tf.math.log(tf.divide(state_0, tf.math.subtract(
            tf.ones(state_0.shape), state_0)))
        state_1_pred = self.perturbation_net(state_0_logit)
        similarity = tf.keras.losses.cosine_similarity(y_true=state_1, y_pred=state_1_pred)
        dissimilarity_logit = tf.math.log(tf.divide(tf.math.subtract(tf.ones(
            similarity.shape), similarity), similarity))
        # compute losses
        avg_embedding_loss = tf.math.divide(agg_embedding_loss, agg_count_tile)
        perturbation_loss = tf.math.reduce_mean(dissimilarity_logit)
        return avg_embedding_loss, perturbation_loss

    def compute_apply_gradients(self, dataset_list_0, dataset_list_1, optimizer):
        with tf.GradientTape() as tape:
            embedding_loss, perturbation_loss = self.compute_loss(dataset_list_0, dataset_list_1)
            loss = tf.math.add(embedding_loss, perturbation_loss)
        gradients = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return embedding_loss, perturbation_loss

