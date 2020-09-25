import numpy as np
import typing
import time

import tensorflow as tf

# convenient abbreviations
tfk = tf.keras
tfka = tfk.activations
tfkl = tf.keras.layers

# adversarial convolutional autoencoder
class AAE(object):
    def __init__(self, 
            latent_dim: int, # latent dimension of style vector
            num_label: int, # number of possible labels
            feature_shape: typing.Tuple[int, int, int], # x-y-channel
            MNIST: bool=False, # test flag
            ):
        # initiation of the superclass of CVAE, ie. tf.keras.Model
        super().__init__()
        # set attributes
        self.latent_dim = latent_dim
        self.num_label = num_label
        self.feature_shape = feature_shape
        # download VGG19 first two convolution layers
        print('downloading VGG19...')
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
        print('parsing VGG19...')
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
        # define params
        deconv_shape = (
            int(feature_shape[0]/x_shrink_factor),
            int(feature_shape[1]/y_shrink_factor),
            int(forward_layers[-2].filters),
            )
        reconstruct_filter = feature_shape[2]
        reconstruct_kernel_size = (3, 3)
        reconstruct_strides = (1, 1)
        # define networks
        print('building networks...')
        self.inference_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=feature_shape)]\
                + forward_layers\
                + [tfkl.Flatten()]\
                + [tfkl.Dense(units=latent_dim+num_label)])
        self.generation_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=(latent_dim+num_label,))]\
                + [tfkl.Dense(
                        units=deconv_shape[0]*deconv_shape[1]*deconv_shape[2],
                        activation=tfka.relu)]\
                + [tfkl.Reshape(target_shape=deconv_shape)]\
                + backward_layers\
                + [tfkl.Conv2DTranspose( # final deconvolution layer
                    filters=reconstruct_filter,
                    kernel_size=reconstruct_kernel_size,
                    strides=reconstruct_strides,
                    padding='same',
                    activation=tfka.sigmoid)])
        self.regularization_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=(latent_dim,))]\
                + [tfkl.Dense(units=32, activation=tfka.relu)]\
                + [tfkl.Dense(units=64, activation=tfka.relu)]\
                + [tfkl.Dense(units=2, activation=tfka.softmax)])
        self.classification_net = tfk.Sequential(
                [tfkl.InputLayer(input_shape=(num_label,))]\
                + [tfkl.Dense(units=32, activation=tfka.relu)]\
                + [tfkl.Dense(units=64, activation=tfka.relu)]\
                + [tfkl.Dense(units=2, activation=tfka.softmax)])
        # set weight and none-trainable
        # note: new model layers count skips InputLayer
        print('copying weights from VGG19...')
        for index in range(1, 1+len(forward_layers)):
            if isinstance(vgg_model.layers[index], tfkl.Conv2D):
                self.inference_net.layers[index-1].set_weights(vgg_model.layers[index].get_weights())
                self.inference_net.layers[index-1].trainable = False

    def infer(self, x):
        vector_pred = self.inference_net(x)
        z_pred, y_pred_logit = tf.split(vector_pred, [self.latent_dim, self.num_label], axis=1)
        y_pred = tfka.softmax(y_pred_logit)
        return z_pred, y_pred

    def generate(self, z_pred, y_pred):
        vector_pred = tf.concat([z_pred, y_pred], axis=1)
        return self.generation_net(vector_pred)

    def regularize(self, z_pred):
        N = z_pred.shape[0]
        normal_sample = tf.random.normal(
                shape=z_pred.shape,
                mean=0.0,
                stddev=1.0,
                )
        positive_pred = self.regularization_net(normal_sample)
        negative_pred = self.regularization_net(z_pred)
        positive_label = tf.one_hot(tf.ones([N], dtype=tf.int32), depth=2)
        negative_label = tf.one_hot(tf.zeros([N], dtype=tf.int32), depth=2)
        discrimination_loss = tfk.losses.categorical_crossentropy(positive_label, positive_pred)\
                + tfk.losses.categorical_crossentropy(negative_label, negative_pred)
        confusion_loss = tfk.losses.categorical_crossentropy(positive_label, negative_pred)\
                + tfk.losses.categorical_crossentropy(negative_label, positive_pred)
        return discrimination_loss, confusion_loss

    def classify(self, y_pred):
        N = y_pred.shape[0]
        categorical_sample = tf.random.categorical(
                logits=tf.math.log(y_pred),
                num_samples=1, # per row
                )
        onehot_sample = tf.one_hot(categorical_sample, depth=self.num_label)
        positive_pred = self.classification_net(onehot_sample)
        negative_pred = self.classification_net(y_pred)
        positive_label = tf.one_hot(tf.ones([N], dtype=tf.int32), depth=2)
        negative_label = tf.one_hot(tf.zeros([N], dtype=tf.int32), depth=2)
        discrimination_loss = tfk.losses.categorical_crossentropy(positive_label, positive_pred)\
                + tfk.losses.categorical_crossentropy(negative_label, negative_pred)
        confusion_loss = tfk.losses.categorical_crossentropy(positive_label, negative_pred)\
                + tfk.losses.categorical_crossentropy(negative_label, positive_pred)
        return discrimination_loss, confusion_loss

    def compute_loss(self, x, context):
        N = x.shape[0]
        z_pred, y_pred = self.infer(x)
        x_pred = self.generate(z_pred, y_pred)
        reconstruction_loss = tfk.losses.MSE(x, x_pred)
        if context == 'reconstruction':
            return reconstruction_loss
        regularization_loss = self.regularize(z_pred)
        classification_loss = self.classify(y_pred)
        discrimination_loss = regularization_loss[0] + classification_loss[0]
        confusion_loss = regularization_loss[1] + classification_loss[1]
        if context == 'discrimination':
            return discrimination_loss
        elif context == 'confusion':
            return confusion_loss
        elif context == 'evaluation':
            return reconstruction_loss, discrimination_loss, confusion_loss

    def compute_apply_gradients(self, x, context, optimizer_dict):
        if context == 'reconstruction':
            var_list = self.inference_net.trainable_variables + self.generation_net.trainable_variables
            optimizer_dict[context].minimize(
                    loss=lambda: self.compute_loss(x, context=context),
                    var_list=var_list,
                    )
        elif context == 'discrimination':
            var_list =self.regularization_net.trainable_variables\
                    + self.classification_net.trainable_variables
            optimizer_dict[context].minimize(
                    loss=lambda: self.compute_loss(x, context=context),
                    var_list=var_list,
                    )
        elif context == 'confusion':
            var_list = self.inference_net.trainable_variables
            optimizer_dict[context].minimize(
                    loss=lambda: self.compute_loss(x, context=context),
                    var_list=var_list,
                    )

