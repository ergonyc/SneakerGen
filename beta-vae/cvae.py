"""
This file is the shape encoder model which is a convolutional variational autoencoder.
It contains both the encoder and decoder part of the model.
"""

#%% Imports
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Conv2DTranspose,
    InputLayer,
    Flatten,
    Reshape,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras import regularizers
import numpy as np
import os


KERNEL_SZ = 5
DROPOUT_RATE = 0.15
# INPUT_DIM = 128  # squared
# LATENT_DIM = 128


#%% CVAE Class that extends the standard keras model
class CVAE(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, learning_rate=6e-4, training=True, kl_weight=1, name="autoencoder"):
        super(CVAE, self).__init__()

        self.kl_weight = kl_weight

        self.optimizer = tf.keras.optimizers.Adam(6e-4)
        self.latent_dim = latent_dim
        # WHAT ARE THESE BEING USED FOR...
        self.gen_layers = 5  # from 5
        self.gen_init_size = int(input_dim / (2 ** (self.gen_layers - 1)))
        print(f"gen init size = {self.gen_init_size}")
        self.reshape_channels = 20
        print(f"reshape-channels = {self.reshape_channels}")

        self.training = training
        DROPOUT_RATE = 0.15  # i may not have enouth data for dropout...
        REGULARIZE_FACT = 0.001

        self.enc_model = tf.keras.Sequential()
        self.enc_model.add(InputLayer(input_shape=(input_dim, input_dim, 3)))
        self.enc_model.add(
            Conv2D(
                filters=16,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(Dropout(DROPOUT_RATE))
        self.enc_model.add(
            Conv2D(
                filters=32,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(Dropout(DROPOUT_RATE))
        self.enc_model.add(
            Conv2D(
                filters=64,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(Dropout(DROPOUT_RATE))
        self.enc_model.add(
            Conv2D(
                filters=128,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(Dropout(DROPOUT_RATE))
        self.enc_model.add(
            Conv2D(
                filters=256,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        # this is the latent represention...
        self.enc_model.add(Flatten())
        self.enc_model.add(Dense(latent_dim + latent_dim))

        # ENCODE ^^^^
        #########################------------------------------
        # DECODE vvvv

        self.gen_model = tf.keras.Sequential()
        self.gen_model.add(InputLayer(input_shape=(latent_dim,)))
        self.gen_model.add(
            Dense(
                units=(self.gen_init_size ** 2) * 3 * self.reshape_channels,
                activation=tf.nn.relu,
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(
            Reshape(target_shape=(self.gen_init_size, self.gen_init_size, 3 * self.reshape_channels))
        )
        self.gen_model.add(
            Conv2DTranspose(
                filters=256,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(Dropout(DROPOUT_RATE))
        self.gen_model.add(
            Conv2DTranspose(
                filters=128,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(Dropout(DROPOUT_RATE))
        self.gen_model.add(
            Conv2DTranspose(
                filters=64,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(Dropout(DROPOUT_RATE))
        self.gen_model.add(
            Conv2DTranspose(
                filters=32,
                kernel_size=KERNEL_SZ,
                strides=2,
                padding="SAME",
                activation="relu",
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(Dropout(DROPOUT_RATE))
        self.gen_model.add(Conv2DTranspose(filters=3, kernel_size=KERNEL_SZ, strides=1, padding="SAME"))

    def reconstruct(self, train_x, training):
        temp_training = self.training
        self.training = training
        mean, logvar = self.encode(train_x)
        z = self.reparameterize(mean, logvar)
        x_logits = self.decode(z)
        probs = tf.sigmoid(x_logits)
        self.training = temp_training
        return probs

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(5, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, reparam=False):
        #AH: changed axis=1 from axis=-1
        mean, logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=1)
        if reparam:
            return self.reparameterize(mean, logvar)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.gen_model(z, training=self.training)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.math.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.math.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_test_loss(self, x):
        """
        TODO: change to compute_test_cost
            this is actually "cost" not loss since its across the batch...
        """
        temp_training = self.training
        self.training = False
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.math.reduce_sum(cross_ent, axis=[1, 2, 3])  # JAH removed 3D axis=[1, 2, 3,4]
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)

        kl_divergence = logqz_x - logpz
        neg_log_likelihood = -logpx_z

        elbo = tf.math.reduce_mean(-self.kl_weight * kl_divergence - neg_log_likelihood)  # shape=()
        self.training = temp_training
        return -elbo

    @tf.function
    def compute_loss(self, x):
        """
        TODO: change to compute_test_cost
            this is actually "cost" not loss since its across the batch...
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recons_logits = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_recons_logits, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])  # JAH removed 3D axis=[1, 2, 3,4]

        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mu, logvar)
        kl_divergence = logqz_x - logpz

        neg_log_likelihood = -logpx_z
        #return -tf.reduce_mean(logpx_z + logpz - logqz_x)
        elbo = tf.math.reduce_mean(-self.kl_weight*kl_divergence - neg_log_likelihood)  # shape=()
        return -elbo

    # # vae cost function as negative ELBO
    # @tf.function
    # def vae_cost(self, x_true):
    #     mu, logvar = self.encode(x_true)
    #     z_sample = self.reparameterize(mu,logvar)
    #     x_recons_logits = self.decode(z_sample)
    #     # compute cross entropy loss for each dimension of every datapoint
    #     raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true,
    #                         logits=x_recons_logits)  #
    #     # compute cross entropy loss for all instances in mini-batch; shape=(batch_size,)
    #     neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    #     # through MC approximation with one sample
    
    #     # logpz = normal_log_pdf(z_sample, 0., 1.)  # shape=(batch_size,)
    #     # logqz_x = normal_log_pdf(z_sample, mu, tf.math.square(sd))  # shape=(batch_size,)
    #     logpz = self.log_normal_pdf(z_sample, 0., 0.)  # shape=(batch_size,)
    #     logqz_x = self.log_normal_pdf(z_sample, mu, logvar)  # shape=(batch_size,)
    #     kl_divergence = logqz_x - logpz

    #     elbo = tf.math.reduce_mean(-self.kl_weight * kl_divergence - neg_log_likelihood)  # shape=()
    #     return -elbo


    @tf.function
    def trainStep(self, x):
        with tf.GradientTape() as tape:
            #loss = self.compute_loss(x)
            #cost_mini_batch = self.vae_cost(x)
            cost_mini_batch = self.compute_loss(x)

        gradients = tape.gradient(cost_mini_batch, self.trainable_variables)
        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return cost_mini_batch # loss


    def compile_models(self):
        self.gen_model.compile(optimizer=self.optimizer)
        self.enc_model.compile(optimizer=self.optimizer)

    def setLR(self, learning_rate):
        self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def print_model_summary(self):
        print("Inference Net Summary:\n")
        self.enc_model.summary()
        print("\nGenerative Net Summary:\n")
        self.gen_model.summary()

    def print_model_IO(self):
        print("\nInference Net Summary (input then output):")
        print(self.enc_model.input_shape)
        print(self.enc_model.output_shape)
        print("\nGenerative Net Summary:")
        print(self.gen_model.input_shape)
        print(self.gen_model.output_shape)

    def save_model(self, dir_path, epoch):
        self.enc_model.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.gen_model.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.enc_model.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.gen_model.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))


#%% CVAE Class that extends the standard keras model
#  From the arcitecture shared from Emmanuel Fuentes. examples from GOAT
# Architecture Hyperparameters:
#     Latent Size (research default 256, production default 32)
#     Filter Factor Size (research default 16, production default 32)
#     Latent Linear Hidden Layer Size (research default 2048, production default 1024)
# The encoder architecture is as follows with research defaults from above:
#     Input 3x128x128 (conv2d block [conv2d, batchnorm2d, relu])
#     16x64x64 (conv2d block [conv2d, batchnorm2d, relu])
#     32x32x32 (conv2d block [conv2d, batchnorm2d, relu])
#     64x16x16 (conv2d block [conv2d, batchnorm2d, relu])
#     128x8x8 (conv2d block [conv2d, batchnorm2d, relu])
#     Flatten to 8192
#     2048 (linear block [linear, batchnorm1d, relu])
#     Split the 2048 dimension into mu and log variance for the parameters of the latent distribution
#     Latent mu size 256 (linear layer only with bias)
#     Latent logvar size 256 (linear layer only with bias)


class CVAE_EF(tf.keras.Model):
    def __init__(self, latent_dim, input_dim, learning_rate=6e-4, training=True):
        super(CVAE_EF, self).__init__()

        self.optimizer = tf.keras.optimizers.Adam(6e-4)
        self.latent_dim = latent_dim
        # WHAT ARE THESE BEING USED FOR...
        self.gen_layers = 5  # from 5
        self.gen_init_size = int(input_dim / (2 ** (self.gen_layers - 1)))
        print(f"gen init size = {self.gen_init_size}")
        self.reshape_channels = 128
        print(f"reshape-channels = {self.reshape_channels}")

        self.training = training
        DROPOUT_RATE = 0.15  # i may not have enouth data for dropout...
        REGULARIZE_FACT = 0.001

        self.enc_model = tf.keras.Sequential()
        self.enc_model.add(InputLayer(input_shape=(input_dim, input_dim, 3)))
        self.enc_model.add(
            Conv2D(
                filters=16,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(BatchNormalization())
        self.enc_model.add(
            Conv2D(
                filters=32,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(BatchNormalization())
        self.enc_model.add(
            Conv2D(
                filters=64,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(BatchNormalization())
        self.enc_model.add(
            Conv2D(
                filters=128,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(BatchNormalization())
        # self.enc_model.add(
        #     Conv2D(
        #         filters=256,
        #         kernel_size=KERNEL_SZ,
        #         strides=(2, 2),
        #         padding="SAME",
        #         activation="relu",
        #         kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
        #     )
        # )
        # this is the latent represention...
        self.enc_model.add(Flatten())
        self.enc_model.add(
            Dense(
                units=2048,
                activation=tf.nn.relu,
                use_bias=True,
                kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.enc_model.add(BatchNormalization())
        self.enc_model.add(Dense(latent_dim + latent_dim))

        # ENCODE ^^^^
        #########################------------------------------
        # Decoder architecture an exact mirror
        # Input 256
        # 2048 (linear block [linear, relu])
        # 8192 (linear block [linear, batchnorm1d, relu])
        # reshape (128x8x8)
        # 64x16x16 (conv2d transpose block [convtranspose2d, batchnorm2d, relu])
        # 32x32x32 (conv2d transpose block [convtranspose2d, batchnorm2d, relu])
        # 16x64x64 (conv2d transpose block [convtranspose2d, batchnorm2d, relu])
        # 3x128x128 (conv2d transpose [convtranspose2d, sigmoid)
        # DECODE vvvv

        self.gen_model = tf.keras.Sequential()
        self.gen_model.add(InputLayer(input_shape=(latent_dim,)))

        self.gen_model.add(
            Dense(
                units=2048,
                activation=tf.nn.relu,
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )

        self.gen_model.add(
            Dense(
                units=self.gen_init_size * self.gen_init_size * self.reshape_channels,
                activation=tf.nn.relu,
                #    kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(BatchNormalization())
        self.gen_model.add(
            Reshape(target_shape=(self.gen_init_size, self.gen_init_size, self.reshape_channels))
        )
        # self.gen_model.add(
        #     Conv2DTranspose(
        #         filters=256,
        #         kernel_size=KERNEL_SZ,
        #         strides=(2, 2),
        #         padding="SAME",
        #         activation="relu",
        #         kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
        #     )
        # )
        # self.gen_model.add(BatchNormalization())
        # self.gen_model.add(
        #     Conv2DTranspose(
        #         filters=128,
        #         kernel_size=KERNEL_SZ,
        #         strides=(2, 2),
        #         padding="SAME",
        #         activation="relu",
        #         kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
        #     )
        # )
        self.gen_model.add(BatchNormalization())
        self.gen_model.add(
            Conv2DTranspose(
                filters=64,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(BatchNormalization())
        self.gen_model.add(
            Conv2DTranspose(
                filters=32,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(BatchNormalization())
        # self.gen_model.add(Conv2DTranspose(filters=3, kernel_size=KERNEL_SZ, strides=(1, 1), padding="SAME"))
        self.gen_model.add(
            Conv2DTranspose(
                filters=16,
                kernel_size=KERNEL_SZ,
                strides=(2, 2),
                padding="SAME",
                activation="relu",
                # kernel_regularizer=regularizers.l2(REGULARIZE_FACT),
            )
        )
        self.gen_model.add(BatchNormalization())
        self.gen_model.add(
            Conv2DTranspose(
                filters=3, kernel_size=KERNEL_SZ, strides=(2, 2), padding="SAME", activation="sigmoid",
            )
        )
        # self.gen_model.add(Conv2DTranspose(filters=3, kernel_size=KERNEL_SZ, strides=(1, 1), padding="SAME"))

    def reconstruct(self, train_x, training):
        temp_training = self.training
        self.training = training
        mean, logvar = self.encode(train_x)
        z = self.reparameterize(mean, logvar)
        x_logits = self.decode(z)
        probs = tf.sigmoid(x_logits)
        self.training = temp_training
        return probs

    @tf.function
    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(5, self.latent_dim))
        return self.decode(eps, apply_sigmoid=True)

    def encode(self, x, reparam=False):
        mean, logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=-1)
        if reparam:
            return self.reparameterize(mean, logvar)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean

    def decode(self, z, apply_sigmoid=False):
        logits = self.gen_model(z, training=self.training)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits

    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2.0 * np.pi)
        return tf.reduce_sum(-0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + log2pi), axis=raxis)

    def compute_test_loss(self, x):
        temp_training = self.training
        self.training = False
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])  # JAH removed 3D axis=[1, 2, 3,4]
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        test_loss = -tf.reduce_mean(logpx_z + logpz - logqz_x)
        self.training = temp_training
        return test_loss

    @tf.function
    def compute_loss(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        x_logit = self.decode(z)

        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])  # JAH removed 3D axis=[1, 2, 3,4]
        logpz = self.log_normal_pdf(z, 0.0, 0.0)
        logqz_x = self.log_normal_pdf(z, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)

    @tf.function
    def trainStep(self, x):
        with tf.GradientTape() as tape:
            loss = self.compute_loss(x)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        return loss

    # def compile_models(self):
    #     self.gen_model.compile(optimizer=self.optimizer)
    #     self.enc_model.compile(optimizer=self.optimizer)

    # def setLR(self, learning_rate):
    #     self.optimizer = tf.keras.optimizers.Adam(learning_rate)

    def print_model_summary(self):
        print("Inference Net Summary:\n")
        self.enc_model.summary()
        print("\nGenerative Net Summary:\n")
        self.gen_model.summary()

    def print_model_IO(self):
        print("\nInference Net Summary (input then output):")
        print(self.enc_model.input_shape)
        print(self.enc_model.output_shape)
        print("\nGenerative Net Summary:")
        print(self.gen_model.input_shape)
        print(self.gen_model.output_shape)

    # def plotIO(model):
    #     print("\n Net Summary (input then output):")
    #     enc = tf.keras.utils.plot_model(model.enc_model, show_shapes=True, show_layer_names=True)
    #     gen = tf.keras.utils.plot_model(model.gen_model, show_shapes=True, show_layer_names=True)
    #     return enc, gen

    def save_model(self, dir_path, epoch):
        self.enc_model.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.gen_model.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.enc_model.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.gen_model.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def restoreLatestMyModel(self, dir_path):
        latest = tf.train.latest_checkpoint(dir_path)
        self.model.load_weights(latest)


# %%
