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


#%% BCVAE Class that extends the standard keras model
class BCVAE(tf.keras.Model):
    def __init__(self, latent_dim=128, input_dim=224, learning_rate=4e-4, training=True, beta=1.0, name="beta-autoencoder"):
        super(BCVAE, self).__init__()

        self.beta = beta # default beta = 1.0 is a vanilla VAE
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.latent_dim = latent_dim
        self.pixel_dim = input_dim * input_dim * 3
        self.beta_norm = self.beta *self.latent_dim/self.pixel_dim # compute the Higgins et al quantity

        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.kl_mc= tf.keras.metrics.Mean(name="kl")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")
        self.kl_analytic = tf.keras.metrics.Mean(name="kla")
        
        # WHAT ARE THESE BEING USED FOR...
        self.gen_layers = 5  # from 5
        self.gen_init_size = int(input_dim / (2 ** (self.gen_layers - 1)))
        self.reshape_channels = 20

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



    def compile(self, optimizer, **kwargs):
        super(BCVAE, self).compile(**kwargs)
        self.gen_model.compile(optimizer=self.optimizer)
        self.enc_model.compile(optimizer=self.optimizer)        
        #self.loss_fn = loss_fn


    def encode(self, x, reparam=False):
        z_mu, z_logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=1)
        if reparam:
            z = self.reparameterize(z_mu, z_logvar)
            return  z, z_mu, z_logvar
        return z_mu, z_logvar


    # def encode(self, x):
    #     z_mu, rho = tf.split(self.enc_model(x), num_or_size_splits=2, axis=1)
    #     eps_dims = z_mu.shape
    #     z_sd = tf.math.log(1+tf.math.exp(rho))
    #     epsilon = tf.random.normal(shape=eps_dims)
    #     z_sample = z_mu + z_sd * epsilon
    #     return z_sample, z_mu, z_sd


    def reparameterize(self, mu, logvar):
        eps = tf.random.normal(shape=mu.get_shape())
        #sig = tf.math.exp(0.5*logvar) 
        #log_sig = logvar*0.5  #sqrt
        # if batchn := mu.shape[0] < 32:
        #     eps = tf.slice(eps,[0,0],[32-batchn,self.latent_dim])
        return eps * tf.exp(logvar * 0.5) + mu

    def decode(self, z, apply_sigmoid=False):
        logits = self.gen_model(z, training=self.training)
        if apply_sigmoid:
            probs = tf.math.softmax(logits)
        return logits

    
    def mylog_normal_pdf(self, sample, mu, logvar, raxis=1): #raxis is the latent_dim
        log2pi = tf.math.log(2. * np.pi)
        #sig_sq_inv = 1/sig_sq = 1/(sig*sig) = 1/sig_sq
        #sig_sq_inv = tf.math.exp(-logvar)
        # sig*sig = tf.math.exp(logvar)
        # logvar = log(sig*sig)  = 2* log(sig)
        # exp(.5*logvar) = sig
        # 2.0*tf.math.log(sig) = tf.math.log(sig*sig) = tf.math.log(sig_sq) = logvar
        #    0.5*2.0*tf.math.log(sig) = 0.5*logvar
        #       sig = tf.math.exp(tf.math.log(sig)) = tf.math.exp(0.5*logvar) 
        # For a single observed value x, the log-likelihood is:
        #  ll = -tf.math.log(sig) - 0.5*log2pi -0.5*tf.math.square(sample-mu)/sig_sq 
        #  ll = -0.5 * ( 2. * tf.math.log(sig) + log2pi + tf.math.square(sample-mu)*sig_sq_inv )
        #  ll = -0.5 * ( logvar + log2pi + tf.math.square(sample - mu)*sig_sq_inv)
        # -------------------------------------
        # for a sample of observed values X = x_1, .., x_n , ll_i = ll(x_i)
        # ll = tf.reduce_sum( ll_i ) = -n*tf.math.log(sig) - 0.5*n*log2pi - 0.5*sig_sq_inv*tf.reduce_sum(tf.math.square(sample-mu))
        # ll = -0.5*n* ( 2*tf.math.log(sig) - log2pi ) - 0.5*sig_sq_inv*tf.reduce_sum(tf.math.square(sample-mu))
        return tf.reduce_sum( -.5 * ( logvar + log2pi + tf.math.square(sample - mu) * tf.math.exp(-logvar) ),
                              axis=raxis)


    def log_normal_pdf(self, sample, mean, logvar, raxis=1):
        log2pi = tf.math.log(2. * np.pi)
        return tf.reduce_sum(
            -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
            axis=raxis)

    def compute_loss(self, x):
        """
        TODO: change to compute_test_cost
            this is actually "cost" not loss since its across the batch...
        """
        z_sample, z_mu, z_logvar = self.encode(x,reparam=True)

        # why is the negative in the reduce_mean?
        # kl_div_a =  - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd)) 
        #                                         - tf.math.square(mu) 
        #                                         - tf.math.square(sd),   axis=1)
        kl_div_a = - 0.5 * tf.math.reduce_sum(1 + z_logvar 
                                                - tf.math.square(z_mu) 
                                                - tf.math.exp(z_logvar), axis=1)

                                                
        x_recons = self.decode(z_sample,apply_sigmoid=True)
        #x_logits = self.decode(z_sample)
        # z_mu, z_logvar = self.encode(x)

        # z = self.reparameterize(z_mu, z_logvar)
        # x_recons = self.decode(z,apply_sigmoid=True)
        
        # log_likelihood log normal is MSE
        # loss is [0, 255]
        # mse = 0.00392156862745098* tf.math.squared_difference(255.*x,255.*x_recons)# 0.00392156862745098 - 1/255.
        mse = tf.math.squared_difference(x,x_recons)

        # for images the neg LL is the MSE
        neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])

        # # compute reverse KL divergence, either analytically 
        # # MC KL:         # or through MC approximation with one sample
        # logpz = self.log_normal_pdf(z, 0., 0.) #standard lognormal: mu = 0. logvar=0.
        # logqz_x = self.log_normal_pdf(z, z_mu, z_logvar)
        # kl_div_mc = logqz_x - logpz
        
        # def normal_log_pdf(sample, mean, logvar, raxis=1):
        #     log2pi = tf.math.log(2. * np.pi)
        #     return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


        def analytic_kl(sample, mean, logvar, raxis=1):
            # log((qz||x)/pz = difference in the log of the gaussian PDF
            log2pi = tf.math.log(2. * np.pi)
            logpz = tf.reduce_sum( -.5 * ((sample*sample) + log2pi),axis=raxis)
            logqz_x = tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
            return logqz_x - logpz

        kl_div_mc = analytic_kl(z_sample, z_mu, z_logvar)  # shape=(batch_size,)
        

        # # analytic KL for representing SIGMA (log_sig)
        # # kl_div_a = - 0.5 * tf.math.reduce_sum(
        # #                                 1 + 0.5*z_logvar - tf.math.square(z_mu) - tf.math.exp(0.5*z_logvar), axis=1)
        # # KL for representing the VARIANCE
        # kl_div_a = - 0.5 * tf.math.reduce_sum(
        #                                 1 + z_logvar - tf.math.square(z_mu) - tf.math.exp(z_logvar), axis=1)

        elbo = tf.math.reduce_mean(-self.beta * kl_div_a - neg_log_likelihood)  # shape=()
        kl = tf.math.reduce_mean(kl_div_mc)  # shape=()
        nll = tf.math.reduce_mean(neg_log_likelihood)  # shape=()
        kla = tf.math.reduce_mean(kl_div_a)  # shape=()
        
        return (-elbo, kl, nll, kla)  #negative ELBO



    # def call(self, x_input):
    #     z_mu, z_logvar = self.encode(x)
    #     z = self.reparameterize(z_mu, z_logvar)
    #     x_recons_logits = self.decode(z)

    #     sigsq = tf.math.exp(logvar)
    #     #mse = tf.reduce_sum(tf.math.square(output-input_img))
    #     neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
    #     kl_divergence = - 0.5 * tf.math.reduce_sum(1+tf.math.log(sigsq)-tf.math.square(mu)-sigsq, axis=1)

    #     # CVAE is inherited from tfk.Model, thus have class method add_loss()
    #     self.add_loss( self.kl_weight * kl_divergence)
    #     return x_recons_logits

    @tf.function
    def train_step(self, x):
        if isinstance(x, tuple):  # should always be
            x = x[0]
        with tf.GradientTape() as tape:
            #loss = self.compute_loss(x)
            #cost_mini_batch = self.vae_cost(x)
            cost_mini_batch, kl, nll, kla = self.compute_loss(x)

        gradients = tape.gradient(cost_mini_batch, self.trainable_variables)
        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return cost_mini_batch # loss
        # Compute our own metrics    
        
        self.elbo_tracker.update_state(cost_mini_batch) 
        self.kl_mc.update_state(kl)
        self.nll_tracker.update_state(nll)
        self.kl_analytic.update_state(kla)

        return {m.name: m.result() for m in self.metrics}


    #@tf.function
    def test_step(self, x):
        temp_training = self.training
        self.training = False

        if isinstance(x, tuple):  # should always be
            x = x[0]

        cost_mini_batch, kl, nll, kla = self.compute_loss(x)
        
        self.elbo_tracker.update_state(cost_mini_batch) 
        self.kl_mc.update_state(kl)
        self.nll_tracker.update_state(nll)
        self.kl_analytic.update_state(kla)

        self.training = temp_training
        return {m.name: m.result() for m in self.metrics}
    

    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.x
        # If you don't implement this property, you have to call
        # `reset_states()` yourself at the time of your choosing.
        return [self.elbo_tracker, self.kl_mc, self.kl_analytic, self.nll_tracker]


    def reset_metrics(self):
        # `reset_states()` yourself at the time of your choosing.
        for m in self.metrics:
            m.reset_states()


    def reconstruct(self, train_x, training):
        temp_training = self.training
        self.training = training

        # mean, logvar = self.encode(train_x)
        # z = self.reparameterize(mean, logvar)
        # probs = self.decode(z,apply_sigmoid = True)
        # do the long way to avoid @tf.function
        mean, logvar = tf.split(self.enc_model(train_x, training=self.training), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=(self.latent_dim,))
        z = eps * tf.exp(logvar * 0.5) + mean        
        logits = self.gen_model(z)
        probs = tf.math.softmax(logits)
        
        self.training = temp_training
        return probs

    def predict(self, images):
        """
        returns logits [-inf inf] "reconstruct returns probability (0,1)
        """
        # mu, logvar = self.encode(images)
        # z = self.reparameterize(mu, logvar)
        # reconst_images = self.decode(z)
        # do the long way to avoid @tf.function
        mean, logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=1)
        eps = tf.random.normal(shape=(self.latent_dim,))
        z = eps * tf.exp(logvar * 0.5) + mean
        return self.gen_model(z)


    def sample(self, eps=None):
        if eps is None:
            eps = tf.random.normal(shape=(5, self.latent_dim))  #the 5 is the "monte carlo... but maybe it should be larger"
        logits = self.gen_model(eps, training=False)
        return tf.math.softmax(logits)


#######################
## UTILITIES...
#######################

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
        

    def plot_model(self,encoder=True):
        """
        by default print teh encoder. print generator if False
        """
        if encoder:
            print("\n Summary (encoder):")
            return tf.keras.utils.plot_model(enc_model, show_shapes=True, show_layer_names=True)
        else:
            print("\n Summary (generator):")
            return tf.keras.utils.plot_model(gen_model, show_shapes=True, show_layer_names=True)
          

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


# %%
