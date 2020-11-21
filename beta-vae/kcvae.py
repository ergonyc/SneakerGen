"""
tf prob for the variational encoder  using Keras Sequential
"""

#%% Imports

import tensorflow as tf
import tensorflow_probability as tfp

# import tensorflow_probability.layers as tfpl
# import tensorflow.keras.layers as tfkl

# is this a better way to import shortcuts??
tfk = tf.keras
tfpl = tfp.layers
tfkl = tf.keras.layers

import functools

import numpy as np
import os


KERNEL_SZ = 5
DROPOUT_RATE = 0.15
# INPUT_DIM = 128  # squared
# LATENT_DIM = 128
DROPOUT_RATE = 0.15  # i may not have enouth data for dropout...
REGULARIZE_FACT = 0.001

#%% PCVAE Class that extends the standard keras model

# TODO: change these to "partial" functions instead of clunky wrappers

def Conv2D(
    input_shape=None,
    filters=32,
    kernel_size=KERNEL_SZ,
    strides=2,
    padding="SAME",
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT),
):
    """
    Local wrapper for tf.keras.Conv2D units where only the filters
    is changing    
    """
    if input_shape is None:
        return tfkl.Conv2D(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,  # (strides,strides)?
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
    else:
        return tfkl.Conv2D(
            filters=filters,
            input_shape=input_shape,
            kernel_size=kernel_size,
            strides=strides,  # (strides,strides)?
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )


def Conv2DTranspose(
    input_shape=None,
    filters=32,
    kernel_size=KERNEL_SZ,
    strides=2,
    padding="SAME",
    activation="relu",
    kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT),
):
    """
    Local wrapper for tf.keras.Conv2DTranspose units where only the filters
    is changing    
    """
    if input_shape is None:
        return tfkl.Conv2DTranspose(
            filters=filters,
            kernel_size=kernel_size,
            strides=strides,  # (strides,strides)?
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )
    else:
        return tfkl.Conv2DTranspose(
            filters=filters,
            input_shape=input_shape,
            kernel_size=kernel_size,
            strides=strides,  # (strides,strides)?
            padding=padding,
            activation=activation,
            kernel_regularizer=kernel_regularizer,
        )

############################33
#
#   make the models
#
###################################

############################33
#
#   make the models
#
###################################

def encoder_z_KL_reg(dim_z=64,dim_x=(192,192,3), kl_weight=1.0):
    # self.encoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='encoder_input')
    # decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(dim_z), scale=1),
                    reinterpreted_batch_ndims=1)

    encoder_z = tf.keras.Sequential([
        Conv2D(filters=16, input_shape=dim_x),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=32),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=64),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=128),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=256),
        tfkl.Dropout(DROPOUT_RATE),
        tfkl.Flatten(),
        # might want to use IndependentNormal vs multivariate?  easier to disentangle?
        # tfkl.Dense(tfp.layers.MultivariateNormalTriL.params_size(dim_z),
        #                 activation=None, 
        #                 name='z_params'),
        # tfpl.MultivariateNormalTriL(dim_z,
        #                 convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        #                 activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=kl_weight)),
        #                 name='z_layer'),
        tfkl.Dense(tfpl.IndependentNormal.params_size(dim_z), 
                                 activation=None, name='z_params',),
        tfpl.IndependentNormal(dim_z, 
            convert_to_tensor_fn=tfp.distributions.Distribution.sample, 
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=kl_weight), 
            name='z_layer'),
        ],        
        name="encoder",)
    
    return encoder_z
 
def encoder_z(dim_z=64, dim_x=(192,192,3), kl_weight=1.0):
    # self.encoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='encoder_input')
    # decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(dim_z), scale=1),
                    reinterpreted_batch_ndims=1)

    encoder_z = tf.keras.Sequential([
        Conv2D(filters=16, input_shape=dim_x),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=32),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=64),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=128),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2D(filters=256),
        tfkl.Dropout(DROPOUT_RATE),
        tfkl.Flatten(),
        # might want to use IndependentNormal vs multivariate?  easier to disentangle?
        # tfkl.Dense(tfp.layers.MultivariateNormalTriL.params_size(dim_z),
        #                 activation=None, 
        #                 name='z_params'),
        # tfpl.MultivariateNormalTriL(dim_z,
        #                 convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        #                 activity_regularizer=None, 
        #                 name='z_layer'),
        # tfpl.KLDivergenceAddLoss(
        #     tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(dim_z)),
        #     weight = kl_weight ,name="kl_loss"), #scale defaults to identity
        tfkl.Dense(tfpl.IndependentNormal.params_size(dim_z), 
                                 activation=None, name='z_params',),
        tfpl.IndependentNormal(dim_z, 
            convert_to_tensor_fn=tfp.distributions.Distribution.sample, 
            activity_regularizer=None,#tfpl.KLDivergenceRegularizer(prior, weight=kl_weight), 
            name='z_layer'),
        tfpl.KLDivergenceAddLoss(
            prior, #tfp.distributions.Normal(loc=tf.zeros(dim_z), scale=1),
            #test_points_fn=tf.convert_to_tensor,
            weight = kl_weight ,
            name="kl_loss"), #scale defaults to identity
        ],        


        name="encoder",)

    return encoder_z


def decoder_x(dim_z=64,dim_x=(192,192,3)):
    """
    """
    n_layers = 5
    pix_dim = dim_x[0]
    init_dim = pix_dim//(2** (n_layers-1))

    decoder_x = tf.keras.Sequential([
        tfkl.InputLayer(input_shape=dim_z),
        tfkl.Reshape((1, 1, dim_z)),
        tfkl.Conv2DTranspose(
            filters=256,
            kernel_size=init_dim,
            strides=1,  # (strides,strides)?
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT),
            ),
        Conv2DTranspose(filters=128),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2DTranspose(filters=64),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2DTranspose(filters=32),
        tfkl.Dropout(DROPOUT_RATE),
        Conv2DTranspose(filters=16),
        tfkl.Conv2DTranspose(
            filters=3, kernel_size=KERNEL_SZ, strides=1, 
            padding="SAME", activation=None
            ),

        # note that here we don't need 
        # `tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.dim_x))` because 
        # we've restored the desired input shape with the last Conv2DTranspose layer
        # tfpl.IndependentBernoulli(self.dim_x, name='x_layer'),
        # OR:
        # layers.append(tfpl.IndependentNormal(self.dim_x, name='x_layer', 
        #     convert_to_tensor_fn=tfd.Distribution.sample, 
        #     activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=self.kl_weight),
        # ), 
        ],
        name="decoder",)

    return decoder_x




class K_PCVAE(tf.keras.Model):
    def __init__(self, dim_z, dim_x, learning_rate, kl_weight=1, name="autoencoder", **kwargs):
        super(K_PCVAE, self).__init__(name=name, **kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight  # beta

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.encoder = encoder_z(dim_z = dim_z, dim_x=dim_x ,kl_weight=1.0) #do weighting in trainstep

        self.decoder = decoder_x(dim_z = dim_z, dim_x=dim_x )

        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.kl_mc= tf.keras.metrics.Mean(name="kl")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")
        self.kl_analytic = tf.keras.metrics.Mean(name="kla")

    def build_vae_keras_model(self):
        x_input = tfk.Input(shape=self.dim_x)
        encoder = encoder_z()
        decoder = decoder_x()
        z = encoder(x_input)

        model = tfk.Model(inputs=x_input, outputs=decoder(z))
        # # compile VAE model
        # model.compile(loss=negative_log_likelihood, 
        #               optimizer=tfk.optimizers.Adam(self.learning_rate))
        return model

    # def encode_and_decode(self, x_input):
    def call(self, x_input):
        if isinstance(x_input, tuple):  # should always be
            x_input = x_input[0]
        z = self.encoder(x_input)
        x_logits = self.decoder(z)
        # this is the raw output... no "activation" has been applied...
        x_hat = tf.math.sigmoid(x_logits)
        return x_hat


    # vae loss function -- only the negative log-likelihood part,
    # since we use add_loss for the KL divergence part
    def partial_vae_loss(self, x_true,x_pred=None):
        # x_recons_logits = model.encode_and_decode(x_true)
        if x_pred is None:
            x_pred = self(x_true) # pass through the encoder/decoder
        mse = tf.math.squared_difference(tf.cast(x_pred,tf.float32), 
                                        tf.cast(x_true,tf.float32))
        # z = self.encoder(x_true)
        # x_logits = self.decoder(z)
        # # compute cross entropy loss for each dimension of every datapoint
        # # change this to MSE
        # mse = tf.math.squared_difference(tf.keras.activations.sigmoid(x_logits), x_true)
        neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
        return tf.math.reduce_mean(neg_log_likelihood)


    @tf.function
    def train_step(self, x):
        if isinstance(x, tuple):  # should always be
            x = x[0]
        with tf.GradientTape() as tape:
            #loss = self.compute_loss(x)
            #cost_mini_batch = self.vae_cost(x)
            neg_log_lik = self.partial_vae_loss(x)
                   
            kl_loss_l = self.encoder.get_layer('kl_loss') #self.encoder.kl_lossTFP.losses 
            kl_loss= kl_loss_l.losses[-1]
        
            # kl_loss_l = self.encoder.layers[-1] #self.encoder.kl_lossTFP.losses 
            # kl_loss = kl_loss_l.losses[-1]
            # make sure its the last entry in the list... 
            total_vae_loss = neg_log_lik + self.kl_weight*kl_loss


        gradients = tape.gradient(total_vae_loss, self.trainable_variables)
        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return cost_mini_batch # loss
        # Compute our own metrics    
        
        self.elbo_tracker.update_state(total_vae_loss) 
        self.kl_mc.update_state(kl_loss)
        self.nll_tracker.update_state(neg_log_lik)
        self.kl_analytic.update_state(kl_loss*self.kl_weight)

        return {m.name: m.result() for m in self.metrics}


    #@tf.function
    def test_step(self, x):        
        if isinstance(x, tuple):  # should always be
            x = x[0]
        neg_log_lik = self.partial_vae_loss(x)

        kl_loss_l = self.encoder.get_layer('kl_loss') #self.encoder.kl_lossTFP.losses 
        kl_loss= kl_loss_l.losses[-1]
        # kl_loss_l = self.encoder.layers[-1] #self.encoder.kl_lossTFP.losses 
        # kl_loss = kl_loss_l.losses[-1]
        total_vae_loss = neg_log_lik + kl_loss

        self.elbo_tracker.update_state(total_vae_loss) 
        self.kl_mc.update_state(kl_loss)
        self.nll_tracker.update_state(neg_log_lik)
        self.kl_analytic.update_state(kl_loss*self.kl_weight)

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

    def save_model(self, dir_path, epoch):
        self.encoder.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.encoder.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))




class K_PCVAE_KL_Reg(tf.keras.Model):
    def __init__(self, dim_z, dim_x, learning_rate, kl_weight=1, name="autoencoder", **kwargs):
        super(K_PCVAE_KL_Reg, self).__init__(name=name, **kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.encoder = encoder_z_KL_reg(dim_z=self.dim_z, dim_x=self.dim_x, kl_weight=self.kl_weight)
        self.decoder = decoder_x(dim_z=self.dim_z, dim_x=self.dim_x)

        # self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.kl_reg= tf.keras.metrics.Mean(name="kl_reg")
        # self.nll_tracker = tf.keras.metrics.Mean(name="nll")
        self.kl_reg1 = tf.keras.metrics.Mean(name="kl_reg1")

    # def encode_and_decode(self, x_input):
    def call(self, x_input):
        z = self.encoder(x_input)
        x_logits = self.decoder(z)
        # this is the raw output... no "activation" has been applied...
        x_hat = tf.math.sigmoid(x_logits)
        return x_hat

    # vae loss function -- only the negative log-likelihood part,
    # since we use add_loss for the KL divergence part
    def partial_vae_loss(self, x_true,x_pred=None):
        # x_recons_logits = model.encode_and_decode(x_true)
        if x_pred is None:
            x_pred = self(x_true)
        mse = tf.math.squared_difference(x_pred, x_true)
        # z = self.encoder(x_true)
        # x_logits = self.decoder(z)
        # # compute cross entropy loss for each dimension of every datapoint
        # # change this to MSE
        # mse = tf.math.squared_difference(tf.keras.activations.sigmoid(x_logits), x_true)
        neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
        return tf.math.reduce_mean(neg_log_likelihood)

    #@tf.function
    def train_step(self, x):
        if isinstance(x, tuple):  # should always be
            x = x[0]
        with tf.GradientTape() as tape:
            y = x
            y_pred = self(x, training=True)  # Forward pass
            
            # Compute the loss value
            # (the loss function is configured in `compile()`)
            neg_log_lik = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

            #loss = self.compute_loss(x)
            #cost_mini_batch = self.vae_cost(x)
            # neg_log_lik = self.partial_vae_loss(x)

        gradients = tape.gradient(neg_log_lik, self.trainable_variables)
        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return cost_mini_batch # loss
        # Compute our own metrics    

        kl_loss_l = self.encoder.get_layer('z_layer') #self.encoder.kl_lossTFP.losses 
        kl_reg = kl_loss_l.losses[0]
        kl_reg1 = kl_loss_l.losses[1]
        
        self.kl_reg.update_state(kl_reg)
        self.kl_reg1.update_state(kl_reg1)

        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

        #return neg_log_lik

    #@tf.function
    def test_step(self, x):        
        if isinstance(x, tuple):  # should always be
            x = x[0]

        y = x
        y_pred = self(x, training=True)  # Forward pass
        
        # Compute the loss value
        # (the loss function is configured in `compile()`)
        neg_log_lik = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        # neg_log_lik = self.partial_vae_loss(x)

        kl_loss_l = self.encoder.get_layer('z_layer') #self.encoder.kl_lossTFP.losses 
        kl_reg = kl_loss_l.losses[0]
        kl_reg1 = kl_loss_l.losses[1]
        
        self.kl_reg.update_state(kl_reg)
        self.kl_reg1.update_state(kl_reg1)


        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}

        #return neg_log_lik
    
    def save_model(self, dir_path, epoch):
        self.encoder.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.encoder.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))



##################FUENTES MODEL############

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



def encoder_z_bn(dim_z=64, dim_x=(192,192,3), kl_weight=1.0):
    # self.encoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='encoder_input')
    # decoder_input = Input(shape=(self.hps['max_seq_len'], 5), name='decoder_input')

    prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(dim_z), scale=1),
                    reinterpreted_batch_ndims=1)

    encoder_z = tf.keras.Sequential([
        Conv2D(filters=16, input_shape=dim_x),
        tfkl.BatchNormalization(),
        Conv2D(filters=32),
        tfkl,BatchNormalization(),
        Conv2D(filters=64),
        tfkl,BatchNormalization(),
        Conv2D(filters=128),
        tfkl,BatchNormalization(),
        Conv2D(filters=256),
        tfkl,BatchNormalization(),
        tfkl.Flatten(),
        # might want to use IndependentNormal vs multivariate?  easier to disentangle?
        # tfkl.Dense(tfp.layers.MultivariateNormalTriL.params_size(dim_z),
        #                 activation=None, 
        #                 name='z_params'),
        # tfpl.MultivariateNormalTriL(dim_z,
        #                 convert_to_tensor_fn=tfp.distributions.Distribution.sample,
        #                 activity_regularizer=None, 
        #                 name='z_layer'),
        # tfpl.KLDivergenceAddLoss(
        #     tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(dim_z)),
        #     weight = kl_weight ,name="kl_loss"), #scale defaults to identity
        tfkl.Dense(tfpl.IndependentNormal.params_size(dim_z), 
                                 activation=None, name='z_params',),
        tfpl.IndependentNormal(dim_z, 
            convert_to_tensor_fn=tfp.distributions.Distribution.sample, 
            activity_regularizer=None,#tfpl.KLDivergenceRegularizer(prior, weight=kl_weight), 
            name='z_layer'),
        tfpl.KLDivergenceAddLoss(
            prior, #tfp.distributions.Normal(loc=tf.zeros(dim_z), scale=1),
            #test_points_fn=tf.convert_to_tensor,
            weight = kl_weight ,
            name="kl_loss"), #scale defaults to identity
        ],        


        name="encoder",)

    return encoder_z


def decoder_x_bn(dim_z=64,dim_x=(192,192,3)):
    """
    """
    n_layers = 5
    pix_dim = dim_x[0]
    init_dim = pix_dim//(2** (n_layers-1))

    decoder_x = tf.keras.Sequential([
        tfkl.InputLayer(input_shape=dim_z),
        tfkl.Reshape((1, 1, dim_z)),
        tfkl.Conv2DTranspose(
            filters=256,
            kernel_size=init_dim,
            strides=1,  # (strides,strides)?
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT),
            ),
        Conv2DTranspose(filters=128),
        tfkl,BatchNormalization(),
        Conv2DTranspose(filters=64),
        tfkl,BatchNormalization(),
        Conv2DTranspose(filters=32),
        tfkl,BatchNormalization(),
        Conv2DTranspose(filters=16),
        tfkl.Conv2DTranspose(
            filters=3, kernel_size=KERNEL_SZ, strides=1, 
            padding="SAME", activation=None
            ),

        # note that here we don't need 
        # `tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.dim_x))` because 
        # we've restored the desired input shape with the last Conv2DTranspose layer
        # tfpl.IndependentBernoulli(self.dim_x, name='x_layer'),
        # OR:
        # layers.append(tfpl.IndependentNormal(self.dim_x, name='x_layer', 
        #     convert_to_tensor_fn=tfd.Distribution.sample, 
        #     activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=self.kl_weight),
        # ), 
        ],
        name="decoder",)
 
    return decoder_x

# _BN for BATCH NORM
class K_PCVAE_BN(tf.keras.Model):
    def __init__(self, dim_z, dim_x, learning_rate, kl_weight=1, name="autoencoder", **kwargs):
        super(K_PCVAE_BN, self).__init__(name=name, **kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight  # beta

        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)

        self.encoder = encoder_z(dim_z = dim_z, dim_x=dim_x ,kl_weight=1.0) #do weighting in trainstep

        self.decoder = decoder_x(dim_z = dim_z, dim_x=dim_x )

        self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
        self.kl_mc= tf.keras.metrics.Mean(name="kl")
        self.nll_tracker = tf.keras.metrics.Mean(name="nll")
        self.kl_analytic = tf.keras.metrics.Mean(name="kla")

    def build_vae_keras_model(self):
        x_input = tfk.Input(shape=self.dim_x)
        encoder = encoder_z()
        decoder = decoder_x()
        z = encoder(x_input)

        model = tfk.Model(inputs=x_input, outputs=decoder(z))
        # # compile VAE model
        # model.compile(loss=negative_log_likelihood, 
        #               optimizer=tfk.optimizers.Adam(self.learning_rate))
        return model

    # def encode_and_decode(self, x_input):
    def call(self, x_input):
        if isinstance(x_input, tuple):  # should always be
            x_input = x_input[0]
        z = self.encoder(x_input)
        x_logits = self.decoder(z)
        # this is the raw output... no "activation" has been applied...
        x_hat = tf.math.sigmoid(x_logits)
        return x_hat


    # vae loss function -- only the negative log-likelihood part,
    # since we use add_loss for the KL divergence part
    def partial_vae_loss(self, x_true,x_pred=None):
        # x_recons_logits = model.encode_and_decode(x_true)
        if x_pred is None:
            x_pred = self(x_true)
        mse = tf.math.squared_difference(tf.cast(x_pred,tf.float32), 
                                        tf.cast(x_true,tf.float32))
        # mse = tf.math.squared_difference(x_pred, x_true)
        # z = self.encoder(x_true)
        # x_logits = self.decoder(z)
        # # compute cross entropy loss for each dimension of every datapoint
        # # change this to MSE
        # mse = tf.math.squared_difference(tf.keras.activations.sigmoid(x_logits), x_true)
        neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
        return tf.math.reduce_mean(neg_log_likelihood)


    @tf.function
    def train_step(self, x):
        if isinstance(x, tuple):  # should always be
            x = x[0]
        with tf.GradientTape() as tape:
            #loss = self.compute_loss(x)
            #cost_mini_batch = self.vae_cost(x)
            neg_log_lik = self.partial_vae_loss(x)
                   
            kl_loss_l = self.encoder.get_layer('kl_loss') #self.encoder.kl_lossTFP.losses 
            kl_loss= kl_loss_l.losses[-1]
        
            # kl_loss_l = self.encoder.layers[-1] #self.encoder.kl_lossTFP.losses 
            # kl_loss = kl_loss_l.losses[-1]
            # make sure its the last entry in the list... 
            total_vae_loss = neg_log_lik + self.kl_weight*kl_loss


        gradients = tape.gradient(total_vae_loss, self.trainable_variables)
        #gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        # return cost_mini_batch # loss
        # Compute our own metrics    
        
        self.elbo_tracker.update_state(total_vae_loss) 
        self.kl_mc.update_state(kl_loss)
        self.nll_tracker.update_state(neg_log_lik)
        self.kl_analytic.update_state(kl_loss*self.kl_weight)

        return {m.name: m.result() for m in self.metrics}


    #@tf.function
    def test_step(self, x):        
        if isinstance(x, tuple):  # should always be
            x = x[0]
        neg_log_lik = self.partial_vae_loss(x)

        kl_loss_l = self.encoder.get_layer('kl_loss') #self.encoder.kl_lossTFP.losses 
        kl_loss= kl_loss_l.losses[-1]
        # kl_loss_l = self.encoder.layers[-1] #self.encoder.kl_lossTFP.losses 
        # kl_loss = kl_loss_l.losses[-1]
        total_vae_loss = neg_log_lik + kl_loss

        self.elbo_tracker.update_state(total_vae_loss) 
        self.kl_mc.update_state(kl_loss)
        self.nll_tracker.update_state(neg_log_lik)
        self.kl_analytic.update_state(kl_loss*self.kl_weight)

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

    def save_model(self, dir_path, epoch):
        self.encoder.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.encoder.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))


#####################################
#
#
#########################################
class Sampler_Z(tfkl.Layer):
    """
    Sampling Layer Class for the "reparam" trick
    """
    def __init__(self, dim_z=64, name="sampler", **kwargs):
        super(Sampler_Z, self).__init__(name=name, **kwargs)
        self.dim_z = dim_z

    def call(self, inputs):
        mu, rho = inputs
        sd = tf.math.log(1 + tf.math.exp(rho))
        z_sample = mu + sd * tf.random.normal(shape=sd.shape)
        return z_sample, sd

    # def latent_z(self, encoder_output):
    #     """ Return a latent vector z of size [batch_size]X[z_size] """

    #     def transform2layer(z_params):
    #         """ Auxiliary function to feed into Lambda layer.
    #          Gets a list of [mu, sigma] and returns a random tensor from the corresponding normal distribution """
    #         mu, sigma = z_params
    #         sigma_exp = K.exp(sigma / 2.0)
    #         colored_noise = mu + sigma_exp*K.random_normal(shape=K.shape(sigma_exp), mean=0.0, stddev=1.0)
    #         return colored_noise
    #     # Dense layers to create the mean and stddev of the latent vector
    #     self.mu = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)
    #     self.sigma = Dense(units=self.hps['z_size'], kernel_initializer=RandomNormal(stddev=0.001))(encoder_output)

    #     # We cannot simply use the operations and feed to the next layer, so a Lambda layer must be used
    #     return Lambda(transform2layer)([self.mu, self.sigma])


# deconv = functools.partial(
#       tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)

# conv = functools.partial(
#       tf.keras.layers.Conv2D, 
#         padding="SAME", 
#         activation=activation,
#         filters=32,
#         kernel_size=KERNEL_SZ,
#         strides=2,
#         padding="SAME",
#         activation="relu",
#         kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT
#        )



class Encoder_Z_KL_Reg(tfkl.Layer):
    """
    KL Div is treated as an activity regularizer on the tensorflow Probability encoder
    """
    def __init__(self, dim_z=64, dim_x=(192, 192, 3), name="encoder", **kwargs):
        super(Encoder_Z, self).__init__(name=name, **kwargs)
        self.pix_dim = dim_x[0]
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.dim_z), scale=1),
                        reinterpreted_batch_ndims=1)

        self.dropout_layer = tfkl.Dropout(DROPOUT_RATE)
        self.conv_layer_0 = Conv2D(filters=16, input_shape=self.dim_x)
        self.conv_layer_1 = Conv2D(filters=32)
        self.conv_layer_2 = Conv2D(filters=64)
        self.conv_layer_3 = Conv2D(filters=128)
        self.conv_layer_4 = Conv2D(filters=256)

        self.flatten_layer = tfkl.Flatten()


        self.sampler = tfkl.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.dim_z),activation=None)
           
        self.normalTFP = tfpl.MultivariateNormalTriL(self.dim_z,
                        activity_regularizer=tfpl.KLDivergenceRegularizer(self.prior))

    # Functional
    def call(self, x_input):
        z = self.conv_layer_0(x_input)
        z = self.dropout_layer(z)
        z = self.conv_layer_1(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_2(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_3(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_4(z)
        z = self.flatten_layer(z)

        z = self.sampler(z)
        z = self.normalTFP(z)
        return z


class Decoder_X(tfkl.Layer):
    def __init__(self, dim_z, dim_x, name="decoder", **kwargs):
        super(Decoder_X, self).__init__(name=name, **kwargs)
        # n_layers = 5
        # conv_fact = 2**(n_layers-1) = 16
        # startdim= pix_dim//cfact  = 12 = 3*4
        # 3* 4 *n_layers
        self.dim_z = dim_z
        self.dim_x = dim_x
        self.pix_dim = dim_x[0]

        n_layers = 5
        init_dim = self.pix_dim//(2** (n_layers-1))

        self.input_l = tfkl.InputLayer(input_shape=[self.dim_z])

        # self.dense_z_input = tfkl.Dense(
        #     units=(self.pix_dim // 16) ** 2 * 3 * 4, activation=None, input_shape=self.dim_x
        # )
        #self.reshape_layer = tfkl.Reshape((self.pix_dim // 16, self.pix_dim // 16, 3 * 4))
        self.reshape_layer = tfkl.Reshape((1, 1, self.dim_z))

        self.conv_transpose_layer_start = tfkl.Conv2DTranspose(
            filters=256,
            kernel_size=init_dim,
            strides=1,  # (strides,strides)?
            padding="valid",
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l2(REGULARIZE_FACT),
        )
        self.conv_transpose_layer_4 = tfkl.Conv2DTranspose(
            filters=3, kernel_size=KERNEL_SZ, strides=1, padding="SAME", activation="sigmoid"
        )
        # self.conv_transpose_layer_3b = Conv2DTranspose(filters=8)
        # self.conv_transpose_layer_3a = Conv2DTranspose(filters=16)
        self.conv_transpose_layer_3 = Conv2DTranspose(filters=16)
        self.conv_transpose_layer_2 = Conv2DTranspose(filters=32)
        self.conv_transpose_layer_1 = Conv2DTranspose(filters=64)
        self.conv_transpose_layer_0 = Conv2DTranspose(filters=128)
        self.dropout_layer = tfkl.Dropout(DROPOUT_RATE)

    # Functional
    def call(self, z):
        x_output = self.input_l(z)
        x_output = self.reshape_layer(x_output)
        x_output = self.conv_transpose_layer_start(x_output)
        x_output = self.conv_transpose_layer_0(x_output)
        x_output = self.dropout_layer(x_output)
        x_output = self.conv_transpose_layer_1(x_output)
        x_output = self.dropout_layer(x_output)
        x_output = self.conv_transpose_layer_2(x_output)
        x_output = self.dropout_layer(x_output)
        x_output = self.conv_transpose_layer_3(x_output)
        x_output = self.dropout_layer(x_output)
        # x_output = self.conv_transpose_layer_3a(x_output)
        # x_output = self.dropout_layer(x_output)
        # x_output = self.conv_transpose_layer_3b(x_output)
        # x_output = self.dropout_layer(x_output)
        x_output = self.conv_transpose_layer_4(x_output)

        return x_output


class PCVAE_KL_Reg(tf.keras.Model):
    def __init__(self, dim_z, dim_x, learning_rate, name="autoencoder", **kwargs):
        super(PCVAE_KL_Reg, self).__init__(name=name, **kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z

        self.learning_rate = learning_rate
        self.encoder = Encoder_Z(dim_z=self.dim_z)
        self.decoder = Decoder_X(dim_z=self.dim_z, dim_x=self.dim_x)


    # def encode_and_decode(self, x_input):
    def call(self, x_input):

        z = self.encoder(x_input)
        x_hat = self.decoder(z)

        # z_sample, mu, sd = self.encoder(x_input)
        # # maybe need to put this through a sigmoid??
        # x_recons_logits = self.decoder(z_sample)

        # kl_divergence = -0.5 * tf.math.reduce_sum(
        #     1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd), axis=1
        # )
        # kl_divergence = tf.math.reduce_mean(kl_divergence)
        # # self.add_loss(lambda: self.kl_weight * kl_divergence)
        # self.add_loss(self.kl_weight * kl_divergence)
        return x_hat
    
    def save_model(self, dir_path, epoch):
        self.encoder.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.encoder.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

class Encoder_Z(tfkl.Layer):
    """
    KL Div will be calculated and added as loss
    """
    def __init__(self, dim_z=64, dim_x=(192, 192, 3), kl_weight=1.0, name="encoder", **kwargs):
        super(Encoder_Z, self).__init__(name=name, **kwargs)
        self.pix_dim = dim_x[0]
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.prior = tfp.distributions.Independent(tfp.distributions.Normal(loc=tf.zeros(self.dim_z), scale=1),
                       reinterpreted_batch_ndims=1)


        # tfkl.InputLayer(input_shape=dim_x)

        self.dropout_layer = tfkl.Dropout(DROPOUT_RATE)
        self.conv_layer_0 = Conv2D(filters=16, input_shape=self.dim_x)
        self.conv_layer_1 = Conv2D(filters=32)
        self.conv_layer_2 = Conv2D(filters=64)
        self.conv_layer_3 = Conv2D(filters=128)
        self.conv_layer_4 = Conv2D(filters=256)

        self.flatten_layer = tfkl.Flatten()
        # self.dense_mean = tfkl.Dense(self.dim_z, activation=None, name="z_mean")
        # self.dense_raw_stddev = tfkl.Dense(self.dim_z, activation=None, name="z_raw_stddev")
        # self.sampler_z = Sampler_Z(self.dim_z)

        self.sampler = tfkl.Dense(tfp.layers.MultivariateNormalTriL.params_size(self.dim_z),activation=None)
           
        self.normalTFP = tfpl.MultivariateNormalTriL(self.dim_z,
                        activity_regularizer=None ) #tfpl.KLDivergenceRegularizer(self.prior))
        #self.enc_model.add(tfkl.Dense(latent_dim + l
        self.kl_lossTFP =        tfpl.KLDivergenceAddLoss(
            tfp.distributions.MultivariateNormalDiag(loc=tf.zeros(self.dim_z)),
            weight = self.kl_weight ) #scale defaults to identity
        #self.kl_lossTFP =        tfpl.KLDivergenceAddLoss(self.prior),#weight = self.kl_weight ),

    # Functional
    def call(self, x_input):
        z = self.conv_layer_0(x_input)
        z = self.dropout_layer(z)
        z = self.conv_layer_1(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_2(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_3(z)
        z = self.dropout_layer(z)
        z = self.conv_layer_4(z)
        z = self.flatten_layer(z)

        z = self.sampler(z)
        z = self.normalTFP(z)
        z = self.kl_lossTFP(z)

        return z

        # mu = self.dense_mean(z)
        # rho = self.dense_raw_stddev(z)
        # z_sample, sd = self.sampler_z((mu, rho))
        # return z_sample, mu, sd



class PCVAE(tf.keras.Model):
    def __init__(self, dim_z, dim_x, learning_rate, kl_weight=1, name="autoencoder", **kwargs):
        super(PCVAE, self).__init__(name=name, **kwargs)
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.kl_weight = kl_weight  # beta

        self.learning_rate = learning_rate
        self.encoder = Encoder_Z(dim_z=self.dim_z,kl_weight=self.kl_weight)
        self.decoder = Decoder_X(dim_z=self.dim_z, dim_x=self.dim_x)

    # def encode_and_decode(self, x_input):
    def call(self, x_input):
        z = self.encoder(x_input)
        x_hat = self.decoder(z)

        return x_hat

    def save_model(self, dir_path, epoch):
        self.encoder.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    def load_model(self, dir_path, epoch):
        self.encoder.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
        self.decoder.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

    
# vae loss function -- only the negative log-likelihood part,
# since we use add_loss for the KL divergence part
def partial_vae_loss(x_true, model):
    # x_recons_logits = model.encode_and_decode(x_true)
    x_recons_logits = model(x_true)
    # compute cross entropy loss for each dimension of every datapoint
    # change this to MSE
    mse = tf.math.squared_difference(tf.keras.activations.sigmoid(x_recons_logits), x_true)
    neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
    return tf.math.reduce_mean(neg_log_likelihood)


def train_step_KL_Reg(x_true, model, optimizer, loss_metric):
    with tf.GradientTape() as tape:
        neg_log_lik = partial_vae_loss(x_true, model)

    gradients = tape.gradient(neg_log_lik, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(neg_log_lik)


def train_step(x_true, model, optimizer, loss_metric, kl_loss_metric):
    with tf.GradientTape() as tape:
        neg_log_lik = partial_vae_loss(x_true, model)
        kl_loss_l = model.encoder.kl_lossTFP.losses 
        kl_loss = kl_loss_l[-1] # make sure its the last entry in the list... 
        total_vae_loss = neg_log_lik + kl_loss
    gradients = tape.gradient(total_vae_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(total_vae_loss)
    kl_loss_metric(kl_loss)



#     def reconstruction_loss(X, X_pred):
#         if loss_type == "bce":
#             bce = tf.losses.BinaryCrossentropy()
#             return bce(X, X_pred) * np.prod(input_shape)
#         elif loss_type == "mse":
#             mse = tf.losses.MeanSquaredError()
#             return mse(X, X_pred) * np.prod(input_shape)
#         else:
#             raise ValueError("Unknown reconstruction loss type. Try 'bce' or 'mse'")

#     def kl_divergence(X, X_pred):
#         self.C += (1/1440) # TODO use correct scalar
#         self.C = min(self.C, 35) # TODO make variable
#         kl = -0.5 * tf.reduce_mean(1 + Z_logvar - Z_mu**2 - tf.math.exp(Z_logvar))
#         return self.gamma * tf.math.abs(kl - self.C)

#     def loss(X, X_pred):
#         return reconstruction_loss(X, X_pred) + kl_divergence(X, X_pred)

#         # create models
#         self.encoder = Model(encoder_input, [Z_mu, Z_logvar, Z])
#         self.decoder = Model(decoder_input, decoder_output)
#         self.vae = Model(encoder_input, self.decoder(Z))
#         self.vae.compile(optimizer='adam', loss=loss, metrics=[reconstruction_loss, kl_divergence])

# latent_dim=128, input_dim=224, learning_rate=4e-4, training=True, beta=1.0, name="beta-autoencoder"):

#         self.beta = beta # default beta = 1.0 is a vanilla VAE
#         self.learning_rate = learning_rate
#         self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
#         self.latent_dim = latent_dim
#         self.pixel_dim = input_dim * input_dim * 3
#         self.beta_norm = self.beta *self.latent_dim/self.pixel_dim # compute the Higgins et al quantity

#         self.elbo_tracker = tf.keras.metrics.Mean(name="elbo")
#         self.kl_mc= tf.keras.metrics.Mean(name="kl")
#         self.nll_tracker = tf.keras.metrics.Mean(name="nll")
#         self.kl_analytic = tf.keras.metrics.Mean(name="kla")

#         # WHAT ARE THESE BEING USED FOR...
#         self.gen_layers = 5  # from 5
#         self.gen_init_size = int(input_dim / (2 ** (self.gen_layers - 1)))
#         self.reshape_channels = 20

#         self.training = training

#     def compile(self, optimizer, **kwargs):
#         super(BCVAE, self).compile(**kwargs)
#         self.gen_model.compile(optimizer=self.optimizer)
#         self.enc_model.compile(optimizer=self.optimizer)
#         #self.loss_fn = loss_fn


#     def encode(self, x, reparam=False):
#         z_mu, z_logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=1)
#         if reparam:
#             z = self.reparameterize(z_mu, z_logvar)
#             return  z, z_mu, z_logvar
#         return z_mu, z_logvar

#     def reparameterize(self, mu, logvar):
#         eps = tf.random.normal(shape=mu.get_shape())
#         #sig = tf.math.exp(0.5*logvar)
#         #log_sig = logvar*0.5  #sqrt
#         # if batchn := mu.shape[0] < 32:
#         #     eps = tf.slice(eps,[0,0],[32-batchn,self.latent_dim])
#         return eps * tf.exp(logvar * 0.5) + mu

#     def decode(self, z, apply_sigmoid=False):
#         logits = self.gen_model(z, training=self.training)
#         if apply_sigmoid:
#             probs = tf.math.softmax(logits)
#         return logits


#     def mylog_normal_pdf(self, sample, mu, logvar, raxis=1): #raxis is the latent_dim
#         log2pi = tf.math.log(2. * np.pi)
#         #sig_sq_inv = 1/sig_sq = 1/(sig*sig) = 1/sig_sq
#         #sig_sq_inv = tf.math.exp(-logvar)
#         # sig*sig = tf.math.exp(logvar)
#         # logvar = log(sig*sig)  = 2* log(sig)
#         # exp(.5*logvar) = sig
#         # 2.0*tf.math.log(sig) = tf.math.log(sig*sig) = tf.math.log(sig_sq) = logvar
#         #    0.5*2.0*tf.math.log(sig) = 0.5*logvar
#         #       sig = tf.math.exp(tf.math.log(sig)) = tf.math.exp(0.5*logvar)
#         # For a single observed value x, the log-likelihood is:
#         #  ll = -tf.math.log(sig) - 0.5*log2pi -0.5*tf.math.square(sample-mu)/sig_sq
#         #  ll = -0.5 * ( 2. * tf.math.log(sig) + log2pi + tf.math.square(sample-mu)*sig_sq_inv )
#         #  ll = -0.5 * ( logvar + log2pi + tf.math.square(sample - mu)*sig_sq_inv)
#         # -------------------------------------
#         # for a sample of observed values X = x_1, .., x_n , ll_i = ll(x_i)
#         # ll = tf.reduce_sum( ll_i ) = -n*tf.math.log(sig) - 0.5*n*log2pi - 0.5*sig_sq_inv*tf.reduce_sum(tf.math.square(sample-mu))
#         # ll = -0.5*n* ( 2*tf.math.log(sig) - log2pi ) - 0.5*sig_sq_inv*tf.reduce_sum(tf.math.square(sample-mu))
#         return tf.reduce_sum( -.5 * ( logvar + log2pi + tf.math.square(sample - mu) * tf.math.exp(-logvar) ),
#                               axis=raxis)


#     def log_normal_pdf(self, sample, mean, logvar, raxis=1):
#         log2pi = tf.math.log(2. * np.pi)
#         return tf.reduce_sum(
#             -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
#             axis=raxis)

#     def compute_loss(self, x):
#         """
#         TODO: change to compute_test_cost
#             this is actually "cost" not loss since its across the batch...
#         """
#         z_sample, z_mu, z_logvar = self.encode(x,reparam=True)

#         # why is the negative in the reduce_mean?
#         # kl_div_a =  - 0.5 * tf.math.reduce_sum(1 + tf.math.log(tf.math.square(sd))
#         #                                         - tf.math.square(mu)
#         #                                         - tf.math.square(sd),   axis=1)
#         kl_div_a = - 0.5 * tf.math.reduce_sum(1 + z_logvar
#                                                 - tf.math.square(z_mu)
#                                                 - tf.math.exp(z_logvar), axis=1)


#         x_recons = self.decode(z_sample,apply_sigmoid=True)
#         #x_logits = self.decode(z_sample)
#         # z_mu, z_logvar = self.encode(x)

#         # z = self.reparameterize(z_mu, z_logvar)
#         # x_recons = self.decode(z,apply_sigmoid=True)

#         # log_likelihood log normal is MSE
#         # loss is [0, 255]
#         # mse = 0.00392156862745098* tf.math.squared_difference(255.*x,255.*x_recons)# 0.00392156862745098 - 1/255.
#         mse = tf.math.squared_difference(x,x_recons)

#         # for images the neg LL is the MSE
#         neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])

#         # # compute reverse KL divergence, either analytically
#         # # MC KL:         # or through MC approximation with one sample
#         # logpz = self.log_normal_pdf(z, 0., 0.) #standard lognormal: mu = 0. logvar=0.
#         # logqz_x = self.log_normal_pdf(z, z_mu, z_logvar)
#         # kl_div_mc = logqz_x - logpz

#         # def normal_log_pdf(sample, mean, logvar, raxis=1):
#         #     log2pi = tf.math.log(2. * np.pi)
#         #     return tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)


#         def analytic_kl(sample, mean, logvar, raxis=1):
#             # log((qz||x)/pz = difference in the log of the gaussian PDF
#             log2pi = tf.math.log(2. * np.pi)
#             logpz = tf.reduce_sum( -.5 * ((sample*sample) + log2pi),axis=raxis)
#             logqz_x = tf.reduce_sum( -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),axis=raxis)
#             return logqz_x - logpz

#         kl_div_mc = analytic_kl(z_sample, z_mu, z_logvar)  # shape=(batch_size,)


#         # # analytic KL for representing SIGMA (log_sig)
#         # # kl_div_a = - 0.5 * tf.math.reduce_sum(
#         # #                                 1 + 0.5*z_logvar - tf.math.square(z_mu) - tf.math.exp(0.5*z_logvar), axis=1)
#         # # KL for representing the VARIANCE
#         # kl_div_a = - 0.5 * tf.math.reduce_sum(
#         #                                 1 + z_logvar - tf.math.square(z_mu) - tf.math.exp(z_logvar), axis=1)

#         elbo = tf.math.reduce_mean(-self.beta * kl_div_a - neg_log_likelihood)  # shape=()
#         kl = tf.math.reduce_mean(kl_div_mc)  # shape=()
#         nll = tf.math.reduce_mean(neg_log_likelihood)  # shape=()
#         kla = tf.math.reduce_mean(kl_div_a)  # shape=()

#         return (-elbo, kl, nll, kla)  #negative ELBO


#     # def call(self, x_input):
#     #     z_mu, z_logvar = self.encode(x)
#     #     z = self.reparameterize(z_mu, z_logvar)
#     #     x_recons_logits = self.decode(z)

#     #     sigsq = tf.math.exp(logvar)
#     #     #mse = tf.reduce_sum(tf.math.square(output-input_img))
#     #     neg_log_likelihood = tf.math.reduce_sum(mse, axis=[1, 2, 3])
#     #     kl_divergence = - 0.5 * tf.math.reduce_sum(1+tf.math.log(sigsq)-tf.math.square(mu)-sigsq, axis=1)

#     #     # CVAE is inherited from tfk.Model, thus have class method add_loss()
#     #     self.add_loss( self.kl_weight * kl_divergence)
#     #     return x_recons_logits

#     @tf.function
#     def train_step(self, x):
#         if isinstance(x, tuple):  # should always be
#             x = x[0]
#         with tf.GradientTape() as tape:
#             #loss = self.compute_loss(x)
#             #cost_mini_batch = self.vae_cost(x)
#             cost_mini_batch, kl, nll, kla = self.compute_loss(x)

#         gradients = tape.gradient(cost_mini_batch, self.trainable_variables)
#         #gradients = tape.gradient(loss, self.trainable_variables)
#         self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
#         # return cost_mini_batch # loss
#         # Compute our own metrics

#         self.elbo_tracker.update_state(cost_mini_batch)
#         self.kl_mc.update_state(kl)
#         self.nll_tracker.update_state(nll)
#         self.kl_analytic.update_state(kla)

#         return {m.name: m.result() for m in self.metrics}


#     #@tf.function
#     def test_step(self, x):
#         temp_training = self.training
#         self.training = False

#         if isinstance(x, tuple):  # should always be
#             x = x[0]

#         cost_mini_batch, kl, nll, kla = self.compute_loss(x)

#         self.elbo_tracker.update_state(cost_mini_batch)
#         self.kl_mc.update_state(kl)
#         self.nll_tracker.update_state(nll)
#         self.kl_analytic.update_state(kla)

#         self.training = temp_training
#         return {m.name: m.result() for m in self.metrics}


#     @property
#     def metrics(self):
#         # We list our `Metric` objects here so that `reset_states()` can be
#         # called automatically at the start of each epoch
#         # or at the start of `evaluate()`.x
#         # If you don't implement this property, you have to call
#         # `reset_states()` yourself at the time of your choosing.
#         return [self.elbo_tracker, self.kl_mc, self.kl_analytic, self.nll_tracker]


#     def reset_metrics(self):
#         # `reset_states()` yourself at the time of your choosing.
#         for m in self.metrics:
#             m.reset_states()


#     def reconstruct(self, train_x, training):
#         temp_training = self.training
#         self.training = training

#         # mean, logvar = self.encode(train_x)
#         # z = self.reparameterize(mean, logvar)
#         # probs = self.decode(z,apply_sigmoid = True)
#         # do the long way to avoid @tf.function
#         mean, logvar = tf.split(self.enc_model(train_x, training=self.training), num_or_size_splits=2, axis=1)
#         eps = tf.random.normal(shape=(self.latent_dim,))
#         z = eps * tf.exp(logvar * 0.5) + mean
#         logits = self.gen_model(z)
#         probs = tf.math.softmax(logits)

#         self.training = temp_training
#         return probs

#     def predict(self, images):
#         """
#         returns logits [-inf inf] "reconstruct returns probability (0,1)
#         """
#         # mu, logvar = self.encode(images)
#         # z = self.reparameterize(mu, logvar)
#         # reconst_images = self.decode(z)
#         # do the long way to avoid @tf.function
#         mean, logvar = tf.split(self.enc_model(x, training=self.training), num_or_size_splits=2, axis=1)
#         eps = tf.random.normal(shape=(self.latent_dim,))
#         z = eps * tf.exp(logvar * 0.5) + mean
#         return self.gen_model(z)


#     def sample(self, eps=None):
#         if eps is None:
#             eps = tf.random.normal(shape=(5, self.latent_dim))  #the 5 is the "monte carlo... but maybe it should be larger"
#         logits = self.gen_model(eps, training=False)
#         return tf.math.softmax(logits)


# #######################
# ## UTILITIES...
# #######################

#     def print_model_summary(self):
#         print("Inference Net Summary:\n")
#         self.enc_model.summary()
#         print("\nGenerative Net Summary:\n")
#         self.gen_model.summary()

#     def print_model_IO(self):
#         print("\nInference Net Summary (input then output):")
#         print(self.enc_model.input_shape)
#         print(self.enc_model.output_shape)
#         print("\nGenerative Net Summary:")
#         print(self.gen_model.input_shape)
#         print(self.gen_model.output_shape)


#     def plot_model(self,encoder=True):
#         """
#         by default print teh encoder. print generator if False
#         """
#         if encoder:
#             print("\n Summary (encoder):")
#             return tf.keras.utils.plot_model(enc_model, show_shapes=True, show_layer_names=True)
#         else:
#             print("\n Summary (generator):")
#             return tf.keras.utils.plot_model(gen_model, show_shapes=True, show_layer_names=True)


#     def save_model(self, dir_path, epoch):
#         self.enc_model.save_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
#         self.gen_model.save_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))

#     def load_model(self, dir_path, epoch):
#         self.enc_model.load_weights(os.path.join(dir_path, "enc_epoch_{}.h5".format(epoch)))
#         self.gen_model.load_weights(os.path.join(dir_path, "dec_epoch_{}.h5".format(epoch)))


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
