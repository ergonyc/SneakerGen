


##############
#  code snips from "6 differenty ways of implimenting vae with tensorflow 2"
# https://towardsdatascience.com/6-different-ways-of-implementing-vae-with-tensorflow-2-and-tensorflow-probability-9fe34a8ab981
#  
##################

##################
# generic architecture from https://www.tensorflow.org/tutorials/generative/cvae
##################

import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp
import time

class CVAE(tf.keras.Model):
  """Convolutional variational autoencoder."""

  def __init__(self, latent_dim):
    super(CVAE, self).__init__()
    self.latent_dim = latent_dim
    self.encoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=32, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=64, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            # No activation
            tf.keras.layers.Dense(latent_dim + latent_dim),
        ]
    )

    self.decoder = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
            tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
            tf.keras.layers.Conv2DTranspose(
                filters=64, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=32, kernel_size=3, strides=2, padding='same',
                activation='relu'),
            # No activation
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=1, padding='same'),
        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.decoder(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs
    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)


def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)


def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=x)
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)


@tf.function
def train_step(model, x, optimizer):
  """Executes one training step and returns the loss.

  This function computes the loss and gradients, and uses the latter to
  update the model's parameters.
  """
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))



##################
# version a
##################
# vae cost function as negative ELBO
def vae_cost(x_true, model, analytic_kl=True, kl_weight=1):
    z_sample, mu, sd = model.encode(x_true)
    x_recons_logits = model.decoder(z_sample)
    # compute cross entropy loss for each dimension of every datapoint
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=x_true,
                        logits=x_recons_logits)  # shape=(batch_size, 28, 28, 1)
    # compute cross entropy loss for all instances in mini-batch; shape=(batch_size,)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    # compute reverse KL divergence, either analytically 
    # or through MC approximation with one sample
    if analytic_kl:
        kl_divergence = - 0.5 * tf.math.reduce_sum(
            1 + tf.math.log(tf.math.square(sd)) - tf.math.square(mu) - tf.math.square(sd),
            axis=1)  # shape=(batch_size, )
    else:
        logpz = normal_log_pdf(z_sample, 0., 1.)  # shape=(batch_size,)
        logqz_x = normal_log_pdf(z_sample, mu, tf.math.square(sd))  # shape=(batch_size,)
        kl_divergence = logqz_x - logpz
    elbo = tf.math.reduce_mean(-kl_weight * kl_divergence - neg_log_likelihood)  # shape=()
    return -elbo


@tf.function
def train_step(x_true, model, optimizer, analytic_kl=True, kl_weight=1):
    with tf.GradientTape() as tape:
        cost_mini_batch = vae_cost(x_true, model, analytic_kl, kl_weight)
    gradients = tape.gradient(cost_mini_batch, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))



## KL divergence snippet
prior_dist = tfd.MultivariateNormalDiag(loc=tf.zeros((batch_size, latent_dim)), scale_diag=tf.ones((batch_size, latent_dim)))
var_post_dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=sd)
kl_divergence = tfd.kl_divergence(distribution_a=var_post_dist, distribution_b=prior_dist)



##################
# version b
##################
class VAE_MNIST:
    
    def __init__(self, dim_z, kl_weight, learning_rate):
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.kl_weight = kl_weight
        self.learning_rate = learning_rate

    # Sequential API encoder
    def encoder_z(self):
        # define prior distribution for the code, which is an isotropic Gaussian
        prior = tfd.Independent(tfd.Normal(loc=tf.zeros(self.dim_z), scale=1.), 
                                reinterpreted_batch_ndims=1)
        # build layers argument for tfk.Sequential()
        input_shape = self.dim_x
        layers = [tfkl.InputLayer(input_shape=input_shape)]
        layers.append(tfkl.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Conv2D(filters=64, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Flatten())
        # the following two lines set the output to be a probabilistic distribution
        layers.append(tfkl.Dense(tfpl.IndependentNormal.params_size(self.dim_z), 
                                 activation=None, name='z_params'))
        layers.append(tfpl.IndependentNormal(self.dim_z, 
            convert_to_tensor_fn=tfd.Distribution.sample, 
            activity_regularizer=tfpl.KLDivergenceRegularizer(prior, weight=self.kl_weight), 
            name='z_layer'))
        return tfk.Sequential(layers, name='encoder')
    
    # Sequential API decoder
    def decoder_x(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_z)]
        layers.append(tfkl.Dense(7*7*32, activation=None))
        layers.append(tfkl.Reshape((7,7,32)))
        layers.append(tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
                                           padding='same'))
        layers.append(tfkl.Flatten(name='x_params'))
        # note that here we don't need 
        # `tfkl.Dense(tfpl.IndependentBernoulli.params_size(self.dim_x))` because 
        # we've restored the desired input shape with the last Conv2DTranspose layer
        layers.append(tfpl.IndependentBernoulli(self.dim_x, name='x_layer'))
        return tfk.Sequential(layers, name='decoder')
    
    def build_vae_keras_model(self):
        x_input = tfk.Input(shape=self.dim_x)
        encoder = self.encoder_z()
        decoder = self.decoder_x()
        z = encoder(x_input)

        # compile VAE model
        model = tfk.Model(inputs=x_input, outputs=decoder(z))
        model.compile(loss=negative_log_likelihood, 
                      optimizer=tfk.optimizers.Adam(self.learning_rate))
        return model

# the negative of log-likelihood for probabilistic output
negative_log_likelihood = lambda x, rv_x: -rv_x.log_prob(x)





##################
# version c (a with custom loss)
##################
def custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits):
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                                      labels=x_true, 
                                                      logits=x_recons_logits)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    return tf.math.reduce_mean(neg_log_likelihood)


model.compile(loss=custom_sigmoid_cross_entropy_loss_with_logits, optimizer=tfk.optimizers.Adam(learning_rate))


##################
# version d (b with custom loss)
##################

def custom_binary_cross_entropy_loss(x_true, x_recons_dist):
    x_recons_mean = x_recons_dist.mean()
    raw_cross_entropy = tfk.losses.BinaryCrossentropy(
                reduction=tfk.losses.Reduction.NONE)(x_true, x_recons_mean)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2])
    return tf.math.reduce_mean(neg_log_likelihood)



##################
# version e (a with custom loss)
##################

class VAE_MNIST(tfk.Model):
    
    def __init__(self, dim_z, kl_weight=1, name="autoencoder", **kwargs):
        super(VAE_MNIST, self).__init__(name=name, **kwargs)
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.encoder = self.encoder_z()
        self.decoder = self.decoder_x()
        self.kl_weight = kl_weight
        
    # Sequential API encoder
    def encoder_z(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_x)]
        layers.append(tfkl.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Conv2D(filters=64, kernel_size=3, strides=(2,2), 
                                  padding='valid', activation='relu'))
        layers.append(tfkl.Flatten())
        # *2 because number of parameters for both mean and (raw) standard deviation
        layers.append(tfkl.Dense(self.dim_z*2, activation=None))
        return tfk.Sequential(layers)
    
    def encode(self, x_input):
        mu, rho = tf.split(self.encoder(x_input), num_or_size_splits=2, axis=1)
        sd = tf.math.log(1+tf.math.exp(rho))
        z_sample = mu + sd * tf.random.normal(shape=(self.dim_z,))
        return z_sample, mu, sd
    
    # Sequential API decoder
    def decoder_x(self):
        layers = [tfkl.InputLayer(input_shape=self.dim_z)]
        layers.append(tfkl.Dense(7*7*32, activation=None))
        layers.append(tfkl.Reshape((7,7,32)))
        layers.append(tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                                           padding='same', activation='relu'))
        layers.append(tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
                                           padding='same'))
        return tfk.Sequential(layers, name='decoder')
    
    def call(self, x_input):
        z_sample, mu, sd = self.encode(x_input)
        kl_divergence = tf.math.reduce_mean(- 0.5 * 
                tf.math.reduce_sum(1+tf.math.log(
                tf.math.square(sd))-tf.math.square(mu)-tf.math.square(sd), axis=1))
        x_logits = self.decoder(z_sample)
        # VAE_MNIST is inherited from tfk.Model, thus have class method add_loss()
        self.add_loss(self.kl_weight * kl_divergence)
        return x_logits
    
# custom loss function with tf.nn.sigmoid_cross_entropy_with_logits
def custom_sigmoid_cross_entropy_loss_with_logits(x_true, x_recons_logits):
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
                                            labels=x_true, logits=x_recons_logits)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    return tf.math.reduce_mean(neg_log_likelihood)

  
####################   The following code shows how to train the model   ####################
# set hyperparameters
epochs = 10
batch_size = 32
lr = 0.0001
latent_dim=16
kl_w=3
# compile and train tfk.Model
vae = VAE_MNIST(dim_z=latent_dim, kl_weight=kl_w)
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), 
            loss=custom_sigmoid_cross_entropy_loss_with_logits)
            
train_history = vae.fit(x=train_images, y=train_images, batch_size=batch_size, epochs=epochs, 
                        verbose=1, validation_data=(test_images, test_images), shuffle=True)




##################
# version f (No Keras)
##################
class Sampler_Z(tfk.layers.Layer):
    
    def call(self, inputs):
        mu, rho = inputs
        sd = tf.math.log(1+tf.math.exp(rho))
        batch_size = tf.shape(mu)[0]
        dim_z = tf.shape(mu)[1]
        z_sample = mu + sd * tf.random.normal(shape=(batch_size, dim_z))
        return z_sample, sd

class Encoder_Z(tfk.layers.Layer):
    
    def __init__(self, dim_z, name="encoder", **kwargs):
        super(Encoder_Z, self).__init__(name=name, **kwargs)
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.conv_layer_1 = tfkl.Conv2D(filters=32, kernel_size=3, strides=(2,2), 
                                        padding='valid', activation='relu')
        self.conv_layer_2 = tfkl.Conv2D(filters=64, kernel_size=3, strides=(2,2), 
                                        padding='valid', activation='relu')
        self.flatten_layer = tfkl.Flatten()
        self.dense_mean = tfkl.Dense(self.dim_z, activation=None, name='z_mean')
        self.dense_raw_stddev = tfkl.Dense(self.dim_z, activation=None, name='z_raw_stddev')
        self.sampler_z = Sampler_Z()
    
    # Functional
    def call(self, x_input):
        z = self.conv_layer_1(x_input)
        z = self.conv_layer_2(z)
        z = self.flatten_layer(z)
        mu = self.dense_mean(z)
        rho = self.dense_raw_stddev(z)
        z_sample, sd = self.sampler_z((mu,rho))
        return z_sample, mu, sd
      
class Decoder_X(tfk.layers.Layer):
    
    def __init__(self, dim_z, name="decoder", **kwargs):
        super(Decoder_X, self).__init__(name=name, **kwargs)
        self.dim_z = dim_z
        self.dense_z_input = tfkl.Dense(7*7*32, activation=None)
        self.reshape_layer = tfkl.Reshape((7,7,32))
        self.conv_transpose_layer_1 = tfkl.Conv2DTranspose(filters=64, kernel_size=3, strides=2, 
                                                           padding='same', activation='relu')
        self.conv_transpose_layer_2 = tfkl.Conv2DTranspose(filters=32, kernel_size=3, strides=2, 
                                                           padding='same', activation='relu')
        self.conv_transpose_layer_3 = tfkl.Conv2DTranspose(filters=1, kernel_size=3, strides=1, 
                                                           padding='same')
    
    # Functional
    def call(self, z):
        x_output = self.dense_z_input(z)
        x_output = self.reshape_layer(x_output)
        x_output = self.conv_transpose_layer_1(x_output)
        x_output = self.conv_transpose_layer_2(x_output)
        x_output = self.conv_transpose_layer_3(x_output)
        return x_output
      
class VAE_MNIST(tfk.Model):
    
    def __init__(self, dim_z, learning_rate, kl_weight=1, name="autoencoder", **kwargs):
        super(VAE_MNIST, self).__init__(name=name, **kwargs)
        self.dim_x = (28, 28, 1)
        self.dim_z = dim_z
        self.learning_rate = learning_rate
        self.encoder = Encoder_Z(dim_z=self.dim_z)
        self.decoder = Decoder_X(dim_z=self.dim_z)
        self.kl_weight = kl_weight
        
    # def encode_and_decode(self, x_input):
    def call(self, x_input):
        z_sample, mu, sd = self.encoder(x_input)
        x_recons_logits = self.decoder(z_sample)
        
        kl_divergence = - 0.5 * tf.math.reduce_sum(1+tf.math.log(
          tf.math.square(sd))-tf.math.square(mu)-tf.math.square(sd), axis=1)
        kl_divergence = tf.math.reduce_mean(kl_divergence)
        # self.add_loss(lambda: self.kl_weight * kl_divergence)
        self.add_loss(self.kl_weight * kl_divergence)
        return x_recons_logits
    
    
# vae loss function -- only the negative log-likelihood part, 
# since we use add_loss for the KL divergence part
def partial_vae_loss(x_true, model):
    # x_recons_logits = model.encode_and_decode(x_true)
    x_recons_logits = model(x_true)
    # compute cross entropy loss for each dimension of every datapoint
    raw_cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
      labels=x_true, logits=x_recons_logits)
    neg_log_likelihood = tf.math.reduce_sum(raw_cross_entropy, axis=[1, 2, 3])
    return tf.math.reduce_mean(neg_log_likelihood)

@tf.function
def train_step(x_true, model, optimizer, loss_metric):
    with tf.GradientTape() as tape:
        neg_log_lik = partial_vae_loss(x_true, model)
        # kl_loss = model.losses[-1]
        kl_loss = tf.math.reduce_sum(model.losses)  # vae.losses is a list
        total_vae_loss = neg_log_lik + kl_loss
    gradients = tape.gradient(total_vae_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    loss_metric(total_vae_loss)
    
####################   The following code shows how to train the model   ####################
# set hyperparameters
train_size = 60000
batch_size = 64
test_size = 10000
latent_dim=16
lr = 0.0005
kl_w = 3
epochs = 10
num_examples_to_generate = 16
train_dataset = (tf.data.Dataset.from_tensor_slices(train_images)
                 .shuffle(train_size).batch(batch_size))
test_dataset = (tf.data.Dataset.from_tensor_slices(test_images)
                .shuffle(test_size).batch(batch_size))
# model training
vae = VAE_MNIST(dim_z=latent_dim, learning_rate=lr, analytic_kl=True, kl_weight=kl_w)
loss_metric = tf.keras.metrics.Mean()
opt = tfk.optimizers.Adam(vae.learning_rate)

for epoch in range(epochs):
    start_time = time.time()
    for train_x in tqdm(train_dataset):
        train_step(train_x, vae, opt, loss_metric)
    end_time = time.time()
    elbo = -loss_metric.result()
    #display.clear_output(wait=False)
    print('Epoch: {}, Train set ELBO: {}, time elapse for current epoch: {}'.format(
            epoch, elbo, end_time - start_time))
    generate_images(vae, test_sample)