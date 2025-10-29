import qkeras
from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model

class VAE_Encoder(Model):
    def __init__(self, nodes, feature_size, ap_fixed_kernel, ap_fixed_bias, ap_fixed_activation):
        super(VAE_Encoder, self).__init__()
        self.model = Sequential()
        
        for n in nodes:
            self.model.add(QDense(
                n,
                kernel_quantizer=quantized_bits(*ap_fixed_kernel, alpha=1),
                bias_quantizer=quantized_bits(*ap_fixed_bias, alpha=1),
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                activation=quantized_relu(*ap_fixed_activation)
            ))

        
        self.layer_mu = QDense(
            feature_size,
            kernel_quantizer=quantized_bits(*ap_fixed_kernel, alpha=1),
            bias_quantizer=quantized_bits(*ap_fixed_bias, alpha=1),
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros"
        )
        
        self.layer_log_var = QDense(
            feature_size,
            kernel_quantizer=quantized_bits(*ap_fixed_kernel, alpha=1),
            bias_quantizer=quantized_bits(*ap_fixed_bias, alpha=1),
            kernel_initializer="glorot_uniform",
            bias_initializer="zeros"
        )

    def call(self, x):
        z = self.model(x)
        return self.layer_mu(z), self.layer_log_var(z)

class VAE_Decoder(Model):
    def __init__(self, nodes, ap_fixed_kernel, ap_fixed_bias, ap_fixed_activation):
        super(VAE_Decoder, self).__init__()
        self.model = Sequential()
        
        for i, n in enumerate(nodes):
            activation = quantized_relu(*ap_fixed_activation) if i != len(nodes)-1 else None
            self.model.add(QDense(
                n,
                kernel_quantizer=quantized_bits(*ap_fixed_kernel, alpha=1),
                bias_quantizer=quantized_bits(*ap_fixed_bias, alpha=1),
                kernel_initializer="glorot_uniform",
                bias_initializer="zeros",
                activation=activation
            ))

    def call(self, x):
        return self.model(x)

class VariationalAutoEncoder(Model):
    def __init__(self, encoder, decoder, kl_scale, reco_scale, **kwargs):
        super(VariationalAutoEncoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.kl_scale = kl_scale
        self.reco_scale = reco_scale

        self.total_loss_tracker = tf.keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = tf.keras.metrics.Mean(name="reco_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")


    def reparameterization(self, mean, log_var):
        epsilon = tf.random.normal(tf.shape(log_var))
        return mean + tf.exp(0.5 * log_var) * epsilon

    def call(self, inputs):
        mean, log_var = self.encoder(inputs)
        z = self.reparameterization(mean, log_var)
        x_hat = self.decoder(z)
        return x_hat, mean, log_var
    
    @tf.function
    def train_step(self, data):
        with tf.GradientTape() as tape:
            x_hat, mean, log_var = self(data)
            
            reconstruction_loss = tf.reduce_mean(
                tf.keras.losses.mse(data, x_hat)
            )
            
            kl_loss = -0.5 * tf.reduce_mean(
                1 + log_var - tf.square(mean) - tf.exp(log_var)
            )

            total_loss = self.reco_scale*reconstruction_loss + self.kl_scale*kl_loss

        gradients = tape.gradient(total_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)

        
        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kl_loss": kl_loss
        }
