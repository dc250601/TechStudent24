import qkeras
from qkeras import QBatchNormalization
from qkeras.qlayers import QDense, QActivation
from qkeras.quantizers import quantized_bits, quantized_relu
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model

class ModelBackbone(Model):
    def __init__(self, nodes, ap_fixed_kernel, ap_fixed_bias, ap_fixed_activation):
        super().__init__()
        
        self.model = tf.keras.Sequential()
        
        for i,n in enumerate(nodes):
            activation = quantized_relu(*ap_fixed_activation) if i != len(nodes)-1 else None
            self.model.add(
                QDense(
                    n,
                    kernel_quantizer=quantized_bits(*ap_fixed_kernel, alpha=1),
                    bias_quantizer=quantized_bits(*ap_fixed_bias, alpha=1),
                    kernel_initializer="glorot_uniform",
                    bias_initializer="zeros",
                    activation=activation
                )
            )

    def call(self, inputs, training=False):
        return self.model(inputs, training=training)



class ModelProjector(Model):
    
    def __init__(self, projection_size=128, num_layers=3):
        super().__init__()

        self.blocks = []
        for i in range(num_layers - 1):
            self.blocks.append(
                tf.keras.Sequential([
                    tf.keras.layers.Dense(projection_size),
                    tf.keras.layers.BatchNormalization(),
                    tf.keras.layers.ReLU()
                ])
            )
        self.blocks.append(
            tf.keras.Sequential([
                tf.keras.layers.Dense(projection_size),
                tf.keras.layers.BatchNormalization()
            ])
        )
    
    def call(self, x, training=False):
        for block in self.blocks:
            x = block(x, training=training)
        return x

class VICReg(Model):
    def __init__(
        self,
        backbone,
        projector,
        num_features,  # Size of the projection layer
        batch_size,
        sim_coeff=50,
        std_coeff=50,
        cov_coeff=1,
    ):
        super().__init__()
        self.num_features = num_features
        self.backbone = backbone
        self.projector = projector

        self.batch_size = batch_size

        
        self.sim_coeff = sim_coeff
        self.std_coeff = std_coeff
        self.cov_coeff = cov_coeff

        self.loss_tracker = tf.keras.metrics.Mean(name="Total_Loss")
        self.loss_tracker_repr = tf.keras.metrics.Mean(name="Total_Loss_repr")
        self.loss_tracker_std = tf.keras.metrics.Mean(name="Total_Loss_std")
        self.loss_tracker_cov = tf.keras.metrics.Mean(name="Total_Loss_cov")
        


    @tf.function
    def train_step(self,data):
        x,y = data
        with tf.GradientTape() as tape:
        
            x = self.projector(self.backbone(x, training=True), training=True)
            y = self.projector(self.backbone(y, training=True), training=True)

            repr_loss = tf.keras.losses.mean_squared_error(x,y)
        
            x = x - tf.reduce_mean(x, axis=0, keepdims=True)
            y = y - tf.reduce_mean(y, axis=0, keepdims=True)
    
            std_x = tf.sqrt(tf.math.reduce_variance(x, axis=0) + 0.0001)
            std_y = tf.sqrt(tf.math.reduce_variance(y, axis=0) + 0.0001)
            
            std_loss = tf.reduce_mean(tf.nn.relu(1.0 - std_x)) / 2 + tf.reduce_mean(tf.nn.relu(1.0 - std_y)) / 2
    
            cov_x = tf.linalg.matmul(x, x, transpose_a=True) / (self.batch_size - 1.0)
            cov_y = tf.linalg.matmul(y, y, transpose_a=True) / (self.batch_size - 1.0)
            
            cov_loss = (tf.reduce_sum(tf.square(off_diagonal(cov_x))) +  tf.reduce_sum(tf.square(off_diagonal(cov_y)))) / float(self.num_features)
    
            loss = self.sim_coeff * repr_loss + self.std_coeff * std_loss + self.cov_coeff * cov_loss

        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.loss_tracker.update_state(loss)
        self.loss_tracker_repr.update_state(repr_loss)
        self.loss_tracker_cov.update_state(cov_loss)
        self.loss_tracker_std.update_state(std_loss)

        return {"Loss":self.loss_tracker.result(),
                "Representation Loss":self.loss_tracker_repr.result(),
                "Covariance Loss":self.loss_tracker_cov.result(),
                "Standard Deviation Loss":self.loss_tracker_std.result()
               }

def exclude_bias_and_norm(p):
    return p.ndim == 1


def off_diagonal(x):
    n = tf.shape(x)[0]
    mask = ~tf.cast(tf.eye(n), tf.bool)
    return tf.boolean_mask(x, mask)
