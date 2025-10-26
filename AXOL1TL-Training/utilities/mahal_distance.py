import numpy as np
import tensorflow as tf
from qkeras import quantized_bits

class DistanceLayer(tf.keras.layers.Layer):

    def __init__(
        self, 
        model, 
        bits=32, 
        integer=8, 
        batch_size=8192):
        
        super().__init__()

        self.n_components = model.n_components
        self.batch_size = batch_size
        self.quantizer = quantized_bits(bits=bits, integer=integer)

        covs_inv_quantized = []
        for c in range(self.n_components):
            cov = model.covariances_[c]
            cov_inv = np.linalg.inv(cov)
            cov_inv_q = np.vectorize(lambda x: float(self.quantizer(x).numpy()))(cov_inv)
            covs_inv_quantized.append(cov_inv_q)

        covs_inv_quantized = np.stack(covs_inv_quantized, axis=0)
        self.covariances_inv = tf.constant(covs_inv_quantized, dtype=tf.float32)

        means_quantized = []
        for c in range(self.n_components):
            mean_vec = model.means_[c]
            mean_vec_q = np.vectorize(
                lambda x: float(self.quantizer(x).numpy())
            )(mean_vec)
            means_quantized.append(mean_vec_q)

        means_quantized = np.stack(means_quantized, axis=0)
        self.means_q = tf.constant(means_quantized, dtype=tf.float32)

        weights_ = model.weights_
        weights_q = np.vectorize(lambda x: float(self.quantizer(x).numpy()))(weights_)
        self.weights_q = tf.constant(weights_q, dtype=tf.float32)

    def call(self, inputs):
        
        N = tf.shape(inputs)[0]
        C = self.n_components

        distance_list = []

        for c in range(C):
            
            inv_c = self.covariances_inv[c]
            mean_c = self.means_q[c]

            dist_c_all = []
            
            start = tf.constant(0)
            while True:
                end = start + self.batch_size
                x_batch = inputs[start: end]
                
                if tf.shape(x_batch)[0] == 0:
                    break

                centered = x_batch - mean_c

                step1 = tf.matmul(centered, inv_c)

                bdist = tf.reduce_sum(step1 * centered, axis=1)

                dist_c_all.append(bdist)

                start = end
                if end >= N:
                    break

            dist_c_full = tf.concat(dist_c_all, axis=0)
            distance_list.append(dist_c_full)

        distances_sqr = tf.stack(distance_list, axis=0)
        distances_sqr = tf.transpose(distances_sqr, [1, 0])

        weighted_d = distances_sqr ** self.weights_q

        score = tf.reduce_prod(weighted_d, axis=1)

        return score
