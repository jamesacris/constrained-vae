import numpy as np
import tensorflow as tf

def log_density(mean, log_var, sample):
    inv_sigma = tf.exp(-log_var * .5)
    tmp = (sample - mean) * inv_sigma
    return -0.5 * (tmp * tmp + log_var + tf.math.log(2 * np.pi))
    
def logsumexp(value, dim):
    m = tf.reduce_max(value, axis=dim, keepdims=True)
    value0 = value - m
    m = tf.squeeze(m, dim)
    return m + tf.math.log(tf.reduce_sum(tf.exp(value0), axis=dim))

def estimate_entropies(qz_mean, qz_log_var, qz_sample, n_samples=10000):
    N, K = qz_sample.shape[0], qz_sample.shape[1]
    
    # take subset of samples
    qz_sample = qz_sample[np.random.permutation(N)[:n_samples]]
    S = qz_sample.shape[0]
    
    # weights
    weights = -np.log(N)
    
    # entropy
    entropies = tf.zeros(K)
    
    # expand dimension and send to GPU
    # qz_mean: (N, K) -> (N, K, S)
    qz_mean = tf.convert_to_tensor(np.expand_dims(qz_mean, 2))  # (N, K, 1)
    qz_log_var = tf.convert_to_tensor(np.expand_dims(qz_log_var, 2))  # (N, K, 1)
    # qz_sample: (S, K) -> (N, K, S)
    qz_sample = tf.convert_to_tensor(np.expand_dims(qz_sample.T, 0))  # (1, K, S)

    batch_size = 10
    k = 0
    while k < S:
        batch_size = min(batch_size, S - k)
        logqz_i = log_density(
            tf.tile(qz_mean, (1, 1, batch_size)),
            tf.tile(qz_log_var, (1, 1, batch_size)),
            tf.tile(qz_sample[:, :, k:k + batch_size], (N, 1, 1)))
        k += batch_size
        # computes - log q(z_i) summed over minibatch
        entropies += - tf.reduce_sum(logsumexp(logqz_i + weights, dim=0), axis=1)

    entropies /= S
    return entropies.numpy()
