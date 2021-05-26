from itertools import product
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

# set gpu memory limit, gradually reduce to test demands
import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print(gpus)
if gpus:
  # Restrict TensorFlow to only allocate some of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*24)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)

tf.debugging.set_log_device_placement(True)

try:
  with tf.device('/device:GPU:0'):
    a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    c = tf.matmul(a, b)
except RuntimeError as e:
  print(e)


assert False

# dateset
dset = get_dsprites_tf_dataset()

# use params with max latent size (biggest memory demand)
latent_dim = 200
norm_beta = 0.002

# create bvae
bvae = DspritesBetaVAE(latent_dim=latent_dim, normalized_beta=norm_beta, random_seed=0)

# train and save
# save_dir: where to save all results (use None for automatic dir)
# batch_limit_for_debug: how many batches to use per epoch (for quick debug)
#                        set batch_limit_for_debug=None to use all batches
bvae.train_save(dset, epochs=20, batch_size=256, lr=.01, save_dir='output_train/orig_architecture/20ep/',
                verbose_batch=50, verbose_epoch=1, batch_limit_for_debug=1)
