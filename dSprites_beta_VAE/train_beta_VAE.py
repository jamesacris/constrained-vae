from itertools import product
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

# set gpu memory limit, gradually reduce to test demands
import tensorflow as tf

memory = 4096 + 512

physical_gpus = tf.config.list_physical_devices('GPU')
print(physical_gpus)
# Restrict TensorFlow to only allocate some of memory on the first GPU
tf.config.experimental.set_virtual_device_configuration(
    physical_gpus[0],
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=memory)])
tf.config.set_visible_devices(physical_gpus[0], 'GPU')
logical_gpus = tf.config.experimental.list_logical_devices('GPU')
print(logical_gpus)


print(len(physical_gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
print(logical_gpus)

tf.config.set_soft_device_placement(True)
tf.debugging.set_log_device_placement(True)

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
bvae.train_save(dset, epochs=20, batch_size=256, lr=.01, save_dir=f'output_train/testing_memory/{memory}_MB/',
                verbose_batch=1, verbose_epoch=1, batch_limit_for_debug=1)
