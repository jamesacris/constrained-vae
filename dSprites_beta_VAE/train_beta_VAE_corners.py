from itertools import product
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

# dateset
dset = get_dsprites_tf_dataset()

# 'purple' corner
latent_dim = 200
norm_beta = 0.002

# create bvae
bvae = DspritesBetaVAE(latent_dim=latent_dim, normalized_beta=norm_beta, random_seed=0)

# train and save
# save_dir: where to save all results (use None for automatic dir)
# batch_limit_for_debug: how many batches to use per epoch (for quick debug)
#                        set batch_limit_for_debug=None to use all batches
bvae.train_save(dset, epochs=20, batch_size=256, lr=.01, save_dir='output_train/orig_architecture/20ep/',
                verbose_batch=50, verbose_epoch=1, batch_limit_for_debug=None)
