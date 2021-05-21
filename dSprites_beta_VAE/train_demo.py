from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

# dateset
dset = get_dsprites_tf_dataset()

# create bve
bvae = DspritesBetaVAE(latent_dim=10, normalized_beta=4., random_seed=0)

# train and save
# save_dir: where to save all results (use None for automatic dir)
# batch_limit_for_debug: how many batches to use per epoch (for quick debug)
#                        set batch_limit_for_debug=None to use all batches
bvae.train_save(dset, epochs=2, batch_size=128, lr=.01, save_dir=None,
                verbose_batch=1, verbose_epoch=1, batch_limit_for_debug=5)
