from itertools import product
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

# dateset
dset = get_dsprites_tf_dataset()

# train the models corresponding to the four corners
latent_dims = [10, 200]
norm_betas = [0.002, 10]
for latent_dim, norm_beta in product(latent_dims, norm_betas):
    # create bve
    bvae = DspritesBetaVAE(latent_dim=latent_dim, normalized_beta=norm_beta, random_seed=0)

    # train and save
    # save_dir: where to save all results (use None for automatic dir)
    # batch_limit_for_debug: how many batches to use per epoch (for quick debug)
    #                        set batch_limit_for_debug=None to use all batches
    bvae.train_save(dset, epochs=10, batch_size=256, lr=.01, save_dir=None,
                    verbose_batch=50, verbose_epoch=1, batch_limit_for_debug=None)
