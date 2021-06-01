from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import OrderedDsprites

# load weights
bvae = DspritesBetaVAE(latent_dim=10, normalized_beta=4.)
bvae.load_model_weights('output_train/nlat=10__beta=4.0__seed=0/weights_encoder.h5',
                         'output_train/nlat=10__beta=4.0__seed=0/weights_decoder.h5')

# compute score
ods = OrderedDsprites()
score = ods.compute_disentangle_metric_score(
    bvae, n_zdiff_per_y_train=50, n_zdiff_per_y_test=10, n_img_per_zdiff=4, random_seed=0)
print(score)
