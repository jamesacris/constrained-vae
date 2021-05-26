from argparse import ArgumentParser
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

parser = ArgumentParser()
parser.add_argument("-z", "--nlat", default=10, type=int, help="dimension of latent space")
parser.add_argument("--norm_beta", default=0.002, type=float, help="normalised beta")
parser.add_argument("--seed", default=0, type=int, help="random seed to initialise model weights")
parser.add_argument("--epochs", default=1, type=int, help="number of training epochs")
parser.add_argument("-b", "--batch-size", default=256, type=int, help="size of mini-batch")
parser.add_argument("-l", "--learning-rate", default=.01, type=float, help="learning rate")
parser.add_argument("--savedir", default=None, help="where to save all results (use None for automatic dir)")
parser.add_argument("--verbose-batch", default=1, type=int, help="frequency with which to log batch metrics")
parser.add_argument("--verbose-epoch", default=1, type=int, help="frequency with which to log epoch metrics")
parser.add_argument("--batch-lim-debug", default=None, help="how many batches to use per epoch (for quick debug)")
args = parser.parse_args()

# dataset
dset = get_dsprites_tf_dataset()

# create bvae
bvae = DspritesBetaVAE(latent_dim=args.nlat, normalized_beta=args.norm_beta, random_seed=args.seed)


# train and save
# save_dir: where to save all results (use None for automatic dir)
# batch_limit_for_debug: how many batches to use per epoch (for quick debug)
#                        set batch_limit_for_debug=None to use all batches
bvae.train_save(dset, epochs=args.epochs, batch_size=args.batch_size, lr=args.learning_rate, 
                save_dir=args.savedir, verbose_batch=args.verbose_batch, 
                verbose_epoch=args.verbose_epoch, batch_limit_for_debug=args.batch_lim_debug)