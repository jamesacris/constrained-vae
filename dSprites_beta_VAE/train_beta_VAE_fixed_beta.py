import tensorflow as tf
from argparse import ArgumentParser
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

import multiprocessing 

def train_func(vGPU, nlat, norm_beta, seed, epochs, batch_size, learning_rate,
               savedir, verbose_batch, verbose_epoch, batch_lim_debug):
    
    with tf.device(vGPU):
        # create bvae
        bvae = DspritesBetaVAE(latent_dim=nlat, normalized_beta=norm_beta, random_seed=seed)


        # train and save
        # save_dir: where to save all results (use None for automatic dir)
        # batch_limit_for_debug: how many batches to use per epoch (for quick debug)
        #                        set batch_limit_for_debug=None to use all batches
        bvae.train_save(dset, epochs=epochs, batch_size=batch_size, lr=learning_rate, 
                        save_dir=savedir, verbose_batch=verbose_batch, 
                        verbose_epoch=verbose_epoch, batch_limit_for_debug=batch_lim_debug)


if "name" == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--norm_beta", default=0.002, type=float, help="normalised beta")
    parser.add_argument("--seed", default=0, type=int, help="random seed to initialise model weights")
    parser.add_argument("--epochs", default=1, type=int, help="number of training epochs")
    parser.add_argument("-b", "--batch-size", default=256, type=int, help="size of mini-batch")
    parser.add_argument("-l", "--learning-rate", default=.01, type=float, help="learning rate")
    parser.add_argument("--savedir", default=None, type=str, help="where to save all results (use None for automatic dir)")
    parser.add_argument("--verbose-batch", default=1, type=int, help="frequency with which to log batch metrics")
    parser.add_argument("--verbose-epoch", default=1, type=int, help="frequency with which to log epoch metrics")
    parser.add_argument("--batch-lim-debug", default=None, type=int, help="how many batches to use per epoch (for quick debug)")
    args = parser.parse_args()

    # dataset
    dset = get_dsprites_tf_dataset()

    # create virtual GPU
    # 5 GB x 20 = 100 GB
    # 4 * 32 = 128 GB

    pGPUs = tf.config.list_physical_devices('GPU')
    vGPUs = []
    for pGPU in pGPUs:
        tf.config.experimental.set_virtual_device_configuration(
            pGPU,
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120),
             tf.config.experimental.VirtualDeviceConfiguration(memory_limit=5120)])
        vGPUs.extend(tf.config.experimental.list_logical_devices('GPU'))

    # each GPU ==> 5 virtual vGPUs
    print(vGPUs)
    assert False

    args_pool = []
    for i, n_lat in enumerate(range(10, 201, 10)):
        args = (vGPUs[i].name, n_lat, args.norm_beta, args.seed, args.epochs, args.batch_size, 
                args.learning_rate, args.savedir, args.verbose_batch, args.verbose_epoch, args.batch_lim_debug)
    
    with multiprocessing.Pool(processes=20) as pool:
        pool.starmap(train_func, args)
    
