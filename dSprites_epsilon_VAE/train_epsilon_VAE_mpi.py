from mpi4py import MPI
import tensorflow as tf

import argparse
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import get_dsprites_tf_dataset

if __name__ == "__main__":
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # args
    # TODO: correct args for epsilon-vae
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--norm-epsilon-list", type=float, nargs='+', required=True, help="list of normalised betas")
    parser.add_argument("-z", "--nlat-list", type=int, nargs='+', required=True, help="list of latent dimensions")
    parser.add_argument("--nfilters", default=32, type=int, help="number of filters in the first Conv2D layer")
    parser.add_argument("--seed", default=0, type=int, help="random seed to initialise model weights")
    parser.add_argument("--epochs", default=20, type=int, help="epochs")
    parser.add_argument("--batch-size", default=256, type=int, help="batch size")
    parser.add_argument("--learning-rate", default=.01, type=float, help="learning rate")
    parser.add_argument("--batch-lim-debug", default=None, type=int, help="how many mini-batches used per epoch for quick debug")
    parser.add_argument("--verbose-batch", default=0, type=int, help="interval to print batch info")
    parser.add_argument("--verbose-epoch", default=0, type=int, help="interval to print epoch info")
    parser.add_argument("--num_threads", default=1, type=int, help="max threads per job")
    parser.add_argument("--disable-gpu", default=False, action='store_true', help="use cpu for training")
    parser.add_argument("--virtual-gpu-mem", default=None, type=int, 
        help="virtual gpu memory; use None to disable virtual GPU")
    args = parser.parse_args()
    
    ########### env ###########
    # check mpi size
    njobs = len(args.norm_beta_list) * len(args.nlat_list)
    assert njobs == comm.Get_size()
    # cpu threads
    tf.config.threading.set_inter_op_parallelism_threads(args.num_threads)
    tf.config.threading.set_intra_op_parallelism_threads(args.num_threads)
    # device
    if args.disable_gpu:
        vCPUs = tf.config.experimental.list_logical_devices('CPU')
        device = vCPUs[0]
    else:
        pGPUs = tf.config.list_physical_devices('GPU')
        if args.virtual_gpu_mem is not None:
            nvGPUs_per_pGPU = njobs // len(pGPUs) + int(njobs % len(pGPUs) > 0)
            for pGPU in pGPUs:
                tf.config.experimental.set_virtual_device_configuration(
                    pGPU, [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=args.virtual_gpu_mem)] * nvGPUs_per_pGPU)
        else:
            assert len(pGPUs) >= njobs
            for pGPU in pGPUs:
                tf.config.experimental.set_memory_growth(pGPU, True)
        vGPUs = tf.config.experimental.list_logical_devices('GPU')
        device = vGPUs[rank]
    ########### env ###########
    
    # dataset
    dset = get_dsprites_tf_dataset()

    # create bvae
    ibeta = rank % len(args.norm_beta_list)
    ilat = rank // len(args.norm_beta_list)
    bvae = DspritesBetaVAE(normalized_beta=args.norm_beta_list[ibeta], 
        latent_dim=args.nlat_list[ilat], n_filters_first_conv2d=args.nfilters,
        random_seed=args.seed)

    # train and save
    # save_dir: where to save all results (use None for automatic dir)
    # batch_limit_for_debug: how many batches to use per epoch (for quick debug)
    #                        set batch_limit_for_debug=None to use all batches
    with tf.device(device):
        bvae.train_save(dset, epochs=args.epochs, batch_size=args.batch_size, 
            lr=args.learning_rate, save_dir=None, verbose_batch=args.verbose_batch,
            verbose_epoch=args.verbose_epoch, batch_limit_for_debug=args.batch_lim_debug)
