from mpi4py import MPI
import tensorflow as tf
import time

import argparse
from dsprites_beta_VAE import DspritesBetaVAE
from dsprites_data import OrderedDsprites

if __name__ == "__main__":
    # MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # args
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-b", "--norm-beta-list", type=float, nargs='+', required=True, help="list of normalised betas")
    parser.add_argument("-z", "--nlat-list", type=int, nargs='+', required=True, help="list of latent dimensions")
    parser.add_argument("--seed-model", default=0, type=int, help="random seed to initialise model weights")
    parser.add_argument("--n-zdiff-per-y",default=5000, type=int, help="(disentanglement metric) number of zdiffs to use per ground truth factor to train linear classifier")
    parser.add_argument("--n-img-per-zdiff",default=256, type=int, help="(disentanglement metric) number of images to use to compute each zdiff")
    parser.add_argument("--num_threads", default=1, type=int, help="max threads per job")
    parser.add_argument("--disable-gpu", default=False, action='store_true', help="use cpu for training")
    parser.add_argument("--virtual-gpu-mem", default=None, type=int,
        help="virtual gpu memory; use None to disable virtual GPU")
    parser.add_argument("--seed", default=0, type=int, help="random seed for linear classifier")
    args = parser.parse_args()

    ########### env ###########
    # check mpi size
    njobs = len(args.norm_beta_list) * len(args.nlat_list)
    assert njobs == comm.Get_size()
    # cpu threads
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

    # create bvae
    ibeta = rank % len(args.norm_beta_list)
    ilat = rank // len(args.norm_beta_list)
    bvae = DspritesBetaVAE(normalized_beta=args.norm_beta_list[ibeta],
        latent_dim=args.nlat_list[ilat], n_filters_first_conv2d=32,
        random_seed=args.seed_model)
    # filename
    folder = f'output_train/beta={args.norm_beta_list[ibeta]}__nlat={args.nlat_list[ilat]}__nConv2D=32__seed={args.seed_model}'
    bvae.load_model_weights(folder + '/weights_encoder.h5', folder + '/weights_decoder.h5')


    # train and save
    # save_dir: where to save all results (use None for automatic dir)
    # batch_limit_for_debug: how many batches to use per epoch (for quick debug)
    #                        set batch_limit_for_debug=None to use all batches
    with tf.device(device):
        ordered_dsprites = OrderedDsprites()
        dis_metric = ordered_dsprites.compute_disentangle_metric_score(bvae,
            n_zdiff_per_y=args.n_zdiff_per_y, n_img_per_zdiff=args.n_img_per_zdiff,
            random_seed=args.seed)
        with open(folder + f'/disentanglement_metric__samples={args.n_zdiff_per_y}__imgbatch={args.n_img_per_zdiff}__seed={args.seed}.txt', 'w') as f:
            f.write(str(dis_metric))
