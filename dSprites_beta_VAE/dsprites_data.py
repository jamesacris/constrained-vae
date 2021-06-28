import tensorflow_datasets as tfds
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from mig_compute import estimate_entropies

def get_dsprites_tf_dataset():
    # data have been shuffled by tensorflow
    # data will be batched before training
    return tfds.load('Dsprites', split='train', batch_size=256)

class OrderedDsprites:
    def __init__(self, data_file='dsprites_ordered.npz'):
        # load dataset
        dataset = np.load(data_file, allow_pickle=True, encoding="latin1")
        self.imgs = dataset["imgs"][:]
        self.latent_sizes = dataset["metadata"][()]["latents_sizes"][:]
        # get rid of color dimension here
        self.latent_sizes = self.latent_sizes[1:]
        self.latent_bases = np.concatenate((
            self.latent_sizes[::-1].cumprod()[::-1][1:], np.array([1,]),))

    def sample_latent(self, nsamples=1):
        samples = np.zeros((nsamples, self.latent_sizes.size), dtype=int)
        for lat_i, lat_size in enumerate(self.latent_sizes):
            samples[:, lat_i] = np.random.randint(lat_size, size=nsamples)
        return samples

    def get_images_from_latent(self, latent_samples):
        # latent to indices
        indices = np.dot(latent_samples, self.latent_bases).astype(int)
        return self.imgs[indices]
        
    def compute_zdiff_y(self, vae, n_zdiff_per_y, n_img_per_zdiff):
        # create arrays
        y_size = self.latent_sizes.size
        z_diff_all = np.zeros((y_size, n_zdiff_per_y, vae.latent_dim), dtype=np.float32)
        y_all = np.zeros((y_size, n_zdiff_per_y), dtype=int)
        
        for y in range(y_size):
            # sample
            v1 = self.sample_latent(n_zdiff_per_y * n_img_per_zdiff)
            v2 = self.sample_latent(n_zdiff_per_y * n_img_per_zdiff)
            # keey y the same
            v1[:, y] = v2[:, y]
            # get images
            x1 = self.get_images_from_latent(v1)
            x2 = self.get_images_from_latent(v2)
            # encode    
            z1 = vae.encoder.predict(x1)[0]
            z2 = vae.encoder.predict(x2)[0]
            # z_diff
            z_diff = np.abs(z1 - z2)
            # separate dimensions: n_zdiff_per_y, n_img_per_zdiff
            z_diff = z_diff.reshape((n_zdiff_per_y, n_img_per_zdiff, vae.latent_dim))
            # take average over n_img_per_zdiff
            z_diff_all[y, :, :] = np.mean(z_diff, axis=1)
            # y
            y_all[y, :] = y
    
        # merge dimensions: y_size, n_zdiff_per_y
        z_diff_all = z_diff_all.reshape((y_size * n_zdiff_per_y, vae.latent_dim))
        y_all = y_all.reshape((y_size * n_zdiff_per_y))
        
        # shuffle z_diff and y consistently
        shuffle_indices = np.arange(0, y_size * n_zdiff_per_y)
        np.random.shuffle(shuffle_indices)
        z_diff_all = z_diff_all[shuffle_indices]
        y_all = y_all[shuffle_indices]
        return z_diff_all, y_all
        
    def compute_disentangle_metric_score(self, vae, n_zdiff_per_y=5000, 
        n_img_per_zdiff=64, random_seed=0):
        # seed
        np.random.seed(random_seed)
        # prep training and test data
        zdiff, y = self.compute_zdiff_y(vae, n_zdiff_per_y, n_img_per_zdiff)
        # sklearn linear classifier
        classifier = make_pipeline(
            StandardScaler(), 
            SGDClassifier(loss="log", early_stopping=True, random_state=random_seed)
        )
        # train
        classifier.fit(zdiff, y)
        # score with test data
        return classifier.score(zdiff, y)
                
    def compute_MIG(self, vae, n_samples):
        N = len(self.imgs)
        K = vae.latent_dim

        # encode all images
        qz_mean = np.zeros((N, K), dtype=np.float32)
        qz_log_var = np.zeros((N, K), dtype=np.float32)
        qz_sample = np.zeros((N, K), dtype=np.float32)
        batch_size = 256
        for ibatch, start in enumerate(range(0, N, batch_size)):
            end = min(start + batch_size, N)
            z_mean, z_log_var, z_sample = vae.encoder.predict(self.imgs[start:end])
            qz_mean[start:end] = z_mean
            qz_log_var[start:end] = z_log_var
            qz_sample[start:end] = z_sample
        
        # marginal entropies
        marginal_entropies = np.zeros((K,), dtype=np.float32)
        marginal_entropies = estimate_entropies(
            qz_mean, qz_log_var, qz_sample, n_samples)
        
        # conditional entropies
        cond_entropies = np.zeros((4, K), dtype=np.float32)
        
        # index slices for the entire structured data
        slices = []
        for sz in self.latent_sizes:
            slices.append(slice(sz))
        slices.append(slice(K))
        
        # reshape data to structured
        reshape_id = list(self.latent_sizes) + [K,]
        qz_mean = qz_mean.reshape(reshape_id)
        qz_log_var = qz_log_var.reshape(reshape_id)
        qz_sample = qz_sample.reshape(reshape_id)
        
        # iter over (scale, rotation, pos_x, pos_y)
        for index_y in range(1, len(self.latent_sizes)):
            slices_copy = slices.copy()
            n_y = self.latent_sizes[index_y]
            # iter over y
            for i_y in range(n_y):
                slices_copy[index_y] = i_y
                qz_mean_sub = qz_mean[tuple(slices_copy)].copy().reshape((N // n_y, K))
                qz_log_var_sub = qz_log_var[tuple(slices_copy)].copy().reshape((N // n_y, K))
                qz_sample_sub = qz_sample[tuple(slices_copy)].copy().reshape((N // n_y, K))
                cond_entropies = estimate_entropies(
                    qz_mean_sub, qz_log_var_sub, qz_sample_sub, n_samples)
                cond_entropies[index_y - 1, :] += cond_entropies / n_y
        
        # compute MIG
        factor_entropies = self.latent_sizes[1:]
        mutual_infos = marginal_entropies[None] - cond_entropies
        mutual_infos = np.clip(np.sort(mutual_infos, axis=1)[::-1], a_min=0, a_max=None)
        mi_normed = mutual_infos / np.log(factor_entropies)[:, None]
        mig = np.mean(mi_normed[:, 0] - mi_normed[:, 1])
        return mig
        
