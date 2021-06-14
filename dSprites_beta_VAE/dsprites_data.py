import tensorflow_datasets as tfds
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

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
        
    def compute_zdiff_y(self, bvae, n_zdiff_per_y, n_img_per_zdiff):
        # create arrays
        y_size = self.latent_sizes.size
        z_diff_all = np.zeros((y_size, n_zdiff_per_y, bvae.latent_dim))
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
            z1 = bvae.encoder.predict(x1)[0]
            z2 = bvae.encoder.predict(x2)[0]
            # z_diff
            z_diff = np.abs(z1 - z2)
            # separate dimensions: n_zdiff_per_y, n_img_per_zdiff
            z_diff = z_diff.reshape((n_zdiff_per_y, n_img_per_zdiff, bvae.latent_dim))
            # take average over n_img_per_zdiff
            z_diff_all[y, :, :] = np.mean(z_diff, axis=1)
            # y
            y_all[y, :] = y
    
        # merge dimensions: y_size, n_zdiff_per_y
        z_diff_all = z_diff_all.reshape((y_size * n_zdiff_per_y, bvae.latent_dim))
        y_all = y_all.reshape((y_size * n_zdiff_per_y))
        
        # shuffle z_diff and y consistently
        shuffle_indices = np.arange(0, y_size * n_zdiff_per_y)
        np.random.shuffle(shuffle_indices)
        z_diff_all = z_diff_all[shuffle_indices]
        y_all = y_all[shuffle_indices]
        return z_diff_all, y_all
        
    def compute_disentangle_metric_score(self, bvae, n_zdiff_per_y=5000, 
        n_img_per_zdiff=64, random_seed=0):
        # seed
        np.random.seed(random_seed)
        # prep training and test data
        zdiff, y = self.compute_zdiff_y(bvae, n_zdiff_per_y, n_img_per_zdiff)
        # sklearn linear classifier
        classifier = make_pipeline(
            StandardScaler(), 
            SGDClassifier(loss="log", early_stopping=True, random_state=random_seed)
        )
        # train
        classifier.fit(zdiff, y)
        # score with test data
        return classifier.score(zdiff, y)
