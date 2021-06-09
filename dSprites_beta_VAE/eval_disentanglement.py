import tensorflow as tf
import numpy as np
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dsprites_beta_VAE.py import DspritesBetaVAE


def latent_to_index(latents):
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


def get_zdiffs(encoder, latent_dim, batches, batch_size):
    n_latent_real = len(latents_names[1:])
    z_diffs = np.zeros((n_latent_real, batches, latent_dim))
    latent_indices = np.zeros((n_latent_real, batches), dtype=int)

    for n, latent_label_to_fix in enumerate(latents_names[1:]):
        latent_index = n + 1
        print(n)

        latents_sampled_1 = sample_latent(size=batch_size * batches)
        latents_sampled_2 = sample_latent(size=batch_size * batches)

        latents_sampled_1[:, latent_index] = latents_sampled_2[:, latent_index]

        indices_sampled_1 = latent_to_index(latents_sampled_1)
        indices_sampled_2 = latent_to_index(latents_sampled_2)

        imgs_sampled_1 = imgs[indices_sampled_1]
        imgs_sampled_2 = imgs[indices_sampled_2]

        z_1 = encoder.predict(imgs_sampled_1)[0]
        z_2 = encoder.predict(imgs_sampled_2)[0]

        z_diff = np.abs(z_1 - z_2)
        print(z_diff.shape)

        z_diffs[n, :, :] = np.mean(
            z_diff.reshape((batches, batch_size, latent_dim)), axis=1
        )

        latent_indices[n, :] = n

    shuffle_index = np.arange(0, n_latent_real * batches)
    np.random.shuffle(shuffle_index)

    z_diffs = z_diffs.reshape((n_latent_real * batches, latent_dim))[shuffle_index]

    latent_indices = latent_indices.reshape((n_latent_real * batches))[shuffle_index]
    # latent_indices = np.eye(n_latent_real)[latent_indices]

    return {
        "z_diffs": z_diffs,
        "latent_indices": latent_indices,
    }

def compute_disentanglement_metric(model,
    batch_size = 64, training_batches = 500, testing_batches = 100):
    # prep training data
    training_data = get_zdiffs(model.encoder, model.latent_dim, training_batches, batch_size)

    x_train = np.array(training_data["z_diffs"])
    y_train = np.array(training_data["latent_indices"])

    # sklearn linear classifier
    classifier = make_pipeline(
        StandardScaler(), SGDClassifier(loss="log", early_stopping=True)
    )

    # train
    classifier.fit(x_train, y_train)

    # get testing data to evaluate disentanglement metric
    test_data = get_zdiffs(model.encoder, model.latent_dim, testing_batches, batch_size)

    x_test = np.array(test_data["z_diffs"])
    y_test = np.array(test_data["latent_indices"])

    # evaluate disentanglement metric
    disentanglement_score = classifier.score(x_test, y_test)
    return disentanglement_score

def compute_mutual_info_metric(model):
    return None

if __name__ == "__main__":

    # Load dataset
    dataset_zip = np.load(
        "dsprites_ordered.npz",
        allow_pickle=True,
        encoding="latin1",
    )

    imgs = dataset_zip["imgs"]
    latents_values = dataset_zip["latents_values"]
    latents_classes = dataset_zip["latents_classes"]
    metadata = dataset_zip["metadata"][()]

    # Define number of values per latents and functions to convert to indices
    latents_sizes = metadata["latents_sizes"]
    latents_names = metadata["latents_names"]
    latents_bases = np.concatenate(
        (
            latents_sizes[::-1].cumprod()[::-1][1:],
            np.array(
                [
                    1,
                ]
            ),
        )
    )

    # compute scores for models
    output_path = Path('output_train/')
    for model_path in output_path.iterdir():
        # instantiate model
        model = DspritesBetaVAE()
        model.load_model_weights(model_path / 'weights_encoder.h5', model_path / 'weights_decoder.h5')

        # compute disentanglement metric
        dis_metric = compute_disentanglement_metric(model)

        # compute mutual information gap
        mutual_info_metric = compute_mutual_info_metric(model)

        # save scores
        np.savetxt(model_path / 'disentanglement_metric.txt', dis_metric)
        np.savetxt(model_path / 'mutual_info_metric.txt', mutual_info_metric)