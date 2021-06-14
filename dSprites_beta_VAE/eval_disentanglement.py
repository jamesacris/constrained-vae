import tensorflow as tf
import numpy as np
import ast
from pathlib import Path
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from dsprites_beta_VAE import DspritesBetaVAE


def latent_to_index(latents, latents_sizes):
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
    return np.dot(latents, latents_bases).astype(int)


def sample_latent(latents_sizes, size=1):
    samples = np.zeros((size, latents_sizes.size))
    for lat_i, lat_size in enumerate(latents_sizes):
        samples[:, lat_i] = np.random.randint(lat_size, size=size)
    return samples


def get_zdiffs(encoder, latent_dim, batches, batch_size):
    
    n_latent_real = len(latents_names)
    z_diffs = np.zeros((n_latent_real, batches, latent_dim))
    latent_indices = np.zeros((n_latent_real, batches), dtype=int)

    for latent_index, latent_label_to_fix in enumerate(latents_names):

        latents_sampled_1 = sample_latent(latents_sizes, size=batch_size * batches)
        latents_sampled_2 = sample_latent(latents_sizes, size=batch_size * batches)

        latents_sampled_1[:, latent_index] = latents_sampled_2[:, latent_index]

        indices_sampled_1 = latent_to_index(latents_sampled_1, latents_sizes)
        indices_sampled_2 = latent_to_index(latents_sampled_2, latents_sizes)

        imgs_sampled_1 = imgs[indices_sampled_1]
        imgs_sampled_2 = imgs[indices_sampled_2]

        z_1 = encoder.predict(imgs_sampled_1)[0]
        z_2 = encoder.predict(imgs_sampled_2)[0]

        z_diff = np.abs(z_1 - z_2)

        z_diffs[latent_index, :, :] = np.mean(
            z_diff.reshape((batches, batch_size, latent_dim)), axis=1
        )

        latent_indices[latent_index, :] = latent_index

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

def get_latent_posterior(model, batch_size=64):
    # get inferred latent representation for whole dataset
    dataset_tf = tf.data.Dataset.from_tensor_slices(imgs).batch(batch_size)

    z = tf.zeros([imgs.shape[0], 3, model.latent_dim])
    for ibatch, batch in enumerate(dataset_tf):
        # make predictions
        idcs = slice(ibatch*batch_size, (ibatch+1)*batch_size)
        z[idcs, :, :] = model.encoder.predict(batch)
        
    return z

def estimate_latent_entropies(q_zCx, n_samples_per_ground_truth_value=10000):
    # initialise H_z with size latent_dim
    H_z = tf.zeros(q_zCx.shape[-1])
    print(H_z.shape)

    # stratified sampling from q(z|x)
    shape = list(np.sum(latents_sizes)).append(q_zCx.shape)
    print(shape)
    samples_qzCx = tf.zeros(shape)
    i = 0
    for ground_truth_factor in latents_names:
        for ival, val in enumerate(latents_values[ground_truth_factor]):
            # sample latent space with size n_samples_per_ground_truth_value
            latent_samples = sample_latent(latents_sizes, size=n_samples_per_ground_truth_value)
            # fix ground truth factor of interest
            latent_samples[:, ground_truth_factor] = val
            # get latent indicies (position in dataset) of latent samples
            idcs = latent_to_index(latent_samples, latents_sizes)
            # get corresponding samples of q(z|x) and store
            samples_qzCx[i + ival, :, :, :] = q_zCx[idcs, :, :]
            i += ival
    
    z_mean = samples_qzCx[:, :, 0, :]
    z_log_var = samples_qzCx[:, :, 1, :]
    z_output = samples_qzCx[:, :, 2, :]
    log_N = np.log(len_dataset)


    H_z /= batches
    return H_z

def compute_mutual_info_metric(model):
    # empirical distribution q(z|x) - sample z
    q_zCx = get_latent_posterior(model)

    # marginal entropy H(z_j)
    H_z = estimate_latent_entropies(q_zCx)

    # conditional entropy H(z|v)
    samples_zCx = samples_zCx.view(latents_sizes, model.latent_dim)
    params_zCx = tuple(p.view(latents_sizes, model.latent_dim) for p in params_zCx)
    H_zCv = estimate_H_zCv(samples_zCx, params_zCx, lat_sizes, lat_names)

    # I[z_j;v_k] = E[log \sum_x q(z_j|x)p(x|v_k)] + H[z_j] = - H[z_j|v_k] + H[z_j]
    mut_info = - H_zCv + H_z
    sorted_mut_info = tf.clip_by_value(tf.sort(mut_info, axis=1, direction='DESCENDING'), clip_value_min=0) # figure out how to sort highest to lowest in tf

    # difference between the largest and second largest mutual info
    delta_mut_info = sorted_mut_info[:, 0] - sorted_mut_info[:, 1]

    # NOTE: currently only works if balanced dataset for every factor of variation
    # then H(v_k) = - |V_k|/|V_k| log(1/|V_k|) = log(|V_k|)
    H_v = tf.math.log(tf.convert_to_tensor(latents_sizes, dtype=float))
    mig_k = delta_mut_info / H_v

    # mean over generative factors
    mut_info_metric = mig_k.reduce_mean(axis=1)

    return mut_info_metric

if __name__ == "__main__":

    # Load dataset
    dataset_zip = np.load(
        "dsprites_ordered.npz",
        allow_pickle=True,
        encoding="latin1",
    )

    imgs = dataset_zip["imgs"]
    len_dataset = len(imgs)
    print(len_dataset)
    metadata = dataset_zip["metadata"][()]

    latents_sizes = metadata["latents_sizes"][1:]
    latents_names = metadata["latents_names"][1:]
    latents_values = metadata["latents_possible_values"]
    print(latents_names)
    print(latents_sizes)
    print(latents_values)
    assert False

    # compute scores for models
    output_path = Path('output_train/')
    for model_path in output_path.iterdir():
        print(f'looking for models in {model_path}')
        h5 = model_path / 'weights_encoder.h5'
        if h5.exists():
            print('found a model')
            with open(model_path / 'hypar_network.txt', 'r') as f:
                hypars = ast.literal_eval(f.read())
                print(hypars)
            # instantiate model
            model = DspritesBetaVAE(normalized_beta=hypars['normalized_beta'], latent_dim=hypars['latent_dim'], n_filters_first_conv2d=32)
            model.load_model_weights(model_path / 'weights_encoder.h5', model_path / 'weights_decoder.h5')

            # compute disentanglement metric
            dis_metric = compute_disentanglement_metric(model)

            # compute mutual information gap
            # mutual_info_metric = compute_mutual_info_metric(model)

            # save scores
            np.savetxt(model_path / 'disentanglement_metric.txt', dis_metric)
            # np.savetxt(model_path / 'mutual_info_metric.txt', mutual_info_metric)