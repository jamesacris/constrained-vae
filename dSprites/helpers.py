import matplotlib.pyplot as plt


# logging helper function
def log(logits, training_logs):
    training_logs["losses"].append(logits["loss"])
    training_logs["reconstruction_losses"].append(logits["reconstruction_loss"])
    training_logs["kld_losses"].append(logits["kl_loss"])


# reporting helper function
def report(batch_size, step, logits):
    print(f"\nTraining logs at step {step}:")
    for metric, value in logits.items():
        print(metric, value.numpy())
    print("Seen: %d samples" % ((step + 1) * batch_size))


# plot losses
def plot_losses(losses):
    plt.figure(dpi=100)
    plt.plot([abs(loss) for loss in losses])
    plt.xlabel("Training step")
    plt.ylabel("Loss (absolute value)")
    plt.yscale("log")
    # save
    plt.savefig(f"figs/BVAE/loss_dense_50ep.png")
    plt.show()


# plot reconstruction losses
def plot_reconstruction_losses(reconstruction_losses):
    plt.figure(dpi=100)
    plt.plot(reconstruction_losses)
    plt.xlabel("Training step")
    plt.ylabel("Reconstruction Loss")
    plt.yscale("log")
    # save
    plt.savefig(
        f"figs/BVAE/reconstr_loss_dense_50ep.png"
    )
    plt.show()


# plot kld losses
def plot_kld_lossses(kld_losses):
    plt.figure(dpi=100)
    plt.plot(kld_losses)
    plt.xlabel("Training step")
    plt.ylabel("KL Loss")
    plt.yscale("log")
    # save
    plt.savefig(
        f"figs/BVAE/kld_dense_50ep.png"
    )
    plt.show()


# plot lambdas
def plot_lambdas(Lambdas):
    plt.figure(dpi=100)
    plt.plot(Lambdas)
    plt.xlabel("Training step")
    plt.ylabel("Lambda")
    # save
    # plt.savefig(
        # f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTlambdas_{KLD_aim}_KLDaim_{nd}_nd.png"
    # )
    plt.show()


# function to plot an image in a subplot
def subplot_image(image, label, nrows=1, ncols=1, iplot=0, label2='', label2_color='r'):
    plt.subplot(nrows, ncols, iplot + 1)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(label, c='k', fontsize=12)
    plt.title(label2, c=label2_color, fontsize=12, y=-0.33)
    plt.xticks([])
    plt.yticks([])