import matplotlib.pyplot as plt


# logging helper function
def log(logits, training_logs):
    training_logs["losses"].append(logits["loss"])
    training_logs["reconstruction_losses"].append(logits["reconstruction_loss"])
    training_logs["kld_losses"].append(logits["kl_loss"])
    training_logs["Lambdas"].append(logits["lambda"])
    # training_logs["kld_diff"].append(logits["kld_diff"])


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
    # save
    # plt.savefig(f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/loss_{KLD_aim}_KLDaim_{nd}_nd.png")
    plt.show()


# plot reconstruction losses
def plot_reconstruction_losses(reconstruction_losses):
    plt.figure(dpi=100)
    plt.plot(reconstruction_losses)
    plt.xlabel("Training step")
    plt.ylabel("Reconstruction Loss")
    # save
    # plt.savefig(
        # f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTrecnstr_loss_{KLD_aim}_KLDaim_{nd}_nd.png"
    # )
    plt.show()


# plot kld losses
def plot_kld_lossses(kld_losses):
    plt.figure(dpi=100)
    plt.plot(kld_losses)
    plt.xlabel("Training step")
    plt.ylabel("KL Loss")
    # plt.yscale("log")
    # save
    # plt.savefig(
        # f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTkld_loss_{KLD_aim}_KLDaim_{nd}_nd.png"
    # )
    plt.show()


# plot kld diffs
# def plot_kld_diffs(kld_diff):
#     plt.figure(dpi=100)
#     plt.plot(kld_diff)
#     plt.xlabel("Training step")
#     plt.ylabel("KLD - KLD_aim")
#     plt.yscale("log")
#     # save
#     # plt.savefig(
#         # f"figs/dual_paper_alg/{KLD_aim}_KLD_aim/TESTkld_diff_{KLD_aim}_KLDaim_{nd}_nd.png"
#     # )
#     plt.show()


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