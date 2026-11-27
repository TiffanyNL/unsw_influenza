import emcee
import corner
import numpy as np
import matplotlib.pyplot as plt

reader = emcee.backends.HDFBackend("mcmc_calibration.h5")

tau = reader.get_autocorr_time()
burnin = int(2 * np.max(tau))
thin = int(0.5 * np.min(tau))
samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

log_prob_samples = reader.get_log_prob(discard=burnin, flat=True, thin=thin)
log_prior_samples = reader.get_blobs(discard=burnin, flat=True, thin=thin)

fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
samples = reader.get_chain(discard=burnin, thin=thin)
labels = ["amplitude", "R0", "phase-shift"]
ndim = len(labels)
for i in range(ndim):
    ax = axes[i]
    ax.plot(samples[:, :, i], "k", alpha=0.3)
    ax.set_xlim(0, len(samples))
    ax.set_ylabel(labels[i])
    ax.yaxis.set_label_coords(-0.1, 0.5)
axes[-1].set_xlabel("step number")

flat_samples = reader.get_chain(discard=burnin, flat=True, thin=thin)

fig = corner.corner(flat_samples, labels=labels)
