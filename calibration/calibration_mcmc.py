import numpy as np
from default_params import *
from plotnine import *
from calibration_functions import *
from acefa_functions import *
import pickle
import multiprocessing
import time
from scipy.stats import truncnorm
import matplotlib.pyplot as plt


if __name__ == '__main__':

    nndss_data = pd.read_pickle('results/nndss_data.pkl.gz')
    p_healthseeking_all = pd.read_pickle('results/p_healthseeking_all.pkl.gz')

    # ml results
    ml_results = np.load('results/calibration_results_minus_wcJan.npz', allow_pickle=True)
    ml_R0 = ml_results['R0_vals']
    ml_phase_shift = ml_results['phase_shift_vals']
    ml_amplitude = ml_results['amplitude_vals']
    SW_vals = ml_results['SW_vals']

    #############################
    ##### running the model #####
    #############################
    mcmc_results = []

    ndim = 3
    nwalkers = 30

    SW_ind = np.where(np.round(SW_vals, 2) == 0.15)

    rng = np.random.RandomState(123)

    p0 = np.concatenate(([rng.beta(a=2.5, b=5, size=nwalkers)],
                         [rng.gamma(shape=1.5, scale=1, size=nwalkers)],
                         [rng.uniform(low=0, high=365/2, size=nwalkers)]), axis=0).T

    #p0 = np.concatenate(([truncnorm.rvs(loc = ml_amplitude[SW_ind], scale = 0.05, a = (0 - ml_amplitude[SW_ind])/0.05, b = (1 - ml_amplitude[SW_ind])/0.05, size=nwalkers, random_state = rng)],
    #                     [truncnorm.rvs(loc = ml_R0[SW_ind], scale = 0.15, a = (0 - ml_R0[SW_ind])/0.15, b = (3 - ml_R0[SW_ind])/0.15, size=nwalkers, random_state = rng)],
    #                     [truncnorm.rvs(loc = ml_phase_shift[SW_ind], scale = 1, a = (0 - ml_phase_shift[SW_ind])/1, b = (365/2 - ml_phase_shift[SW_ind])/1,  size=nwalkers, random_state = rng)]
    #                     ),
    #                    axis=0).T

    initial_SW = np.repeat(SW_vals[SW_ind], n_ages)
    vaccine_uptake = 'status_quo'
    vaccine_policy = 'status_quo'
    # warm up
    warm_up = 100
    cores = 5

    filename = "mcmc_calibration.h5"
    backend = emcee.backends.HDFBackend(filename)
    backend.reset(nwalkers, ndim)

    max_n = 10000

    # track how the average autocorrelation time estimate changes
    index = 0
    autocorr = np.empty(max_n)

    # for testing convergence
    old_tau = np.inf

    with multiprocessing.Pool(processes=cores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers=nwalkers, ndim=ndim, log_prob_fn=log_posterior, pool=pool, args=[nndss_data, p_healthseeking_all, initial_SW, default_parameters, vaccine_uptake, vaccine_policy], backend=backend)

        for sample in sampler.sample(p0, iterations=max_n, progress=True):
            # Only check convergence every 100 steps
            if sampler.iteration % 100:
                continue

            # Compute the autocorrelation time so far
            # Using tol=0 means that we'll always get an estimate even
            # if it isn't trustworthy
            tau = sampler.get_autocorr_time(tol=0)
            autocorr[index] = np.mean(tau)
            index += 1

            # Check convergence
            converged = np.all(tau * 100 < sampler.iteration)
            converged &= np.all(np.abs(old_tau - tau) / tau < 0.01)
            if converged:
                break
            old_tau = tau