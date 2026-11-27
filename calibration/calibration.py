import numpy as np
import pandas as pd
from default_params import *
from plotnine import *
from calibration_functions import *
from acefa_functions import *
import pickle
import multiprocessing
import itertools

## state abbreviations
if __name__ == '__main__':
    nndss_data = pd.read_pickle('results/nndss_data.pkl')
    p_healthseeking_all = pd.read_pickle('results/p_healthseeking_all.pkl')

    #############################
    ##### running the model #####
    #############################

    np.max(nndss_data[nndss_data['age_group'] == '18-64']['cases'])

    SW_vals = np.arange(0.05, 0.8, 0.1)
    seed_vals = [1E-8, 1E-7, 1E-6, 1E-5, 1E-4]

    combinations = pd.DataFrame(list(itertools.product(SW_vals, seed_vals)), columns = ['SW_vals', 'seed'])

    optimization_results = []

    with multiprocessing.Pool(processes=9) as pool:
        async_results = [
            pool.apply_async(max_log_likelihood, (nndss_data, p_healthseeking_all, combinations['SW_vals'][i], combinations['seed'][i], default_parameters, status_quo_vac_parameters, True))
            for i in range(0, np.shape(combinations)[0])
        ]
        optimization_results = [res.get() for res in async_results]

    with open('results/optimization_results_p_hs_scale_wo_a_ps_365.pkl', 'wb') as f:
        pickle.dump(optimization_results, f)