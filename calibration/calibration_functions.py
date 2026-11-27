import numpy as np
import pandas as pd
from scipy import optimize
from scipy.optimize import dual_annealing
from scipy.stats import beta as scipy_beta
from scipy.stats import gamma as scipy_gamma
from scipy.stats import uniform as scipy_uniform
from scipy.stats import poisson as scipy_poisson
from acefa_functions import *
from math import isfinite

#########################################
##### functions to load in the data #####
#########################################

def load_flucan_data(filepath):
    df = pd.read_csv(filepath)
    df = df.groupby(['agegroup1', 'week', 'year'], observed=True).size().reset_index(name='hospitalisations')
    df = df.rename(columns={'agegroup1':'age_group'}, inplace=False)
    conditions = [df['week'] < 10, df['week'] >= 10]
    operation = [44 + df['week'], df['week'] - 9]
    df['week_cm'] = np.select(conditions, operation)
    df['week'] = df['week'].astype('float')
    df['week_cm'] = df['week_cm'].astype('float')
    df['year'] = pd.Categorical(df['year'], ordered=True)
    return df

def load_nndss_data(filepath):
    df = pd.read_csv(filepath)
    df['notification_date'] = pd.to_datetime(df['notification_date'], format="%Y-%m-%d")
    df['year'] = df['notification_date'].dt.year
    df['week'] = df['notification_date'].dt.isocalendar().week
    df = df.rename(columns={'location':'state', 'age_categories':'age_group'}, inplace=False)
    conditions = [df['week'] < 10, df['week'] >= 10]
    operation = [44 + df['week'], df['week'] - 9]
    df['week_cm'] = np.select(conditions, operation)
    df['week'] = df['week'].astype('float')
    df['week_cm'] = df['week_cm'].astype('float')
    df = df.groupby(['age_group', 'week', 'week_cm', 'year'], observed=True).agg(cases=('cases', 'sum')).reset_index()
    df['year'] = pd.Categorical(df['year'], ordered=True)
    return df

def load_flutracking_data(filepath):
    df = pd.read_csv(filepath)
    df['SurveyWeek'] = pd.to_datetime(df['SurveyWeek'], format="%Y-%m-%d")
    df['year'] = df['SurveyWeek'].dt.year
    df['week'] = df['SurveyWeek'].dt.isocalendar().week
    df = df.rename(columns={'RegionCode': 'state', 'AgeGroup': 'age_group'}, inplace=False)
    conditions = [df['week'] < 10, df['week'] >= 10]
    operation = [44 + df['week'], df['week'] - 9]
    df['week_cm'] = np.select(conditions, operation)
    df['week'] = df['week'].astype('float')
    df['week_cm'] = df['week_cm'].astype('float')
    df['year'] = pd.Categorical(df['year'], ordered=True)
    return df

####################################
##### data wrangling functions #####
####################################

def calculate_weekly_outcomes_by_age_for_cal(y, default_parameters, status_quo_vac_parameters, N_year, p_healthseeking, acefa_ages = True):

    #vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy = vaccine_policy, vaccine_scenario = vaccine_uptake)

    vc, VE_INF0, VE_GP0, VE_HOSP0 =  status_quo_vac_parameters

    # default parameters
    n_ages, n_states, N, populations_by_fractions, contact_matrix, sigma, gamma, omega, p_asymptomatic, p_hospitalised, reduced_sus, tspan, n_ages_acefa, N_all, cum_weekly_vacc_progress, vacc_dates = default_parameters

    y_IU, y_IV, y_IW = y[:, 15, :], y[:, 16, :], y[:, 17, :]

    i_unvacc = (y_IU[1:] - y_IU[:-1]) * N_year
    i_vacc = (y_IV[1:] - y_IV[:-1]) * N_year
    i_waning = (y_IW[1:] - y_IW[:-1]) * N_year

    my_infections = i_unvacc + i_vacc + i_waning

    ## Recalculate VE conditional on infection (Ioannidis 2022, European Journal of Epidemiology)
    VE_HOSP = 1 - np.divide(1 - VE_HOSP0, 1 - VE_INF0)
    VE_GP = 1 - np.divide(1 - VE_GP0, 1 - VE_INF0)
    #
    my_cases = np.multiply((1 - p_asymptomatic) * i_unvacc, p_healthseeking) + \
               np.multiply((1 - VE_GP) * (1 - p_asymptomatic) * i_vacc, p_healthseeking) + \
               np.multiply((1 - p_asymptomatic) * i_waning, p_healthseeking)

    my_hospitalizations = np.multiply((1 - p_asymptomatic) * i_unvacc, p_hospitalised) + \
                          np.multiply((1 - VE_HOSP) * (1 - p_asymptomatic) * i_vacc, p_hospitalised) + \
                          np.multiply((1 - p_asymptomatic) * i_waning, p_hospitalised)


    if my_infections.shape[0] % 7 != 0:
        print("Warning: The number of rows is not a multiple of 7. Truncating the array.")
        my_infections = my_infections[:-(my_infections.shape[0] % 7)]

    weekly_infections_by_age = my_infections.reshape(my_infections.shape[0] // 7, 7, my_infections.shape[1]).sum(1)

    if my_cases.shape[0] % 7 != 0:
        print("Warning: The number of rows is not a multiple of 7. Truncating the array.")
        my_cases = my_cases[:-(my_cases.shape[0] % 7)]

    weekly_cases_by_age = my_cases.reshape(my_cases.shape[0] // 7, 7, my_cases.shape[1]).sum(1)

    if my_hospitalizations.shape[0] % 7 != 0:
        print("Warning: The number of rows is not a multiple of 7. Truncating the array.")
        my_hospitalizations = my_hospitalizations[:-(my_hospitalizations.shape[0] % 7)]

    weekly_hospitalizations_by_age: object = my_hospitalizations.reshape(my_hospitalizations.shape[0] // 7, 7,
                                                                         my_hospitalizations.shape[1]).sum(1)

    # change to ACEFA age groups
    if acefa_ages:
        weekly_infections_by_age = np.column_stack((weekly_infections_by_age[:, 0] + weekly_infections_by_age[:, 1],
                                                    weekly_infections_by_age[:, 2],
                                                    weekly_infections_by_age[:, 3],
                                                    weekly_infections_by_age[:, 4],
                                                    weekly_infections_by_age[:, 5] + weekly_infections_by_age[:, 6],
                                                    weekly_infections_by_age[:, 7],
                                                    weekly_infections_by_age[:, 8]))

        weekly_cases_by_age = np.column_stack((weekly_cases_by_age[:, 0] + weekly_cases_by_age[:, 1],
                                               weekly_cases_by_age[:, 2],
                                               weekly_cases_by_age[:, 3],
                                               weekly_cases_by_age[:, 4],
                                               weekly_cases_by_age[:, 5] + weekly_cases_by_age[:, 6],
                                               weekly_cases_by_age[:, 7],
                                               weekly_cases_by_age[:, 8]))

        weekly_hospitalizations_by_age = np.column_stack(
            (weekly_hospitalizations_by_age[:, 0] + weekly_hospitalizations_by_age[:, 1],
             weekly_hospitalizations_by_age[:, 2],
             weekly_hospitalizations_by_age[:, 3],
             weekly_hospitalizations_by_age[:, 4],
             weekly_hospitalizations_by_age[:, 5] + weekly_hospitalizations_by_age[:, 6],
             weekly_hospitalizations_by_age[:, 7],
             weekly_hospitalizations_by_age[:, 8]))

    return [weekly_infections_by_age, weekly_cases_by_age, weekly_hospitalizations_by_age]

###############################
##### likelihood function #####
###############################
# initial_SW, seed should be after p_healthseeking_all if not fitted
def log_likelihood(pars, df_cases, p_healthseeking_all, init_SW, seed, default_parameters, status_quo_vac_parameters, max_likelihood = True):

    years = np.unique(df_cases['year'])
    R0, p_hs_scale_y, p_hs_scale_o, amplitude, phase_shift = pars
    n_years = len(years)

    # default parameters
    [n_ages, n_states, N, populations_by_fractions, contact_matrix, sigma, gamma, omega, p_asymptomatic, p_hospitalised, reduced_sus, tspan, n_ages_acefa, N_all, cum_weekly_vacc_progress, vacc_dates] = default_parameters

    initial_SW = np.repeat(init_SW, n_ages)

    model_runs = run_model_equations_weekly_vaccination(default_parameters=default_parameters, status_quo_vac_parameters=status_quo_vac_parameters,
                                                        initial_SW=initial_SW, R0=R0, amplitude=amplitude,
                                                        phase_shift=phase_shift, seed=seed)

    model_runs = model_runs.reshape((-1, n_states + 3, n_ages))

    age_specific_cases = []

    for i in range(0, n_years, 1):
        N_year_in = N_all.loc[N_all['year'] == years[i], "N"].values[0]
        p_healthseeking_subset = p_healthseeking_all.loc[p_healthseeking_all['year'] == years[i], :]
        p_healthseeking_in = np.array([p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '0-4', 'p'] * p_hs_scale_y,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '0-4', 'p'] * p_hs_scale_y,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '0-4', 'p'] * p_hs_scale_y,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '5-17', 'p'] * p_hs_scale_y,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '5-17', 'p'] * p_hs_scale_y,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '18-64', 'p'],
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '18-64', 'p'],
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '>65', 'p'] * p_hs_scale_o,
                                       p_healthseeking_subset.loc[p_healthseeking_subset['age_group'] == '>65', 'p'] * p_hs_scale_o]).flatten()

        age_specific_cases.append(calculate_weekly_outcomes_by_age_for_cal(model_runs, default_parameters, status_quo_vac_parameters, N_year_in, p_healthseeking_in, True)[1])

    sim_cases = np.empty(df_cases.shape[0])
    sim_cases[:] = np.nan

    for i in range(0, df_cases.shape[0]):
        sim_cases[i] = max(1E-6, age_specific_cases[df_cases['year_index'].iloc[i]][df_cases['time_index'].iloc[i], df_cases['age_index'].iloc[i]])

    df_cases['sim_cases'] = sim_cases

    log_likelihood_cases = scipy_poisson.logpmf(k = df_cases['cases'], mu = df_cases['sim_cases'])#nbinom.logpmf(k = df_cases['cases'], n = size, p = size / (size + df_cases['sim_cases']))

    if -np.inf in log_likelihood_cases:
        return -np.inf
    elif max_likelihood == False:
        return np.sum(log_likelihood_cases)
    else:
        return -np.sum(log_likelihood_cases)

# initial_SW, seed, should be after p_healthseeking_all if not fitted
def max_log_likelihood(df_cases, p_healthseeking_all, init_SW, seed, default_parameters, status_quo_vac_parameters, dual_anneal = False):
    years = np.unique(df_cases['year'])
    n_years = len(years)

    upper_limit_p = 1/max(p_healthseeking_all['p'])

    bounds_in = [(0.25, 3.5)] + [(0.01, upper_limit_p)] + [(0.01, upper_limit_p)] + [(0, 1)] + [(0, 365.25)]

    x0_in = np.array([1, 1, 1, 1, 1])

    if dual_anneal == True:
        out = dual_annealing(func=log_likelihood,
                             minimizer_kwargs={'method': 'Nelder-Mead', 'bounds': bounds_in},
                             bounds=bounds_in,
                             x0=x0_in,
                             args=(df_cases, p_healthseeking_all, init_SW, seed,
                                   default_parameters, status_quo_vac_parameters, True),
                             maxiter=2250
                             # options = {'maxiter': 5000, 'xatol': 1e-4, 'fatol': 1e-2}
                             )
    else:
        out = optimize.minimize(fun=log_likelihood, x0=x0_in,
                                args=(df_cases, p_healthseeking_all, init_SW, seed,
                                      default_parameters, status_quo_vac_parameters, True),
                                method='Nelder-Mead', bounds=bounds_in, options = {'maxiter': 10000, 'xatol': 1e-4, 'fatol': 1e-2})

    return out

def log_prior(pars):
    R0, p_hs_scale_y, p_hs_scale_o, init_SW, seed, amplitude, phase_shift = pars
    p_R = scipy_gamma.logpdf(R0, a = 10, scale = 1.5/10)
    p_p_y = scipy_uniform.logpdf(p_hs_scale_y, loc = 0, scale = 6.03)
    p_p_o = scipy_uniform.logpdf(p_hs_scale_o, loc=0, scale = 6.03)
    p_init_SW = scipy_uniform.logpdf(init_SW, loc=0.15, scale = 0.001)#scipy_beta.logpdf(init_SW, a = 100, b = (100 - 0.15 * 100)/0.15)
    p_seed = scipy_beta.logpdf(seed, a = 1.5, b = (1.5 - 0.00001*1.5)/0.00001)
    p_amplitude = scipy_uniform.logpdf(amplitude, loc = 0, scale = 1)
    p_phase_shift = scipy_uniform.logpdf(phase_shift, loc = 0, scale = 365.25/2)
    return (p_R + p_p_y + p_p_o + p_init_SW + p_seed + p_amplitude + p_phase_shift)

def log_posterior(pars, df_cases, p_healthseeking_all, default_parameters, status_quo_vac_parameters): #  seed should be after p_health_seeking_all

    priors = log_prior(pars=pars)

    if not isfinite(priors):
        return -np.inf
    else:
        log_lik = log_likelihood(pars = pars, df_cases = df_cases, p_healthseeking_all = p_healthseeking_all, default_parameters = default_parameters, status_quo_vac_parameters = status_quo_vac_parameters, max_likelihood = False)
        posterior = priors + log_lik
        if not isfinite(posterior):
            return -np.inf
        else:
            return posterior