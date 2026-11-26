"""Default parameters for the UNSW ACEFA influenza vaccination model.

Extracted from default_params_unsw.py without changing numerical parameter values.
Requires the original population workbook and contact-matrix CSV in data/.
This module alone does not implement the model or vaccination schedules.
"""

# from pathlib import Path
import numpy as np
import pandas as pd

state_names = ["SU", "E1U", "E2U", "IU", "RU",
               "SV", "E1V", "E2V", "IV", "RV",
               "SW", "E1W", "E2W", "IW", "RW"]
               # "CaseIU", "CaseIV", "CaseIW"

age_names = ['0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '80+']
acefa_ages = ['<1', '1-4', '5-11', '12-17', '18-64', '65-79', '>80']

SU, E1U, E2U, IU, RU = range(5)
SV, E1V, E2V, IV, RV = range(5, 10)
SW, E1W, E2W, IW, RW = range(10, 15)
CaseIU, CaseIV, CaseIW = range(15, 18)  ## infection counters

def load_population_australia(sheet_name='AUS'):
    ''' ABS: population clock and pyramid 2024 (by yearly age)
    https://www.abs.gov.au/statistics/people/population/population-clock-pyramid

    Population pyramid for Australia 2024 https://www.populationpyramid.net/australia/2024/
    returns dataframe of Australian population by age groups and percentages
    '''

    df = pd.read_excel('data/population_australia_2024_abs_australia.xlsx',  sheet_name=str(sheet_name.upper()))

    p = df.copy()
    total_population = np.sum(p["Male"]) + np.sum(p["Female"])

    p.insert(3, "Percent", (p["Male"] + p["Female"]) / total_population, True)

    C = np.array([p["Percent"][0] * .5,  # 0 to 5 months
                  p["Percent"][0] * .5,  # 6 to 11 months
                  p["Percent"][1:5].sum(),  # 1 to 4Y
                  p["Percent"][5:12].sum(),  # 5 to 11Y
                  p["Percent"][12:18].sum(),  # 12 to 17Y
                  p["Percent"][18:50].sum(),  # 18 to 49Y
                  p["Percent"][50:65].sum(),  # 50 to 64Y
                  p["Percent"][65:80].sum(),  # 65 to 79Y
                  p["Percent"][80:].sum()  # 80Y +
                  ])

    data = {"age": ['0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '80+'],
            "proportion": C}

    df2 = pd.DataFrame(data)
    return df2

def load_contact_matrix(n_ages = 9):
    ''' my_population is the population percentages by age group.
    contact matrix from POLYMOD study '''

    C = np.ones((n_ages, n_ages))

    if n_ages == 9:
        C = pd.read_csv('data/ACEFA_UK_POLYMOD_contact_matrix.csv', header=None).to_numpy()
    return C

def load_reduced_susceptibility(n_ages=9):
    ''' For adults (18Y +), r = 0.21 (Hill2009)
    For children (0-18), r grows linearly by year (assumed)'''
    temp = np.linspace(0, 0.21, 19)[1:]
    reduced_sus = np.array([temp[0] * 0.25,  ## 0-5 month
                            temp[0] * 0.75,  ## 6-11 month
                            np.mean(temp[:4]),  ## 1-4Y
                            np.mean(temp[4:11]),  ## 5-11Y
                            np.mean(temp[11:17]),  ## 12-17
                            0.21, 0.21, 0.21, 0.21  ## 18+
                            ])
    return reduced_sus


def load_VE_and_VC_IIV(vaccine_policy='status_quo', vaccine_scenario='pessimistic'):
    ''' for each vaccine policy (status_quo, laiv_5_12, laiv_5_18, iiv_50, iiv_65) and
    each vaccine scenario (status_quo, pessimistic, mid, optimistic):
    loads the vaccine coverage and vaccine effectiveness.

    In this function, VC is only IIV.

    'status_quo' vaccine policy will give status quo VE and VC (regardless of vaccine scenario)
    '''

    n_ages = 9

    if vaccine_scenario.lower() not in ['status_quo', 'pessimistic', 'mid', 'optimistic']:
        print('Warning: vaccine_scenario is not status_quo, pessimistic, mid, or optimistic.')
    if vaccine_policy.lower() not in ['status_quo', 'laiv_5_12', 'laiv_5_18', 'iiv_50', 'iiv_65']:
        print('Warning: vaccine_policy is not status_quo, laiv_5_12, laiv_5_18, iiv_50, or iiv_65.')

    vc12 = np.array([[0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## standard
                     [0, 0.15, 0.15, 0.  , 0.10, 0.20, 0.20, 0.55, 0.55],  ## pessimistic (same as standard)
                     [0, 0.15, 0.15, 0.  , 0.10, 0.20, 0.20, 0.55, 0.55],  ## middle
                     [0, 0.15, 0.15, 0.  , 0.10, 0.20, 0.20, 0.55, 0.55],  ## optimistic
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.55, 0.55, 0.55],  ## iiv 50 middle
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.80, 0.80, 0.80],  ## iiv 50 optimistic
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## iiv 65 middle
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.80, 0.80]])  ## iiv 65 optimistic

    vc18 = np.array([[0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## standard
                     [0, 0.15, 0.15, 0.  , 0.  , 0.20, 0.20, 0.55, 0.55],  ## pessimistic (same as standard)
                     [0, 0.15, 0.15, 0.  , 0.  , 0.20, 0.20, 0.55, 0.55],  ## middle
                     [0, 0.15, 0.15, 0.  , 0.  , 0.20, 0.20, 0.55, 0.55],  ## optimistic
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.55, 0.55, 0.55],  ## iiv 50 middle
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.80, 0.80, 0.80],  ## iiv 50 optimistic
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## iiv 65 middle
                     [0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.80, 0.80]])  ## iiv 65 optimistic

    vc_of_vaccine_policy = vc12
    if vaccine_policy.lower() == 'laiv_5_18':
        vc_of_vaccine_policy = vc18

    ve_infection_laiv = np.array([0.44 * np.ones(n_ages),  ## standard
                                  0.44 * np.ones(n_ages),  ## pessimistic
                                  0.56 * np.ones(n_ages),  ## middle
                                  0.75 * np.ones(n_ages)])  ## optimistic

    ve_gp_laiv = np.array([0.55 * np.ones(n_ages),  ## standard
                           0.55 * np.ones(n_ages),  ## pessimistic
                           0.65 * np.ones(n_ages),  ## middle
                           0.80 * np.ones(n_ages)]) ## optimistic

    ve_hosp_laiv = np.copy(ve_gp_laiv)  ## same as ve_gp

    ve_inf_main = 0.44 * np.ones(n_ages)
    ve_gp_main = 0.55 * np.ones(n_ages)
    ve_hosp_main = np.copy(ve_gp_main)

    my_policy_index = 0  ## status quo or iiv_50
    if vaccine_policy.lower() != 'status_quo':
        if vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'pessimistic':
            my_policy_index = 1
        elif vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'mid':
            my_policy_index = 2
        elif vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'optimistic':
            my_policy_index = 3
        elif vaccine_policy.lower() == 'iiv_50' and vaccine_scenario.lower() == 'mid':
            my_policy_index = 4
        elif vaccine_policy.lower() == 'iiv_50' and vaccine_scenario.lower() == 'optimistic':
            my_policy_index = 5
    if vaccine_policy.lower() == 'iiv_65' and vaccine_scenario.lower() == 'mid':
        my_policy_index = 6
    elif vaccine_policy.lower() == 'iiv_65' and vaccine_scenario.lower() == 'optimistic':
        my_policy_index = 7

    my_age_index = []  ## status quo or 'iiv'; used to edit the VE for LAIV policies
    if vaccine_policy.lower() != 'status_quo':
        if vaccine_policy.lower() == 'laiv_5_12':
            my_age_index = [3]  ## 5-11 YO
        elif vaccine_policy.lower() == 'laiv_5_18':
            my_age_index = [3, 4]  ## 5-17 YO

    for i in my_age_index:
        ve_inf_main[i] = ve_infection_laiv[my_policy_index, i]
        ve_gp_main[i] = ve_gp_laiv[my_policy_index, i]
        ve_hosp_main[i] = ve_hosp_laiv[my_policy_index, i]

    vaccine_coverage = vc_of_vaccine_policy[my_policy_index]

    if vaccine_policy.lower() == 'iiv_50' and vaccine_scenario.lower() == 'pessimistic':
        vaccine_coverage = np.array([0, 0.15, 0.15, 0.10, 0.10, 0.20, 0.30, 0.55, 0.55])

    return [vaccine_coverage, ve_inf_main, ve_gp_main, ve_hosp_main]


def load_VE_and_VC_LAIV(vaccine_policy='status_quo', vaccine_scenario='pessimistic'):
    ''' for each vaccine policy (status_quo, laiv_5_12, laiv_5_18, iiv_50) and
    each vaccine scenario (status_quo, pessimistic, middle, optimistic):
    loads the vaccine coverage and vaccine effectiveness.

    'status_quo' vaccine policy will give status quo VE and VC (regardless of vaccine scenario)
    '''

    n_ages = 9

    if vaccine_scenario.lower() not in ['status_quo', 'pessimistic', 'mid', 'optimistic']:
        print('Warning: vaccine_scenario is not status_quo, pessimistic, mid, or optimistic.')
    if vaccine_policy.lower() not in ['status_quo', 'laiv_5_12', 'laiv_5_18', 'iiv_50', 'iiv_65']:
        print('Warning: vaccine_policy is not status_quo, laiv_5_12, laiv_5_18, iiv_50, or iiv_65.')

    vc12 = np.zeros((6, n_ages))
    vc12[:, 3] = np.array([0, 0.30, 0.55, 0.80, 0, 0])  ## standard, low, middle, high

    vc18 = np.zeros((6, n_ages))
    vc18[:, 3] = np.array([0, 0.30, 0.55, 0.80, 0, 0])  ## 5-11: standard, low, middle, high
    vc18[:, 4] = np.array([0, 0.30, 0.55, 0.80, 0, 0])  ## 12-17: standard, low, middle, high

    vc_of_vaccine_policy = vc12
    if vaccine_policy.lower() == 'laiv_5_18':
        vc_of_vaccine_policy = vc18

    ve_infection_laiv = np.array([0.44 * np.ones(n_ages),  ## standard
                                  0.44 * np.ones(n_ages),  ## low
                                  0.56 * np.ones(n_ages),  ## middle
                                  0.75 * np.ones(n_ages)]) ## high

    ve_gp_laiv = np.array([0.55 * np.ones(n_ages),  ## standard
                           0.55 * np.ones(n_ages),  ## low
                           0.65 * np.ones(n_ages),  ## middle
                           0.80 * np.ones(n_ages)]) ## high

    ve_hosp_laiv = np.copy(ve_gp_laiv)  ## same as ve_gp

    ve_inf_main = 0.44 * np.ones(n_ages)
    ve_gp_main = 0.55 * np.ones(n_ages)
    ve_hosp_main = np.copy(ve_gp_main)

    my_policy_index = 0  ## status quo or iiv_50
    if vaccine_policy.lower() not in ['status_quo', 'iiv_50', 'iiv_65']:
        if vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'pessimistic':
            my_policy_index = 1
        elif vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'mid':
            my_policy_index = 2
        elif vaccine_policy.lower() != 'iiv_50' and vaccine_scenario.lower() == 'optimistic':
            my_policy_index = 3
        elif vaccine_policy.lower() == 'iiv_50' and vaccine_scenario == 'mid':
            my_policy_index = 4
        elif vaccine_policy.lower() == 'iiv_50' and vaccine_scenario == 'optimistic':
            my_policy_index = 5

    my_age_index = []  ## status quo
    if vaccine_policy.lower() != 'status_quo':
        if vaccine_policy.lower() == 'laiv_5_12':
            my_age_index = [3]  ## 5-11 YO
        elif vaccine_policy.lower() == 'laiv_5_18':
            my_age_index = [3, 4]  ## 5-17 YO

    for i in my_age_index:
        ve_inf_main[i] = ve_infection_laiv[my_policy_index, i]
        ve_gp_main[i] = ve_gp_laiv[my_policy_index, i]
        ve_hosp_main[i] = ve_hosp_laiv[my_policy_index, i]

    vaccine_coverage = vc_of_vaccine_policy[my_policy_index]

    return [vaccine_coverage, ve_inf_main, ve_gp_main, ve_hosp_main]

def read_parameters_sensitivity_analysis(p_healthseeking):
    """
       Load fitted parameter sets for the sensitivity analysis.

       For each parameter set, construct the initial population and
       age-specific health-seeking probabilities.

       ----------
       p_healthseeking : Baseline health-seeking probabilities for the nine age groups.

       Returns
       -------
       tuple
           Initial populations, health-seeking probabilities, R0 values,
           seasonal amplitudes, phase shifts, initial SW proportions,
           and infection seeds.
       """

    df = pd.read_csv('data/optimization_results_p_hs_scale_wo_a_ps_365.csv')

    n_sims = len(df)
    n_ages = len(populations_by_fractions)

    # Fitted parameters
    my_r0s = df["R₀"].to_numpy()
    my_amplitudes = df["amplitude"].to_numpy()
    my_phaseshifts = df["phase-shift"].to_numpy()
    my_sws = df["SW_vals"].to_numpy()
    my_is = df["seed"].to_numpy()

    # Health-seeking scaling factors
    tau_child = df["τ_child"].to_numpy()
    tau_adult = df["τ_adult"].to_numpy()

    # Initialise arrays
    my_inits = np.zeros((n_sims, (n_states + 3) * n_ages))
    my_p_healthseekings = np.zeros((n_sims, n_ages))

    for i in range(n_sims):
        # Initial population
        init = np.zeros((n_states + 3, n_ages))

        init[SU, :] = (1 - my_sws[i]) * populations_by_fractions
        init[IU, 5] = my_is[i]
        init[SU, 5] -= my_is[i]
        init[CaseIU, :] = init[IU, :]
        init[SW, :] = my_sws[i] * populations_by_fractions

        my_inits[i] = init.flatten()

        # Age-specific health-seeking probabilities
        p_hs = np.array(p_healthseeking, copy=True)

        p_hs[:5] *= tau_child[i]
        p_hs[7:9] *= tau_adult[i]

        my_p_healthseekings[i] = p_hs

    return (my_inits, my_p_healthseekings, my_r0s, my_amplitudes, my_phaseshifts, my_sws, my_is)



# Default model parameter values

## in run_model, R0 and amplitude will be arguments
n_ages = len(age_names)
n_states = len(state_names)
N = 1_000_000
populations_by_fractions = np.array(load_population_australia()['proportion'])
contact_matrix = load_contact_matrix()

## Disease natural history parameters (rates per day)
sigma = 1 / 1  ## mean latent period
gamma = 1 / 2.3  ## mean infectious period
omega = 0       ## no waning immunity

## Fitted parameter values
## These parameters will be loaded separately by read_parameters_sensitivity_analysis() to make up 40 trajectories
R0 = 1.178539
amplitude = 1.528736e-07
phase_shift = 167.637059
baseline_healthseeking = np.array([0.093087, 0.093087, 0.093087, 0.091192, 0.091192,
                                   0.101058, 0.101058, 0.111099 , 0.111099])
# initial_SW = 0.15

## Averaged from AIHW for 2018-19, 2022-23, 2023-24
hosp_rates_per_100K = np.array([284.0265, 284.0265, 182.879, 93.70589, 38.60522, 37.92387,
                                60.96131, 113.3298, 281.8181 ])

## Averaged from 2023, 2024 and 2025
death_rates_per_100K = np.array([0.26269098, 0.26269098, 0.52211831, 0.23340239, 0.11667138,
                                 0.36192057, 2.06067362, 6.54977444, 41.83325878])

p_asymptomatic = 0.19 * np.ones(n_ages)
p_hospitalised = hosp_rates_per_100K / 100_000
p_deaths = death_rates_per_100K / 100_000
reduced_sus = load_reduced_susceptibility()

## Vaccine parameters
# (_, VE_SUS, VE_GP, VE_HOSP) = load_VE_and_VC_IIV(vaccine_policy = 'status_quo', vaccine_scenario = 'mid')

## Simulation period: 52 weeks (364 days)
total_weeks = 52
n_days = total_weeks * 7

## Days 0 to 364, inclusive
tspan = np.arange(n_days + 1)
