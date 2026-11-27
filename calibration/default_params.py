## Fixed parameters for the ACEFA model
import numpy as np
import pandas as pd
from datetime import date, timedelta

[SU, E1U, E2U, IU, RU, SV, E1V, E2V, IV, RV, SW, E1W, E2W, IW, RW, CaseIU, CaseIV, CaseIW] = range(18)
state_names = ['SU', 'E1U', 'E2U', 'IU', 'RU', 'SV', 'E1V', 'E2V', 'IV', 'RV', 'SW', 'E1W', 'E2W', 'IW', 'RW']
age_names = ['0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '80+']
acefa_ages = ['0-1', '1-4', '5-11', '12-17', '18-64', '65-79', '>80']

n_ages_acefa = len(acefa_ages)

def load_population_australia(sheet_name='AUS'):
    ''' ABS: population clock and pyramid 2024 (by yearly age)
    https://www.abs.gov.au/statistics/people/population/population-clock-pyramid


    Population pyramid for Australia 2024 https://www.populationpyramid.net/australia/2024/
    returns dataframe of Australian population by age groups and percentages
    '''

    ## sheet_name can be: AUS, NSW, VIC, QLD, SA, WA, NT, TAS, ACT
    df = pd.read_excel('data/population_australia_2024_abs_australia.xlsx',  sheet_name=str(sheet_name.upper()))
    # print( df.head() )

    p = df.copy()
    total_population = np.sum(p["Male"]) + np.sum(p["Female"])

    ## add a column
    p.insert(3, "Percent", (p["Male"] + p["Female"]) / total_population, True)

    ## Create age groups that match the ACEFA model
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

    data = {
        "age": ['0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '79+'],
        "proportion": C
    }

    df2 = pd.DataFrame(data)

    return df2

def load_contact_matrix(my_population, n_ages = 9):
    ''' my_population is the population percentages by age group.
    contact matrix from POLYMOD study '''

    ## from Prem et al.
    # filename = 'data/contact_matrices_australia.csv'
    # df = pd.read_csv(filename).to_numpy()  ## 16 x 16
    # C = np.ones((n_ages, n_ages))

    ## from POLYMOD (with 9 age groups: '0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '79+')
    if(n_ages == 9):
        C = pd.read_csv('data/ACEFA_UK_POLYMOD_contact_matrix.csv', header=None).to_numpy()

    else:
        if(n_ages == 16):
            C = pd.read_csv('data/ACEFA_UK_POLYMOD_contact_matrix_5_year.csv', header=None).to_numpy()

    return C

def load_mortality_rates():
    ''' Average annual influenza-associated excess mortality in Australia by age group, 2007–2015
    (for all influenza types) rates per 100 000 persons (Leung 2022)'''

    ## Age-specific rates as published in Leung 2022
    C = np.array([2.31,   # 0 to 5 months
                  0.3,    # 6 months to 4Y
                  0.11,   # 5 to 14Y
                  0.23,   # 15 to 49Y
                  1.27,   # 50 to 64Y
                  5.31,   # 65 to 74Y
                  35.11   # 75Y +
    ])

    ## Age-specific rates for our model's chosen age groups
    C1 = np.array([C[0],  # 0 to 5 months
                  C[1],   # 6 to 11 months
                  C[1],   # 1 to 4Y
                  C[2],  # 5 to 11Y
                  C[2] * (3 / 6) + C[3] * (3 / 6),  # 12 to 17Y
                  C[3],  # 18 to 49Y
                  C[4],  # 50 to 64Y
                  C[5] * (10 / 15) + C[6] * (5 / 15),  # 65 to 79Y
                  C[6]  # 79Y +
                  ])

    return C1

def load_hospitalisation_rates():
    ''' Average annual influenza-associated excess hospitalizations in Australia by age group, 2007–2015
    (for all influenza types) rates per 100 000 persons (Table 1, Leung 2022)'''
    # C = np.array([164.02,   # 0 to 5 months  ## LEUNG 2022
    #               48.88,    # 6 months to 4Y
    #               24.58,   # 5 to 14Y
    #               38.08,   # 15 to 49Y
    #               59.06,   # 50 to 64Y
    #               134.81,   # 65 to 74Y
    #               302.95   # 75Y +
    #               ])

    C = np.array([187.8,  # 0 to 4Y  ## Muscatello2017, Table 1
                  106.9,  # 5 to 14Y
                  80.7,  # 15 to 49Y
                  95,  # 50 to 64Y
                  178.6,  # 65 to 84Y
                  208.6  # 85Y +
                  ])

    ## Age-specific rates for our model's chosen age groups
    C1 = np.array([C[0],  # 0 to 5 months
                  C[0],   # 6 to 11 months
                  C[0],   # 1 to 4Y
                  C[1],  # 5 to 11Y
                  C[1] * (3 / 6) + C[2] * (3 / 6),  # 12 to 17Y
                  C[2],  # 18 to 49Y
                  C[3],  # 50 to 64Y
                  C[4] ,  # 65 to 79Y
                  C[5]  # 79Y +
                  ])

    return C1

def load_reduced_susceptibility():
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

def load_VE_and_VC(vaccine_policy='status_quo', vaccine_scenario='pessimistic'):
    ''' for each vaccine policy (status_quo, laiv_5_12, laiv_5_18) and
    each vaccine scenario (status_quo, pessimistic, mid, optimistic):
    loads the vaccine coverage and vaccine effectiveness.

    'status_quo' vaccine policy will give status quo VE and VC (regardless of vaccine scenario)
    '''

    n_ages = 9

    if vaccine_scenario not in ['status_quo', 'pessimistic', 'mid', 'optimistic']:
        print('Warning: vaccine_scenario is not status_quo, pessimistic, mid, or optimistic.')
    if vaccine_policy not in ['status_quo', 'laiv_5_12', 'laiv_5_18']:
        print('Warning: vaccine_policy is not status_quo, laiv_5_12, or laiv_5_18.')

    #### Vaccine coverage by vaccine scenario: status quo, pessimistic, mid, optimistic
    ## LAIV 5 to 12:
    vc12 = np.array([[0.15, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## standard
                     [0.15, 0.15, 0.15, 0.40, 0.10, 0.20, 0.20, 0.55, 0.55], ## low (same as standard)
                     [0.15, 0.15, 0.15, 0.60, 0.10, 0.20, 0.20, 0.55, 0.55],  ## middle
                     [0.15, 0.15, 0.15, 0.80, 0.10, 0.20, 0.20, 0.55, 0.55]])  ## high

    ## LAIV 5 to 18:
    vc18 = np.array([[0.15, 0.15, 0.15, 0.10, 0.10, 0.20, 0.20, 0.55, 0.55],  ## standard
                     [0.15, 0.15, 0.15, 0.40, 0.40, 0.20, 0.20, 0.55, 0.55],  ## low (same as standard)
                     [0.15, 0.15, 0.15, 0.60, 0.60, 0.20, 0.20, 0.55, 0.55],  ## middle
                     [0.15, 0.15, 0.15, 0.80, 0.80, 0.20, 0.20, 0.55, 0.55]])  ## high

    vc_of_vaccine_policy = vc12
    if vaccine_policy.lower() == 'laiv_5_18':
        vc_of_vaccine_policy = vc18

    #### Vaccine effectiveness by vaccine scenario: status quo, pessimistic, mid, optimistic
    ## laiv
    ve_infection_laiv = np.array([0.44 * np.ones(n_ages),  ## standard
                                  0.44 * np.ones(n_ages),  ## low
                                  0.56 * np.ones(n_ages),  ## middle
                                  0.75 * np.ones(n_ages)]) ## high

    ve_gp_laiv = np.array([0.55 * np.ones(n_ages),  ## standard
                           0.55 * np.ones(n_ages),  ## low
                           0.65 * np.ones(n_ages),  ## middle
                           0.80 * np.ones(n_ages)]) ## high

    ve_hosp_laiv = np.copy(ve_gp_laiv)  ## same as ve_gp

    ## Vaccine effectiveness by the chosen vaccine policy and vaccine scenario: initiate with status quo scenario
    ve_inf_main = 0.44 * np.ones(n_ages)
    ve_gp_main = 0.55 * np.ones(n_ages)
    ve_hosp_main = np.copy(ve_gp_main)

    ## Assign an index by policy and age so we can update the ve_[inf/gp/hosp]_main accordingly by vaccine_policy
    my_policy_index = 0  ## status quo
    if vaccine_policy.lower() != 'status_quo':
        if vaccine_scenario.lower() == 'pessimistic':
            my_policy_index = 1
        elif vaccine_scenario.lower() == 'mid':
            my_policy_index = 2
        elif vaccine_scenario.lower() == 'optimistic':
            my_policy_index = 3

    my_age_index = []  ## status quo
    if vaccine_policy.lower() != 'status_quo':
        if vaccine_policy.lower() == 'laiv_5_12':
            my_age_index = [3]  ## 5-11 YO
        elif vaccine_policy.lower() == 'laiv_5_18':
            my_age_index = [3, 4]  ## 5-17 YO

    ## update the main VE arrays to policy program
    for i in my_age_index:
        ve_inf_main[i] = ve_infection_laiv[my_policy_index, i]
        ve_gp_main[i] = ve_gp_laiv[my_policy_index, i]
        ve_hosp_main[i] = ve_hosp_laiv[my_policy_index, i]

    vaccine_coverage = vc_of_vaccine_policy[my_policy_index]

    return [vaccine_coverage, ve_inf_main, ve_gp_main, ve_hosp_main]

## Define vaccine policy: status_quo, laiv_5_12, laiv_5_18
## Define vaccine scenario: status_quo, pessimistic, mid, optimistic
vaccine_policy = 'status_quo'
vaccine_scenario = 'mid'

## in run_model, R0 and amplitude will be arguments

n_ages = len(age_names)
n_states = len(state_names)
#N = 1_000_000
# amplitude = 0.3
populations_by_fractions = np.array(load_population_australia()['proportion'])
contact_matrix = load_contact_matrix(populations_by_fractions)
sigma = 1 / 1  ## latent period (Carrat2008)
gamma = 1 / 2.3  ## infectious period (Carrat2008)
omega = 0

## fitted values from Wed 3 Sept, 4:47pm
#initial_SW = 0.15
#R0 = 1.258793
#R0_fitted = 1.258793
#amplitude = 0.061613
#phase_shift = 121.930872
#p_healthseeking = np.array([0.13440617, 0.13440617, 0.13440617, 0.13167023, 0.13167023,  ## mean from flutracking data
#                            0.14591527, 0.14591527, 0.16041393, 0.16041393])

## Epidemiological parameters
p_asymptomatic = 0.19 * np.ones(n_ages) # Furuya-Kanamori2016
p_hospitalised = load_hospitalisation_rates() / 100_000
reduced_sus = load_reduced_susceptibility() ## Hill2009

#### Vaccine parameters
(vaccine_coverage, VE_INF0, VE_GP0, VE_HOSP0) = load_VE_and_VC(vaccine_policy = vaccine_policy, vaccine_scenario = vaccine_scenario)

tspan = np.linspace(0, 371 * 1, 371 * 1 + 1)  ## t0 = 7 March
total_weeks = int(tspan[-1] / 7)

years_all = range(2015, 2026)
N_all = []
for year in years_all:
    N = pd.read_excel('data/population_australia_' + str(year) + '_abs_australia.xlsx', sheet_name='AUS')
    N = np.sum(N["Male"]) + np.sum(N["Female"])
    N_all.append(N)

N_all = pd.DataFrame({'year' : years_all, 'N' : N_all})

N = pd.read_excel('data/population_australia_2024_abs_australia.xlsx', sheet_name='AUS')
N = np.sum(N["Male"]) + np.sum(N["Female"])

#######################
##### vaccination #####
#######################

scenario_hub_folder = 'data/'

x = np.zeros(24)
for sheet_name in range(2021, 2025):
    # print(sheet_name)
    df = pd.read_excel(scenario_hub_folder + 'vaccinations/AUS - All people - influenza vaccine uptake by age group and year - as at 010625.xlsx',
                       sheet_name=str(sheet_name))
    # df.columns = ['week_ending', '0to5', '5to15', '15to50', '50to65', '65+']

    ## drop the old labels row
    nf = df.drop(df.index[0]).to_numpy()
    my_dates = nf[:, 0]
    nf = nf[:, 1:].astype(float)  ## nf.shape = (vac_weeks, 12, n_ages=5)

    mysum = nf[-1, :].sum()
    x = (x + nf.sum(1) / mysum)

x = x / 4

# weekly vaccination progress for the calibration
# 2 weeks added to account for delay in immmunity

cum_weekly_vacc_progress = np.append(np.zeros(9 + 2),x)

start_date = date(2025, 3, 7)
early_dates = []
for i in range(9 + 2):
    early_dates.append(start_date - timedelta(days=(i+1)*7))
    early_dates[i] = early_dates[i].strftime('%b-%d')

vacc_dates = np.append(early_dates[::-1],my_dates)

default_parameters = [n_ages, n_states, N, populations_by_fractions, contact_matrix, sigma, gamma, omega, p_asymptomatic, p_hospitalised, reduced_sus, tspan, n_ages_acefa, N_all, cum_weekly_vacc_progress, vacc_dates]

status_quo_vac_parameters = [vaccine_coverage, VE_INF0, VE_GP0, VE_HOSP0]

# parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
# parameters_outcome = [total_weeks, N, VE_SYMP, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]
