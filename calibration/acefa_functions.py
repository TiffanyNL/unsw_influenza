## functions for ACEFA scenario modelling
from default_params import *
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# functions to run the model
[SU, E1U, E2U, IU, RU, SV, E1V, E2V, IV, RV, SW, E1W, E2W, IW, RW, CaseIU, CaseIV, CaseIW] = range(18)

def get_parameters_by_names(parameters, names):
    if len(names) == 1:
        return parameters[names[0]]
    return (parameters[name] for name in names)

def get_weekly_vaccination_rates():
    ''' reads the weekly vaccination rates data for 2021, 2022, 2023, and 2024 and calculates the average weekly
     speed of vaccinating those who decide to get vaccinated over 24 weeks (Mar 7 to Aug 31).
     OUTPUT is an array (length 24) of cumulative %age completion of the overall vaccination'''

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

    return x, my_dates

def get_weekly_vaccination_rates_LAIV():
    ''' LAIV program over 6 weeks (Mar 1 to Apr 11, say) in school term 1
     OUTPUT is an array (length 24) of cumulative %age completion of the overall vaccination'''

    scenario_hub_folder = '/Users/z3546535/PycharmProjects/ACEFA/projects/proj-5050_acefa_core_data-1128.4.951/scenario-hub-raw-data/'

    x = np.zeros(24)
    df = pd.read_excel(scenario_hub_folder + 'vaccinations/influenza_vaccine_uptake_all.xlsx', sheet_name=str(2024))
    # df.columns = ['week_ending', '0to5', '5to15', '15to50', '50to65', '65+']

    ## drop the old labels row
    nf = df.drop(df.index[0]).to_numpy()
    my_dates = nf[:, 0]

    # my_dates[8:16]
    temp = np.linspace(0, 1, 7)
    x[0:6] = temp[1:]
    x[6:] = 1

    # for d, y in zip(my_dates, x):
    #     print(d, y)

    return x, my_dates

def plot_weekly_vaccination_rates():
    scenario_hub_folder = 'data/'
    my_color_shades = sns.color_palette("rocket", 5)
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    x = np.zeros(24)
    plt.figure(1, figsize=(6, 3), layout='tight')
    plt.title('Vaccination uptake, normalised')
    for sheet_name in range(2020, 2025):
        df = pd.read_excel(scenario_hub_folder + 'vaccinations/AUS - All people - influenza vaccine uptake by age group and year - as at 010625.xlsx',
                           sheet_name=str(sheet_name))
        df.columns = ['week_ending', '0to5', '5to15', '15to50', '50to65', '65+']

        ## drop the old labels row
        nf = df.drop(df.index[0]).to_numpy()

        my_dates = nf[:, 0]
        nf = np.delete(nf, 0, axis=1)  ## nf.shape = (vac_weeks, 12, n_ages=5)

        ## plot with summed age groups (overall vaccination rate)
        mysum = nf[-1, :].sum()
        x = (x + nf.sum(1) / mysum).astype(float)
        plt.plot(nf.sum(1) / mysum, marker='o', markersize=3, linewidth=1, label=sheet_name)
        plt.xticks(range(nf.shape[0]), my_dates, rotation=45)
        plt.ylabel('Vacc rate')
    # plt.plot(x / len(range(2020, 2025)), 'k', linewidth=2, label='avg')
    plt.axhline(y=0.5, xmin=0, xmax=nf.shape[0], color='k', linewidth=0.5)
    for month_end in [3, 7, 11, 15, 19]:
        plt.axvline(x = month_end, color='k', linewidth=0.5, alpha=0.2)
    plt.legend(loc='upper left')
    # plt.show(block=True)

    my_dates_new = np.copy(my_dates)
    for i in range(my_dates.shape[0]):
        my_dates_new[i] = my_dates[i][:3] + '\n' + my_dates[i][-2:]

    # plt.figure(2, figsize=(13, 10), layout='tight')
    plt.figure(2, figsize=(7, 10), layout='tight')
    for sval, sheet_name in enumerate(range(2020, 2025)):
        df = pd.read_excel(scenario_hub_folder + 'vaccinations/AUS - All people - influenza vaccine uptake by age group and year - as at 010625.xlsx',
                           sheet_name=str(sheet_name))
        df.columns = ['week_ending', '0to5', '5to15', '15to50', '50to65', '65+']

        ## drop the old labels row
        nf = df.drop(df.index[0]).to_numpy()

        my_dates = nf[:, 0]
        nf = np.delete(nf, 0, axis=1)  ## nf.shape = (vac_weeks, 12, n_ages=5)

        ## plot by age groups
        plt.subplot(5, 1, sval + 1)
        plt.title(str(sheet_name))
        for a, my_age_group in enumerate(df.columns[1:]):
            # print(a, my_age_group)
            plt.plot(nf[:, a] / nf[-1, a], marker='o', markersize=3, color=my_color_shades[a], label=my_age_group)
        plt.axhline(y=0.5, xmin=0, xmax=nf.shape[0], color='k', linestyle='--', linewidth=0.5)
        plt.xticks(range(nf.shape[0]), '')
        if sheet_name > 2023: #1:
            plt.xticks(range(nf.shape[0]), my_dates_new, rotation=0, fontsize=8)
            plt.xlabel('Week ending in:')
        for month_end in [3, 7, 11, 15, 19]:
            plt.axvline(x=month_end, color='k', linewidth=0.5, alpha=0.2)
        plt.ylabel('Cumulative coverage')
        plt.grid(color='k', linewidth=0.5, alpha=0.2, axis='y')
    plt.subplot(5, 1, 1)
    plt.legend(loc='upper left', fontsize=8)
    # plt.show(block=True)

    plt.figure(3, figsize=(14, 3), layout='tight')
    for sval, sheet_name in enumerate(range(2020, 2025)):
        df = pd.read_excel(scenario_hub_folder + 'vaccinations/AUS - All people - influenza vaccine uptake by age group and year - as at 010625.xlsx',
                           sheet_name=str(sheet_name))
        df.columns = ['week_ending', '0to5', '5to15', '15to50', '50to65', '65+']

        ## drop the old labels row
        nf = df.drop(df.index[0]).to_numpy()

        my_dates = nf[:, 0]
        nf = np.delete(nf, 0, axis=1)  ## nf.shape = (vac_weeks, 12, n_ages=5)

        ## plot by age groups
        plt.subplot(1, 5, sval + 1)
        plt.title(str(sheet_name))
        for a, my_age_group in enumerate(df.columns[1:]):
            # print(a, my_age_group)
            plt.plot(nf[:, a], marker='o', markersize=3, color=my_color_shades[a], label=my_age_group)
        # plt.axhline(y=0.5, xmin=0, xmax=nf.shape[0], color='k', linestyle='--', linewidth=0.5)
        plt.xticks(range(nf.shape[0]), '')
        if sheet_name > 2019:
            plt.xticks(range(nf.shape[0]), my_dates, rotation=90, fontsize=8)
            plt.xlabel('Week ending in:')
        if sval > 0:
            plt.yticks([0, 0.2, 0.4, 0.6], '')
        # for month_end in [3, 7, 11, 15, 19]:
        #     plt.axvline(x=month_end, color='k', linewidth=0.5, alpha=0.2)
        plt.ylim([-0.02, 0.7])
        # plt.ylabel('Vacc rate')
        plt.grid(color='k', linewidth=0.5, alpha=0.2, axis='y')
    # plt.subplot(5, 1, 1)
    plt.legend(loc='upper left', fontsize=8)
    plt.show(block=True)

    return []

def findBeta(R0, C, gamma, mu, totalPop):
    # compute the eigenvalues of -F*V^(-1)
    [n1, n2] = np.shape(C)

    # create F (transmission matrix) and V (transition matrix)
    F = np.zeros((n1, n1))
    V = np.diag(-(gamma + mu) * np.ones(n1))
    N = np.sum(totalPop)

    for jvals in range(n1):
        F[jvals, :] = (C[jvals, :] / N) * totalPop[jvals]

    FV = np.dot(-F, np.linalg.inv(V))

    myEig = np.linalg.eig(FV)
    largestEig = np.max(myEig[0])

    ## double check that the calculated beta gives the right R0:
    # Fnew = beta * F
    # FVnew = np.dot(-Fnew, np.linalg.inv(V))
    # myEignew = np.linalg.eig(FVnew)
    # print('R0:', np.max(myEignew[0]))

    assert largestEig.imag == 0.0, 'largest eigenvalue is not real'
    beta = R0 / largestEig.real
    return beta

def model_equations_seir(initial_population, t, parameters):
    '''age-structured ACEFA model for a single population with seasonality
    SEIR model for unvaccinated, vaccinated, and pre-existing immunity "lines"
    '''

    ## Ensure my population is non-negative
    initial_population_pos = np.maximum(initial_population, 0)

    [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus] = parameters

    population_states = np.reshape(initial_population_pos, (n_states + 3, n_ages))

    SU, E1U, E2U, IU, RU = population_states[0:5]  ## unvaccinated
    SV, E1V, E2V, IV, RV = population_states[5:10]  ## vaccinated
    SW, E1W, E2W, IW, RW = population_states[10:15]  ## preexisting immunity

    # # force of infection (t + 60) brings peak to August
    FOI = (1 - amplitude * np.cos(2 * np.pi * (t + phase_shift) / 371)) * np.dot(contact_matrix, beta * (IU + IV + IW))

    #
    dSU = - np.multiply(FOI, SU)
    dE1U = np.multiply(FOI, SU) - (2 * sigma) * E1U
    dE2U = (2 * sigma) * E1U - (2 * sigma) * E2U
    dIU = (2 * sigma) * E2U - gamma * IU
    dRU = gamma * IU - omega * RU
    #
    dSV = - np.multiply( (1 - VE_SUS) * FOI, SV) - omega * SV
    dE1V = np.multiply( (1 - VE_SUS) * FOI, SV) - (2 * sigma) * E1V
    dE2V = (2 * sigma) * E1V - (2 * sigma) * E2V
    dIV = (2 * sigma) * E2V - gamma * IV
    dRV = gamma * IV - omega * RV
    #
    dSW = omega * SV + omega * (RU + RV + RW) - np.multiply((1 - reduced_sus) * FOI, SW)
    dE1W = np.multiply((1 - reduced_sus) * FOI, SW) - (2 * sigma) * E1W
    dE2W = (2 * sigma) * E1W - (2 * sigma) * E2W
    dIW = (2 * sigma) * E2W - gamma * IW
    dRW = gamma * IW - omega * RW
    #
    ## record recoveries for weekly recordables
    count_IU = gamma * IU
    count_IV = gamma * IV
    count_IW = gamma * IW
    #
    dydt = np.array([dSU, dE1U, dE2U, dIU, dRU,
                     dSV, dE1V, dE2V, dIV, dRV,
                     dSW, dE1W, dE2W, dIW, dRW,
                     count_IU, count_IV, count_IW]).reshape(( (n_states + 3) * n_ages ))

    return dydt

def load_vaccine_coverages():
    ''' for the 3 policy options (status quo, LAIV distributed to 5 to 12Y, and LAIV distributed to 5 to 18Y),
     and 3 vaccine uptakes (status quo, pessimistic, mid, and optimistic), vaccine coverage by age group '''
    n_policyoptions = 3
    n_ages = 9
    vc_statusquo = np.array([0.15, 0.15, 0.15, 0.10, 0.10,  ## 0-5M, 6-11M, 1-4, 5-11, 12-17,
                             0.20, 0.20, 0.55, 0.55])  ## 18-49, 50-64, 65-79, 79+

    vc_pessimistic = np.array([vc_statusquo, vc_statusquo, vc_statusquo])
    vc_pessimistic[1:, 3] = 0.40  ## LAIV 5-12 and LAIV 5-18, age group 5-11
    vc_pessimistic[2:, 4] = 0.40  ## LAIV 5-18, age group 12-17
    #
    vc_mid, vc_optimistic = np.copy(vc_pessimistic), np.copy(vc_pessimistic)
    vc_mid[1:, 3] = 0.60  ## LAIV 5-12 and LAIV 5-18, age group 5-11
    vc_mid[2:, 4] = 0.60  ## LAIV 5-18, age group 12-17
    #
    vc_optimistic[1:, 3] = 0.80  ## LAIV 5-12 and LAIV 5-18, age group 5-11
    vc_optimistic[2:, 4] = 0.80  ## LAIV 5-18, age group 12-17

    return [vc_pessimistic, vc_mid, vc_optimistic]

def run_model_equations(initial_SW, amplitude = 0.3, R0 = 1.5, vaccine_uptake = 'status_quo', vaccine_policy='status_quo'):
    ''' runs the ACEFA model for a specified amplitude, R0, and vaccine uptake (which is further divided into 3
    vaccine policy)

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns a daily time series of (n_states + 3) and n_ages
    '''

    tspan = np.linspace(0, 364 * 1, 364 * 1 + 1)
    total_weeks = int(tspan[-1] / 7)

    ## fixed parameters
    n_ages = 9
    n_states = 15
    N = 1_000_000
    # amplitude = 0.3
    phase_shift = 120  # to be changed after we look at data
    populations_by_fractions = np.array(load_population_australia()['proportion'])
    contact_matrix = load_contact_matrix(populations_by_fractions)
    sigma = 1 / 1.2  ## latent period (Longini2005)
    gamma = 1 / 4.1  ## infectious period (Longini2005)
    omega = 1 / 120  ## duration of immunity (Tenforde2024b)

    ## varied parameters
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    ## Influenza epidemiological parameters (age-structured)
    p_asymptomatic = 0.19 * np.ones(n_ages)  # Furuya-Kanamori2016
    p_healthseeking = np.array([0.296, 0.296, 0.296, 0.296, 0.296, 0.208, 0.219, 0.277, 0.277])  ##Baltrusaitis2022; 0.25 overall (Cheng2024)
    p_hospitalised = load_hospitalisation_rates() / 100_000
    reduced_sus = 0.3  ## relative protection for "waning" classes

    VE_SUS_TIV = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.37, 0.37, 0.18, 0.18])  ## VE against any influenza (Frutos2025)
    VE_SUS_LAIV = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4])  ## against matched strains, Mohn2018
    VE_HOSP = np.array([0., 0.71, 0.71, 0.49, 0.49, 0.51, 0.40, 0.36, 0.36])  ## TIV, Tenforde2024b
    VE_GP = np.array([0.32, 0.32, 0.32, 0.32, 0.32, 0.36, 0.36, 0.36, 0.36])  ## against medically attended influenza (Frutos2025)

    VE_SUS = np.copy(VE_SUS_TIV)
    VE_SUS[:2] = VE_SUS_LAIV[0:2]  ## VE_SUS for LAIV for 5-12YO, and for seasonal vaccine for other ages

    ## run the model
    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]

    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / 1_000_000
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    #### Set vaccine coverage by vaccine uptake and vaccine policy
    ## vaccine_uptake: 'pessimistic' OR 'mid' OR 'optimistic'
    ## vaccine_policy: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'
    vc_pessimistic, vc_mid, vc_optimistic = load_vaccine_coverages()

    ## Set vaccine uptake and vaccine policy to pessimistic, status quo
    my_vaccine_uptake = vc_pessimistic
    vaccine_policy_index = 0

    ## if the vaccine uptake is not pessimistic, and if vaccine_policy is not status quo:
    if vaccine_uptake.lower() == 'mid':
        my_vaccine_uptake = vc_mid
    elif vaccine_uptake.lower() == 'optimistic':
        my_vaccine_uptake = vc_optimistic

    if vaccine_policy.lower() == 'laiv_5_12':
        vaccine_policy_index = 1
    elif vaccine_policy.lower() == 'laiv_5_18':
        vaccine_policy_index = 2


    ## Manually move individuals by vaccine coverage of vaccine uptake and vaccine policy
    my_vaccine_coverage = my_vaccine_uptake[vaccine_policy_index]

    ## Initial run with burn-in period
    tspan_transient = np.linspace(0, 364 * 5, 364 * 5 + 1)
    out = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan_transient,
                 args=(parameters_ode,))
    initial_population_endemic = out[-1].reshape((n_states + 3, n_ages))

    # set initial condition with vaccination
    init_new = np.copy(initial_population_endemic)
    for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
        frac_to_vaccinate = my_vaccine_coverage * initial_population_endemic[old_state]
        init_new[old_state] = initial_population_endemic[old_state] - frac_to_vaccinate
        init_new[new_state] += frac_to_vaccinate
    print('%1f Total Population--Unvaccinated: %.2f, Vaccinated: %.2f, Waning: %.2f' %
          (init_new[:15, :].sum(), init_new[0:5, :].sum(), init_new[5:10, :].sum(), init_new[10:15, :].sum()) )

    # run the model: out.shape = n_days, (n_states+3) * n_ages
    out = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan, args=(parameters_ode,))

    return out

def run_model_equations_weekly_vaccination(default_parameters, status_quo_vac_parameters, initial_SW=np.zeros(n_ages), R0 = 1.5, amplitude = 0, phase_shift = 0, seed = 1):
    ''' runs the ACEFA model for a specified amplitude, R0, and vaccine uptake (which is further divided into 3
    vaccine policy)

    This function differs from run_model_equations in the following ways:
    - Vaccination runs for 24 weeks starting from March.

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns a daily time series of (n_states + 3) and n_ages
    '''
    # default parameters
    [n_ages, n_states, N, populations_by_fractions, contact_matrix, sigma, gamma, omega, p_asymptomatic, p_hospitalised, reduced_sus, tspan, n_ages_acefa, N_all, cum_weekly_vacc_progress, vacc_dates] = default_parameters

    [vc, VE_INF0, VE_GP0, VE_HOSP0] = status_quo_vac_parameters

    ## varied parameters
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = seed # changed to prevalence
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    #### Set vaccine coverage by vaccine uptake and vaccine policy
    ## vaccine_uptake: 'pessimistic' OR 'mid' OR 'optimistic'
    ## vaccine_policy: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    ## policy: status quo
    #vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_INF0, reduced_sus]
    # parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## Set my weekly vaccination schedule
    # total_weeks = 52

    #cum_weekly_vacc_progress, vacc_dates = get_weekly_vaccination_rates()
    vacc_weeks = len(cum_weekly_vacc_progress)

    cum_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks))

    for vacc_week in range(vacc_weeks):
        for a in range(n_ages):
            cum_weekly_vaccination_schedule[a, 1:] = cum_weekly_vacc_progress * vc[a]
            new_weekly_vaccination_schedule[a] = cum_weekly_vaccination_schedule[a][1:] - \
                                                 cum_weekly_vaccination_schedule[a][:-1]

    initial_population_endemic = np.copy(initial_population)

    out_main = initial_population_endemic.reshape((n_states + 3) * n_ages)  ## to be vstacked

    # set initial condition with vaccination
    old_init = out_main.reshape((n_states + 3, n_ages))
    my_t0 = 0  ## time counter
    # total_doses_frac = np.zeros(n_ages)
    for vacc_week in range(vacc_weeks):
        vc = new_weekly_vaccination_schedule[:, vacc_week]
        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            frac_to_vaccinate = vc * old_init[old_state]
            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
            # total_doses_frac += frac_to_vaccinate
        ## run simulation
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                           args=(parameters_ode,), full_output=True)

        ## save my time series
        out_main = np.vstack((out_main, out[1:]))

        ## update initial condition and time span
        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7
    # total_doses_sq = N * total_doses_frac

    ## run to week 52 (end of year)
    out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out[1:]))

    return out_main

## LAIV and seasonal vaccination campaign both run over March to August over 24 weeks
def run_model_equations_weekly_vaccination_3_laiv_together(initial_SW=np.zeros(n_ages), amplitude = 0.3, R0 = 1.5, vaccine_uptake = 'pessimistic', vaccine_policy='status_quo',contact_matrix = np.ones((n_ages, n_ages))):
    ''' vaccination campaign starts in week 10 (corresponding to 7 march)
    this change is implemented in response to Isaac's latest calibration for Wed 3 Sep 2025
    runs the ACEFA model for a specified amplitude, R0, and vaccine uptake (which is further divided into 3
    vaccine policy)

    This function differs from run_model_equations in the following ways:
    - Vaccination runs for 24 weeks starting from March.
    - t0 is March 1 here (i.e. phase_shift = 60 here)

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns a daily time series of (n_states + 3) and n_ages'''

    ## varied parameters
    # contact_matrix = load_contact_matrix(populations_by_fractions)
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)
    # vaccine_uptake = 'optimistic'
    # vaccine_policy = 'status_quo' # 'laiv_5_18'

    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / 27_586_007  ### I CHANGED THIS!!!
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    #### Set vaccine coverage by vaccine uptake and vaccine policy
    ## vaccine_uptake: 'pessimistic' OR 'mid' OR 'optimistic'
    ## vaccine_policy: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    ## policy: status quo
    vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    # vc_laiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_LAIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)
    # vc_iiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_IIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    # parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## Run model without vaccination for 10 weeks, then weeks 11 to 35 is the campaign, and weeks 36-52 is no vaccination again
    out_main = np.copy(initial_population).reshape((n_states + 3) * n_ages)  ## to be vstacked

    weeks_no_vacc = 10  ## initially no vaccination
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan[0:(weeks_no_vacc*7)],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out))

    old_init = out[-1].reshape((n_states + 3, n_ages))

    ## Set my weekly vaccination schedule (starting from week 10)
    # total_weeks = 52
    vacc_weeks = 24

    cum_weekly_vacc_progress, vacc_dates = get_weekly_vaccination_rates()
    # cum_weekly_vacc_progress_laiv, _ = get_weekly_vaccination_rates_LAIV()
    # cum_weekly_vacc_progress_laiv, _ = get_weekly_vaccination_rates()  ## TESTING


    cum_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks))

    for vacc_week in range(vacc_weeks):
        for a in range(n_ages):
            cum_weekly_vaccination_schedule[a, 1:] = cum_weekly_vacc_progress * vc[a]
            new_weekly_vaccination_schedule[a] = cum_weekly_vaccination_schedule[a][1:] - \
                                                 cum_weekly_vaccination_schedule[a][:-1]

    # set initial condition with vaccination
    # old_init = out_main.reshape((n_states + 3, n_ages))
    # my_t0 = 0  ## time counter
    my_t0 = int(tspan[0:(weeks_no_vacc*7)][-1] + 1)
    total_doses_frac = np.zeros(n_ages)

    for vacc_week in range(vacc_weeks):
        # print(vacc_week, my_t0)
        ## do IIV campaign
        vc = new_weekly_vaccination_schedule[:, vacc_week]
        ## recalculate vc relative to the leftover unvaccinated population
        my_sr_pop = old_init[(SU, SW, RU, RW), :].sum()
        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            frac_to_vaccinate = vc * old_init[old_state]
            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
            total_doses_frac += frac_to_vaccinate

        ## run simulation
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                           args=(parameters_ode,), full_output=True)

        ## save my time series
        out_main = np.vstack((out_main, out[1:]))

        ## update initial condition and time span
        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7
    total_doses = N * total_doses_frac

    ## run to week 52 (end of year)
    out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out[1:]))

    return out_main, total_doses

## LAIV vaccination campaign both run over 6 weeks from March (to mid April)
def run_model_equations_weekly_vaccination_3_laiv(initial_SW=np.zeros(n_ages), amplitude = 0.3, R0 = 1.5, vaccine_uptake = 'pessimistic', vaccine_policy='status_quo', contact_matrix = np.ones((n_ages, n_ages))):
    ''' LAIV and IIV program run separately here so that LAIV can run during school term (6 weeks)
    vaccination campaign starts in week 10 (corresponding to 7 march)
    this change is implemented in response to Isaac's latest calibration for Wed 3 Sep 2025
    runs the ACEFA model for a specified amplitude, R0, and vaccine uptake (which is further divided into 3
    vaccine policy)

    This function differs from run_model_equations in the following ways:
    - Vaccination runs for 24 weeks starting from March.
    - t0 is March 1 here (i.e. phase_shift = 60 here)

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns a daily time series of (n_states + 3) and n_ages
    '''

    ## varied parameters
    # contact_matrix = load_contact_matrix(populations_by_fractions)
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / 27_586_007  ### I CHANGED THIS!!!
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    #### Set vaccine coverage by vaccine uptake and vaccine policy
    ## vaccine_uptake: 'pessimistic' OR 'mid' OR 'optimistic'
    ## vaccine_policy: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    ## policy: status quo
    vc_laiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_LAIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)
    vc_iiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_IIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    # parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## Run model without vaccination for 10 weeks, then weeks 11 to 35 is the campaign, and weeks 36-52 is no vaccination again
    out_main = np.copy(initial_population).reshape((n_states + 3) * n_ages)  ## to be vstacked

    weeks_no_vacc = 12  ## initially no vaccination
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan[0:(weeks_no_vacc*7)],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out))

    old_init = out[-1].reshape((n_states + 3, n_ages))



    ## Set my weekly vaccination schedule (starting from week 10)
    # total_weeks = 52
    vacc_weeks = 24

    cum_weekly_vacc_progress, vacc_dates = get_weekly_vaccination_rates()
    cum_weekly_vacc_progress_laiv, _ = get_weekly_vaccination_rates_LAIV()

    cum_weekly_vaccination_schedule_iiv = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule_iiv = np.zeros((n_ages, vacc_weeks))

    cum_weekly_vaccination_schedule_laiv = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule_laiv = np.zeros((n_ages, vacc_weeks))

    vc_main = np.zeros(9)  ## used to check overall VC at the end

    # plt.figure(figsize=(4, 3), layout='tight')
    # plt.plot(cum_weekly_vacc_progress, label='iiv')
    # plt.plot(cum_weekly_vacc_progress_laiv, label='laiv')
    # plt.legend()
    #
    # plt.xlabel('Vaccination week')
    # plt.ylabel('Progress (%)')
    # plt.show(block=True)

    # plt.figure(figsize=(5, 4), layout='tight')
    # plt.title('Vaccination campaign progression (%)', fontsize=10)
    # plt.plot(range(len(cum_weekly_vacc_progress)), cum_weekly_vacc_progress * 100, 'k', marker='o', label='iiv')
    # plt.plot(range(len(cum_weekly_vacc_progress_laiv)), cum_weekly_vacc_progress_laiv * 100, 'b', marker='o', label='laiv')
    # # plt.xlabel('Vaccination campaign progression (%)')
    # plt.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False)
    # # plt.rcParams.update({'axes.spines.top': True, 'axes.spines.right': True})
    # plt.xticks(range(len(cum_weekly_vacc_progress)), vacc_dates, rotation=45)
    # # plt.gca().invert_yaxis()
    # plt.ylabel('Progression (%)')
    # plt.legend()
    # # plt.xlabel('Week ending in')
    # plt.show(block=True)


    for a in range(n_ages):
        cum_weekly_vaccination_schedule_iiv[a, 1:] = cum_weekly_vacc_progress * vc_iiv[a]
        new_weekly_vaccination_schedule_iiv[a] = cum_weekly_vaccination_schedule_iiv[a][1:] - \
                                             cum_weekly_vaccination_schedule_iiv[a][:-1]
        cum_weekly_vaccination_schedule_laiv[a, 1:] = cum_weekly_vacc_progress_laiv * vc_laiv[a]
        new_weekly_vaccination_schedule_laiv[a] = cum_weekly_vaccination_schedule_laiv[a][1:] - \
                                             cum_weekly_vaccination_schedule_laiv[a][:-1]


    # initial_population_endemic = np.copy(initial_population)
    # out_main = initial_population_endemic.reshape((n_states + 3) * n_ages)  ## to be vstacked

    # set initial condition with vaccination
    # old_init = out_main.reshape((n_states + 3, n_ages))
    my_t0 = 0  ## time counter
    my_t0 = int(tspan[0:(weeks_no_vacc*7)][-1] + 1)
    total_doses_frac = np.zeros(n_ages)

    for vacc_week in range(vacc_weeks):
        # print(vacc_week, my_t0)
        ## do IIV campaign
        vc_iiv = new_weekly_vaccination_schedule_iiv[:, vacc_week]
        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            # frac_to_vaccinate = vc_iiv * old_init[old_state]

            ## Adjust vc_iiv rate with denominator as the eligible unvaccinated S+R population
            unvaccinated_fraction_of_SR = (old_init[(SU, SW, RU, RW), :] * (vc_iiv > 0)).sum() / (populations_by_fractions * (vc_iiv > 0)).sum()
            vaccinated_fraction_of_SR = (old_init[(SV, RV), :] * (vc_iiv > 0)).sum() / (populations_by_fractions * (vc_iiv > 0)).sum()

            n_doses_actual = (populations_by_fractions * N) * vc_iiv  ## length n_ages
            vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * N))

            # do the vaccination with vc_weekly_actual
            frac_to_vaccinate = vc_weekly_actual * old_init[old_state]

            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
            total_doses_frac += frac_to_vaccinate

        ## do LAIV campaign
        vc_laiv = new_weekly_vaccination_schedule_laiv[:, vacc_week]
        if vc_laiv.sum() > 0:
            for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
                # frac_to_vaccinate = vc_laiv * old_init[old_state]

                ## Adjust vc_iiv rate with denominator as the eligible unvaccinated S+R population
                unvaccinated_fraction_of_SR = (old_init[(SU, SW, RU, RW), :] * (vc_laiv > 0)).sum() / (populations_by_fractions * (vc_laiv > 0)).sum()
                vaccinated_fraction_of_SR = (old_init[(SV, RV), :] * (vc_laiv > 0)).sum() / (populations_by_fractions * (vc_laiv > 0)).sum()

                n_doses_actual = (populations_by_fractions * N) * vc_laiv  ## length n_ages
                # vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * (vc_laiv > 0)).sum() * N)
                vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * N))

                # do the vaccination with vc_weekly_actual
                frac_to_vaccinate = vc_weekly_actual * init_new[old_state]

                init_new[old_state] = init_new[old_state] - frac_to_vaccinate
                init_new[new_state] += frac_to_vaccinate
                total_doses_frac += frac_to_vaccinate

        ## run simulation
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                           args=(parameters_ode,), full_output=True)

        ## save my time series
        out_main = np.vstack((out_main, out[1:]))

        ## update initial condition and time span
        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7
        vc_main += (vc_laiv + vc_iiv)
    total_doses = N * total_doses_frac

    ## run to week 52 (end of year)
    out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out[1:]))

    # (my_i, my_c, my_h) = calculate_weekly_outcomes(y=out_main.reshape((-1, n_states + 3, n_ages)),
    #                                                parameters=parameters_outcome)
    # print(my_i.max())
    # plt.plot(my_i, label=vaccine_policy)
    # plt.legend()
    # plt.show(block=True)

    ## check number of total and LAIV doses used per scenario to see if it checks out (it does!)
    # print(vaccine_policy, vaccine_uptake)
    # print(vc_main)
    # total_doses[3:5].sum()
    # total_doses.sum() - total_doses[3:5].sum()
    # print(int(total_doses.sum()), 'doses,', int(total_doses[3:5].sum()), 'doses in 5-18')
    #
    # total_doses[3].sum()
    # total_doses.sum() - total_doses[3].sum()
    # print(int(total_doses.sum()), 'doses,', int(total_doses[3].sum()), 'doses in 5-12')


    return out_main, total_doses



def calculate_weekly_outcomes(y, parameters):
    '''takes a daily time series (of length 7 * total_weeks + 1) and returns weekly infections, weekly (observed) cases, and
    weekly hospitalizations
    y.shape = (365, 18, 9) or (days, n_states + 3, n_ages)'''

    [total_weeks, N, VE_INF0, VE_GP0, VE_HOSP0, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus] = parameters

    y_IU, y_IV, y_IW = y[:, 15, :], y[:, 16, :], y[:, 17, :]
    i_unvacc = (y_IU[1:] - y_IU[:-1]) * N
    i_vacc = (y_IV[1:] - y_IV[:-1]) * N
    i_waning = (y_IW[1:] - y_IW[:-1]) * N
    #
    # plt.plot(i_unvacc.sum(-1), label='unvac')
    # plt.plot(i_vacc.sum(-1), label='vacc')
    # plt.plot(i_waning.sum(-1), label='waning')
    # plt.show()

    ## Recalculate VE conditional on infection (Ioannidis 2022, European Journal of Epidemiology)
    VE_HOSP = 1 - np.divide(1 - VE_HOSP0, 1 - VE_INF0)
    VE_GP = 1 - np.divide(1 - VE_GP0, 1 - VE_INF0)

    my_infections = np.sum(i_unvacc + i_vacc + i_waning, -1)

    my_cases = np.dot((1 - p_asymptomatic) * i_unvacc, p_healthseeking) + \
               np.dot((1 - VE_GP) * (1 - p_asymptomatic) * i_vacc, p_healthseeking) + \
               np.dot((1 - p_asymptomatic) * i_waning, p_healthseeking)

    my_hospitalizations = np.dot((1 - p_asymptomatic) * i_unvacc, p_hospitalised) + \
                          np.dot((1 - VE_HOSP) * (1 - p_asymptomatic) * i_vacc, p_hospitalised) + \
                          np.dot((1 - p_asymptomatic) * i_waning, p_hospitalised)
    #
    weekly_infections = my_infections.reshape((-1, 7)).sum(1)
    weekly_cases = my_cases.reshape((-1, 7)).sum(1)
    weekly_hospitalizations = my_hospitalizations.reshape((-1, 7)).sum(1)

    return [weekly_infections, weekly_cases, weekly_hospitalizations]

def calculate_total_outcomes_by_age(y, parameters):
    '''  '''
    [total_weeks, N, VE_INF0, VE_GP0, VE_HOSP0,
     p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus] = parameters

    y_IU, y_IV, y_IW = y[:, 15, :], y[:, 16, :], y[:, 17, :]
    i_unvacc = (y_IU[1:] - y_IU[:-1]) * N
    i_vacc = (y_IV[1:] - y_IV[:-1]) * N
    i_waning = (y_IW[1:] - y_IW[:-1]) * N
    #
    ## Recalculate VE conditional on infection (Ioannidis 2022, European Journal of Epidemiology)
    VE_HOSP = 1 - np.divide(1 - VE_HOSP0, 1 - VE_INF0)
    VE_GP = 1 - np.divide(1 - VE_GP0, 1 - VE_INF0)
    #
    my_infections = i_unvacc + i_vacc + i_waning

    my_cases = np.multiply((1 - p_asymptomatic) * i_unvacc, p_healthseeking) + \
               np.multiply((1 - VE_GP) * (1 - p_asymptomatic) * i_vacc, p_healthseeking) + \
               np.multiply((1 - p_asymptomatic) * i_waning, p_healthseeking)

    my_hospitalizations = np.multiply((1 - p_asymptomatic) * i_unvacc, p_hospitalised) + \
                          np.multiply((1 - VE_HOSP) * (1 - p_asymptomatic) * i_vacc, p_hospitalised) + \
                          np.multiply((1 - p_asymptomatic) * i_waning, p_hospitalised)
    #
    total_infections = my_infections.sum(0)
    total_cases = my_cases.sum(0)
    total_hospitalizations = my_hospitalizations.sum(0)

    return [total_infections, total_cases, total_hospitalizations]

def calculate_weekly_outcomes_by_age(y, parameters):
    [total_weeks, N, VE_INF0, VE_GP0, VE_HOSP0, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus] = parameters

    y_IU, y_IV, y_IW = y[:, 15, :], y[:, 16, :], y[:, 17, :]
    i_unvacc = (y_IU[1:] - y_IU[:-1]) * N
    i_vacc = (y_IV[1:] - y_IV[:-1]) * N
    i_waning = (y_IW[1:] - y_IW[:-1]) * N
    my_infections = i_unvacc + i_vacc + i_waning
    #
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
        y = my_infections[:-(my_infections.shape[0] % 7)]

    weekly_infections_by_age = my_infections.reshape(my_infections.shape[0] // 7, 7, my_infections.shape[1]).sum(1)

    if my_cases.shape[0] % 7 != 0:
        print("Warning: The number of rows is not a multiple of 7. Truncating the array.")
        y = my_cases[:-(my_cases.shape[0] % 7)]

    weekly_cases_by_age = my_cases.reshape(my_cases.shape[0] // 7, 7, my_cases.shape[1]).sum(1)

    if my_hospitalizations.shape[0] % 7 != 0:
        print("Warning: The number of rows is not a multiple of 7. Truncating the array.")
        y = my_hospitalizations[:-(my_hospitalizations.shape[0] % 7)]

    weekly_hospitalizations_by_age = my_hospitalizations.reshape(my_hospitalizations.shape[0] // 7, 7, my_hospitalizations.shape[1]).sum(1)

    return [weekly_infections_by_age, weekly_cases_by_age, weekly_hospitalizations_by_age]

def load_scenario_VC(vaccine_policy='status_quo', vaccine_scenario='pessimistic'):
    ''' for each vaccine policy (status_quo, laiv_5_12, laiv_5_18) and
    each vaccine scenario (status_quo, pessimistic, mid, optimistic):
    loads the vaccine coverage and vaccine effectiveness:
    - status quo: standard vc, standard ve
    - pessimistic: low vc, low ve
    - mid: mid vc, mid ve
    - optimistic: high vc, high ve
    Uses 9 age groups.
    '''

    n_ages = 9
    # vaccine_policy = 'laiv_5_12'
    # vaccine_scenario = 'optimistic'

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

    my_vaccine_dict = {
        'vaccine_policy': vaccine_policy,
        'vaccine_scenario': vaccine_scenario,
        'vaccine_coverage': vc_of_vaccine_policy[my_policy_index],
        've_infection': ve_inf_main,
        've_gp': ve_gp_main,
        've_hosp': ve_hosp_main
    }

    return my_vaccine_dict

def run_ACEFA_scenarios():
    ''' here, t0 = May 1 (so phase shift is 120), and the vaccination campaign is done in one go at the start of simulation '''

    ## fixed parameters obtained from default_params.py
    amplitude = 0.3

    ## Vaccine parameters
    (vaccine_coverage, VE_SUS, VE_GP, VE_HOSP) = load_VE_and_VC(vaccine_policy='status_quo', vaccine_scenario='pessimistic')

    # VE_SUS_TIV = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.37, 0.37, 0.18, 0.18])  ## VE against any influenza (Frutos2025)
    # VE_SUS_LAIV = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4])  ## against matched strains, Mohn2018
    # VE_HOSP = np.array([0., 0.71, 0.71, 0.49, 0.49, 0.51, 0.40, 0.36, 0.36])  ## TIV, Tenforde2024b
    # VE_GP = np.array([0.32, 0.32, 0.32, 0.32, 0.32, 0.36, 0.36, 0.36, 0.36])  ## against medically attended influenza (Frutos2025)

    VE_SUS = np.copy(VE_SUS_TIV)
    VE_SUS[:2] = VE_SUS_LAIV[0:2]  ## VE_SUS for LAIV for 5-12YO, and for seasonal vaccine for other ages

    ## varied parameters
    R0 = 1.5
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    tspan = np.linspace(0, 364 * 1, 364 * 1 + 1)
    total_weeks = int(len(tspan) / 7)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS]
    parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    #### Set initial condition
    initial_poxpulation = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / N
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of I's
    print('Initial population sums to: ', initial_population[:-3, :].sum())

    ## transient run of 5 years with a fully susceptible population:
    tspan_transient = np.linspace(0, 364 * 5, 364 * 5 + 1)
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan_transient,
                       args=(parameters_ode,), full_output=True)
    initial_population_endemic = out[-1].reshape((n_states + 3, n_ages))
    print('yinit after 5-year burn-in'
          '\n  Unvaccinated SEEIR: ', np.round(initial_population_endemic[0:5, :].sum(1), 3),
          '\n    Vaccinated SEEIR: ', np.round(initial_population_endemic[5:10, :].sum(1), 3),
          '\n        Waning SEEIR: ', np.round(initial_population_endemic[10:15, :].sum(1), 3))

    my_policies = ['laiv_5_12', 'laiv_5_18']
    my_vaccine_scenarios = ['status_quo', 'pessimistic', 'mid', 'optimistic']

    ## My loop through the vaccine policies and scenarios
    counter = 1
    scenarios_string = 'ABCDEFG'
    colors_colorblind = sns.color_palette("colorblind")
    scenario_colors = ['black', colors_colorblind[-3], colors_colorblind[-5], colors_colorblind[-1]]

    ## policy: status quo
    my_vaccine_dict = load_scenario_VC(vaccine_policy='status_quo', vaccine_scenario='status_quo')
    VE_SUS = my_vaccine_dict["ve_infection"]
    VE_GP = my_vaccine_dict["ve_gp"]
    VE_HOSP = my_vaccine_dict["ve_hosp"]

    VE_SUS_L, VE_SUS_U = my_vaccine_dict["ve_infection_bounds"]
    VE_GP_L, VE_GP_U = my_vaccine_dict["ve_gp_bounds"]
    VE_HOSP_L, VE_HOSP_U = my_vaccine_dict["ve_hosp_bounds"]

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## vaccinate people manually
    vc = my_vaccine_dict["vaccine_coverage"]
    init_new = np.copy(initial_population_endemic)
    for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
        frac_to_vaccinate = vc * initial_population_endemic[old_state]
        init_new[old_state] = initial_population_endemic[old_state] - frac_to_vaccinate
        init_new[new_state] += frac_to_vaccinate

    # run the model
    out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan,
                       args=(parameters_ode,), full_output=True)

    (weekly_infections_sq, weekly_cases_sq,
     weekly_hosps_sq) = calculate_weekly_outcomes(y=out.reshape((-1, n_states + 3, n_ages)),
                                                  parameters=parameters_outcome)

    (weekly_infections_age_sq, weekly_cases_age_sq,
     weekly_hosps_age_sq) = calculate_weekly_outcomes_by_age(y=out.reshape((-1, n_states + 3, n_ages)),
                                                             parameters=parameters_outcome)

    (total_infections_sq, total_cases_sq,
     total_hospitalizations_sq) = calculate_total_outcomes_by_age(y=out.reshape((-1, n_states + 3, n_ages)),
                                                                  parameters=parameters_outcome)



    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    for fig_val, my_policy in enumerate(['laiv_5_12', 'laiv_5_18']):

        ## Set up arrays for model outcomes
        weekly_infections_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_cases_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), total_weeks))

        weekly_infections_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_cases_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_hospitalizations_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))

        total_infections_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_cases_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), n_ages))

        ## fill in status quo values
        weekly_infections_main[0] = weekly_infections_sq
        weekly_cases_main[0] = weekly_cases_sq
        weekly_hospitalizations_main[0] = weekly_hosps_sq

        weekly_infections_age_main[0] = weekly_infections_age_sq
        weekly_cases_age_main[0] = weekly_cases_age_sq
        weekly_hospitalizations_age_main[0] = weekly_hosps_age_sq

        total_infections_main[0] = total_infections_sq
        total_cases_main[0] = total_cases_sq
        total_hospitalizations_main[0] = total_hospitalizations_sq

        for ival, my_vaccine_scenario in enumerate(['pessimistic', 'mid', 'optimistic']):
            # print('Scenario', scenarios_string[counter], my_policy.upper(), ',', my_vaccine_scenario)
            counter += 1
            ## update my vaccine parameters
            my_vaccine_dict = load_scenario_VC(vaccine_policy=my_policy, vaccine_scenario=my_vaccine_scenario)
            VE_SUS = my_vaccine_dict["ve_infection"]
            VE_GP = my_vaccine_dict["ve_gp"]
            VE_HOSP = my_vaccine_dict["ve_hosp"]

            parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS]
            parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

            ## vaccinate people manually
            vc = my_vaccine_dict["vaccine_coverage"]
            init_new = np.copy(initial_population_endemic)
            for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
                frac_to_vaccinate = vc * initial_population_endemic[old_state]
                init_new[old_state] = initial_population_endemic[old_state] - frac_to_vaccinate
                init_new[new_state] += frac_to_vaccinate

            # run the model
            out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan,
                               args=(parameters_ode,), full_output=True)

            (weekly_infections_main[ival + 1], weekly_cases_main[ival + 1],
             weekly_hospitalizations_main[ival + 1]) = calculate_weekly_outcomes(y=out.reshape((-1, n_states + 3, n_ages)),
                                                                             parameters=parameters_outcome)

            (total_infections_main[ival + 1], total_cases_main[ival + 1],
             total_hospitalizations_main[ival + 1]) = calculate_total_outcomes_by_age(y=out.reshape((-1, n_states + 3, n_ages)),
                                                                                  parameters=parameters_outcome)

        ## Plot
        vaccine_strategy_names = my_vaccine_scenarios
        outcome_names = ['new infections', 'new cases', 'new hospitalisations']
        my_outcomes_weekly = [weekly_infections_main, weekly_cases_main, weekly_hospitalizations_main]
        my_outcomes_total = [total_infections_main, total_cases_main, total_hospitalizations_main]

        months = 'MJJASONDJFMAM'

        plt.figure(1 + fig_val*10, figsize=(13, 4), layout='tight')
        plt.suptitle('Model outcomes for policy option ' + my_policy.upper(), fontsize=16)
        print(1 + fig_val, my_policy.upper())
        # print('outcome_name [status_quo, pessimistic, mid, optimistic]')
        # print('outcome_name [outcomes prevented compared to status_quo]')
        for fval, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print(outcome_name, np.round(outcome.sum(-1), 0))  ## outcome shape: (n_vaccine_scenarios, n_total_weeks)
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(outcome_names), fval + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot(outcome[jval], label=vaccine_strategy_names[jval], color=scenario_colors[jval])
            if fval == 0:
                plt.ylabel('Individuals', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Weeks', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
            # plt.legend()
        figname = 'plots/model_outcome_prototype_timeseries_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        # plt.figure(2 + fig_val)
        barWidth = 0.2
        bar1 = np.arange(n_ages)
        bar2 = [x + barWidth for x in bar1]
        bar3 = [x + barWidth for x in bar2]
        bar4 = [x + barWidth for x in bar3]
        my_bars = [bar1, bar2, bar3, bar4]

        plt.figure(2 + fig_val*10, figsize=(6, 8), layout='tight')
        plt.suptitle(my_policy.upper() + ': Total outcomes by age', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name[4:], fontsize=16)
            ## print the outcomes
            print(2 + fig_val, my_policy.upper())
            print(' ', outcome_name)
            for jval in range(len(outcome)):
                print('    ', vaccine_strategy_names[jval], np.round(outcome[jval], 0))
                print('    ', vaccine_strategy_names[jval], np.round(outcome[jval] - outcome[0], 0))
                plt.bar(my_bars[jval], outcome[jval], width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_by_age_and_scenario_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(5 + fig_val*10, figsize=(6, 8), layout='tight')
        plt.suptitle(my_policy.upper() + ': Total outcomes by age (per thousand)', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name[4:], fontsize=16)
            ## print the outcomes
            print(2 + fig_val, my_policy.upper())
            print(' ', outcome_name)
            for jval in range(len(outcome)):
                x = outcome[jval] / (populations_by_fractions * N) * 1000
                plt.bar(my_bars[jval], x, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_per_1000_by_age_and_scenario_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(4 + fig_val*10, figsize=(6, 8), layout='tight')
        ymaxes = [10000, 3200, 3.1]  ## currently manually set because I'm lazy
        plt.suptitle(my_policy.upper() + ': Outcomes prevented by age', fontsize=16)
        ## create my status quo infections/cases/hospitalisations
        outcomes_prevented = np.zeros((len(outcome_names), len(vaccine_strategy_names), n_ages))
        outcomes_status_quo = np.zeros((len(outcome_names), n_ages))
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            outcomes_status_quo[ival] = outcome[0]
            # print(np.round(outcome[0],0))  ## outcomes for status quo
            #
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range(len(outcome)):
                outcome_prevented = outcome[jval] - outcomes_status_quo[ival]
                outcomes_prevented[ival, jval, :] = outcome_prevented
                plt.bar(my_bars[jval], -outcome_prevented, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
                plt.ylim([0, ymaxes[ival]])
        figname = 'plots/model_outcomes_prevented_prototype_by_age_and_scenario_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(3 + fig_val*10, figsize=(13, 4), layout='tight')
        plt.suptitle(my_policy.upper() + ': Cumulative outcomes by scenario', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print('Cumulative', outcome_name[4:], np.round(outcome.sum(-1), 0))  ## outcome.shape (n_vaccine_scenarios, n_total_weeks) = (4, 54)
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(outcome_names), ival + 1)
            plt.title('Cumulative ' + outcome_name[4:], fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot((np.cumsum(outcome[jval]) / N) * 100, label=vaccine_strategy_names[jval],
                         color=scenario_colors[jval])
            if ival == 0:
                plt.ylabel('Population (%)', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Time (Months)', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_cumulative_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
    plt.show(block=True)

    return []


def run_ACEFA_scenarios_with_weekly_vacc():
    '''
    here t0 is mar 1 (so phase shift is 60)
    '''
    
    amplitude = 0.3

    ## varied parameters
    R0 = 1.5
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    tspan = np.linspace(0, 364 * 1, 364 * 1 + 1)
    total_weeks = int(len(tspan) / 7)

    vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy='status_quo', vaccine_scenario='status_quo')

    #### Set initial condition
    initial_SW = np.repeat(0.1, 9)
    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / N
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    ## transient run of 5 years with a fully susceptible population:
    parameters_burnin = [amplitude, beta, contact_matrix, gamma, 1/180, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    tspan_transient = np.linspace(0, 364 * 5, 364 * 5 + 1)
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan_transient,
                       args=(parameters_burnin,), full_output=True)
    initial_population_endemic = out[-1].reshape((n_states + 3, n_ages))
    print('yinit after 5-year burn-in'
          '\n  Unvaccinated SEEIR: ', np.round(initial_population_endemic[0:5, :].sum(1), 3),
          '\n    Vaccinated SEEIR: ', np.round(initial_population_endemic[5:10, :].sum(1), 3),
          '\n        Waning SEEIR: ', np.round(initial_population_endemic[10:15, :].sum(1), 3))

    my_policies = ['laiv_5_12', 'laiv_5_18']
    my_vaccine_scenarios = ['status_quo', 'pessimistic', 'mid', 'optimistic']

    ## My loop through the vaccine policies and scenarios
    counter = 1
    scenarios_string = 'ABCDEFG'
    colors_colorblind = sns.color_palette("colorblind")
    scenario_colors = ['black', colors_colorblind[-3], colors_colorblind[-5], colors_colorblind[-1]]

    ## policy: status quo
    vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy='status_quo', vaccine_scenario='status_quo')

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## run model weekly from Mar 1 to Aug 31 for the vaccination ramp-up period over 24 weeks
    total_weeks = 52
    vacc_weeks = 24

    cum_weekly_vacc_progress, vacc_dates = get_weekly_vaccination_rates()

    cum_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks))

    out_main = initial_population_endemic.reshape((n_states + 3) * n_ages)  ## to be vstacked

    for vacc_week in range(vacc_weeks):
        for a in range(n_ages):
            cum_weekly_vaccination_schedule[a, 1:] = cum_weekly_vacc_progress * vc[a]
            new_weekly_vaccination_schedule[a] = cum_weekly_vaccination_schedule[a][1:] - \
                                                 cum_weekly_vaccination_schedule[a][:-1]

    old_init = out_main.reshape((n_states + 3, n_ages))
    my_t0 = 0
    total_doses_frac = np.zeros(n_ages)
    for vacc_week in range(vacc_weeks):
        vc = new_weekly_vaccination_schedule[:, vacc_week]
        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            frac_to_vaccinate = vc * old_init[old_state]
            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
            total_doses_frac += frac_to_vaccinate
        ## run simulation
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                           args=(parameters_ode,), full_output=True)

        ## save my time series
        out_main = np.vstack((out_main, out[1:]))

        ## update initial condition and time span
        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7
    total_doses_sq = N * total_doses_frac


    ## run to week 52 (end of year)
    out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out[1:]))

    init_new = out_main[-1].reshape((n_states + 3, n_ages))

    (weekly_infections_sq, weekly_cases_sq,
     weekly_hosps_sq) = calculate_weekly_outcomes(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                  parameters=parameters_outcome)

    (weekly_infections_age_sq, weekly_cases_age_sq,
     weekly_hosps_age_sq) = calculate_weekly_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                             parameters=parameters_outcome)

    (total_infections_sq, total_cases_sq,
     total_hospitalizations_sq) = calculate_total_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                  parameters=parameters_outcome)

    total_doses_iiv_main = np.zeros((3, 3))  ## nrows = sq, laiv-12, laiv-18 // ncols = pes, mid, opt (so row 1 = sq, sq, sq)
    total_doses_iiv_main[0, :] = total_doses_sq.sum()

    total_doses_laiv_main = np.zeros((3, 3))  ## nrows = sq, laiv-12, laiv-18 // ncols = pes, mid, opt (so row 1 = sq, sq, sq)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    for fig_val, my_policy in enumerate(['laiv_5_12', 'laiv_5_18']):

        ## Set up arrays for model outcomes
        weekly_infections_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_cases_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), total_weeks))

        weekly_infections_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_cases_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_hospitalizations_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))

        total_infections_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_cases_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), n_ages))

        ## fill in status quo values
        weekly_infections_main[0] = weekly_infections_sq
        weekly_cases_main[0] = weekly_cases_sq
        weekly_hospitalizations_main[0] = weekly_hosps_sq

        weekly_infections_age_main[0] = weekly_infections_age_sq
        weekly_cases_age_main[0] = weekly_cases_age_sq
        weekly_hospitalizations_age_main[0] = weekly_hosps_age_sq

        total_infections_main[0] = total_infections_sq
        total_cases_main[0] = total_cases_sq
        total_hospitalizations_main[0] = total_hospitalizations_sq


        for ival, my_vaccine_scenario in enumerate(['pessimistic', 'mid', 'optimistic']):
            # print('Scenario', scenarios_string[counter], my_policy.upper(), ',', my_vaccine_scenario)
            counter += 1
            ## update my vaccine parameters
            vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy=my_policy, vaccine_scenario=my_vaccine_scenario)

            parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
            parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

            ## vaccinate people manually: define my weekly vaccination schedule
            # vc = my_vaccine_dict["vaccine_coverage"]
            cum_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks + 1))
            new_weekly_vaccination_schedule = np.zeros((n_ages, vacc_weeks))

            out_main = initial_population_endemic.reshape((n_states + 3) * n_ages)  ## to be vstacked

            for vacc_week in range(vacc_weeks):
                for a in range(n_ages):
                    cum_weekly_vaccination_schedule[a, 1:] = cum_weekly_vacc_progress * vc[a]
                    new_weekly_vaccination_schedule[a] = cum_weekly_vaccination_schedule[a][1:] - \
                                                         cum_weekly_vaccination_schedule[a][:-1]

            old_init = out_main.reshape((n_states + 3, n_ages))
            my_t0 = 0
            total_doses_frac = np.zeros(n_ages)
            for vacc_week in range(vacc_weeks):
                # print('week ', vacc_week, 'vac, t_start:', tspan[my_t0:my_t0 + 8][0], ',', tspan[my_t0:my_t0 + 8][-1])
                vc = new_weekly_vaccination_schedule[:, vacc_week]
                init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
                for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
                    frac_to_vaccinate = vc * old_init[old_state]
                    init_new[old_state] = old_init[old_state] - frac_to_vaccinate
                    init_new[new_state] += frac_to_vaccinate
                    total_doses_frac += frac_to_vaccinate
                ## run simulation
                out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages),
                                   tspan[my_t0:my_t0 + 8],
                                   args=(parameters_ode,), full_output=True)

                ## save my time series
                out_main = np.vstack((out_main, out[1:]))

                ## update initial condition and time span
                old_init = out[-1].reshape((n_states + 3, n_ages))
                my_t0 += 7
            total_doses = N * total_doses_frac

            total_doses_laiv_main[fig_val + 1, ival] = total_doses[3]
            total_doses_iiv_main[fig_val + 1, ival] = total_doses.sum() - total_doses[3]

            if my_policy == 'laiv_5_18':
                total_doses_laiv_main[fig_val + 1, ival] += total_doses[4]
                total_doses_iiv_main[fig_val + 1, ival] = total_doses.sum() - total_doses_laiv_main[fig_val + 1, ival]

            ## run to week 52 (end of year)
            out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                               args=(parameters_ode,), full_output=True)

            out_main = np.vstack((out_main, out[1:]))

            (weekly_infections_main[ival + 1], weekly_cases_main[ival + 1],
             weekly_hospitalizations_main[ival + 1]) = calculate_weekly_outcomes(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                                 parameters=parameters_outcome)

            (total_infections_main[ival + 1], total_cases_main[ival + 1],
             total_hospitalizations_main[ival + 1]) = calculate_total_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                                      parameters=parameters_outcome)

        ## Plot
        vaccine_strategy_names = my_vaccine_scenarios
        outcome_names = ['new infections', 'new cases', 'new hospitalisations']
        my_outcomes_weekly = [weekly_infections_main, weekly_cases_main, weekly_hospitalizations_main]
        my_outcomes_total = [total_infections_main, total_cases_main, total_hospitalizations_main]

        # months = 'MJJASONDJFMAM'
        months = 'MAMMJJASONDJF'  ## mar to feb


        plt.figure(1 + fig_val*10, figsize=(13, 4), layout='tight')
        plt.suptitle('Model outcomes for policy option ' + my_policy.upper(), fontsize=16)
        print(1 + fig_val, my_policy.upper())
        # print('outcome_name [status_quo, pessimistic, mid, optimistic]')
        # print('outcome_name [outcomes prevented compared to status_quo]')
        for fval, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print(outcome_name, np.round(outcome.sum(-1), 0))  ## outcome shape: (n_vaccine_scenarios, n_total_weeks)
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(outcome_names), fval + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot(outcome[jval], label=vaccine_strategy_names[jval], color=scenario_colors[jval])
            if fval == 0:
                plt.ylabel('Individuals', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Weeks', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
            # plt.legend()
        figname = 'plots/model_outcome_prototype_timeseries_weekly_campaign_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        # plt.figure(2 + fig_val)
        barWidth = 0.2
        bar1 = np.arange(n_ages)
        bar2 = [x + barWidth for x in bar1]
        bar3 = [x + barWidth for x in bar2]
        bar4 = [x + barWidth for x in bar3]
        my_bars = [bar1, bar2, bar3, bar4]

        plt.figure(2 + fig_val*10, figsize=(6, 8), layout='tight')
        plt.suptitle(my_policy.upper() + ': Total outcomes by age', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name[4:], fontsize=16)
            ## print the outcomes
            print(2 + fig_val, my_policy.upper())
            print(' ', outcome_name)
            for jval in range(len(outcome)):
                print('    ', vaccine_strategy_names[jval], np.round(outcome[jval], 0))
                print('    ', vaccine_strategy_names[jval], np.round(outcome[jval] - outcome[0], 0))
                plt.bar(my_bars[jval], outcome[jval], width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(5 + fig_val*10, figsize=(6, 8), layout='tight')
        plt.suptitle(my_policy.upper() + ': Total outcomes by age (per thousand)', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name[4:], fontsize=16)
            ## print the outcomes
            print(2 + fig_val, my_policy.upper())
            print(' ', outcome_name)
            for jval in range(len(outcome)):
                x = outcome[jval] / (populations_by_fractions * N) * 1000
                plt.bar(my_bars[jval], x, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_per_1000_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(4 + fig_val*10, figsize=(6, 8), layout='tight')
        ymaxes = [10000, 3200, 3.1]  ## currently manually set because I'm lazy
        plt.suptitle(my_policy.upper() + ': Outcomes prevented by age', fontsize=16)
        ## create my status quo infections/cases/hospitalisations
        outcomes_prevented = np.zeros((len(outcome_names), len(vaccine_strategy_names), n_ages))
        outcomes_status_quo = np.zeros((len(outcome_names), n_ages))
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            outcomes_status_quo[ival] = outcome[0]
            # print(np.round(outcome[0],0))  ## outcomes for status quo
            #
            plt.subplot(len(outcome_names), 1, ival + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range(len(outcome)):
                outcome_prevented = outcome[jval] - outcomes_status_quo[ival]
                outcomes_prevented[ival, jval, :] = outcome_prevented
                plt.bar(my_bars[jval], -outcome_prevented, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
                        color=scenario_colors[jval])
                plt.xticks(bar2, '')
                if ival == 0:
                    plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
                plt.ylim([0, ymaxes[ival]])
        figname = 'plots/model_outcomes_prevented_prototype_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
        # plt.show(block=True)

        plt.figure(3 + fig_val*10, figsize=(13, 4), layout='tight')
        plt.suptitle(my_policy.upper() + ': Cumulative outcomes by scenario', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print('Cumulative', outcome_name[4:], np.round(outcome.sum(-1), 0))  ## outcome.shape (n_vaccine_scenarios, n_total_weeks) = (4, 54)
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(outcome_names), ival + 1)
            plt.title('Cumulative ' + outcome_name[4:], fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot((np.cumsum(outcome[jval]) / N) * 100, label=vaccine_strategy_names[jval],
                         color=scenario_colors[jval])
            if ival == 0:
                plt.ylabel('Population (%)', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Time (Months)', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
        figname = 'plots/model_outcome_prototype_cumulative_weekly_campaign_' + my_policy.upper() + '.pdf'
        # plt.savefig(figname)
    # plt.show(block=True)


    my_policies = ['status_quo', 'laiv_5_12', 'laiv_5_18']
    laiv_means = {'Pessimistic': total_doses_laiv_main[:, 0] + total_doses_iiv_main[:, 0],
                  'Mid': total_doses_laiv_main[:, 1] + total_doses_iiv_main[:, 1],
                  'Optimistic': total_doses_laiv_main[:, 2] + total_doses_iiv_main[:, 2]}

    iiv_means = {'Pessimistic': total_doses_iiv_main[:, 0],
                 'Mid': total_doses_iiv_main[:, 1],
                 'Optimistic': total_doses_iiv_main[:, 2]}

    x = np.arange(len(my_policies))  # the label locations
    width = 0.2  # the width of the bars
    multiplier = 0
    colors = ['royalblue', 'darkorange', 'green']

    fig, ax = plt.subplots(layout='constrained')

    for attribute, measurement in laiv_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute + ':LAIV', color=colors[multiplier], alpha=0.6)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    multiplier = 0
    for attribute, measurement in iiv_means.items():
        offset = width * multiplier
        rects = ax.bar(x + offset, measurement, width, label=attribute + ':IIV', color=colors[multiplier], alpha=0.8)
        # ax.bar_label(rects, padding=3)
        multiplier += 1

    # Add some text for labels, title and custom x-axis tick labels, etc. (not stacked)
    ax.set_ylabel('Number of doses')
    ax.set_title('Vaccine doses by vaccine type and policy')
    ax.set_xticks(x + width, my_policies)
    ax.legend(loc='upper left', ncols=2)
    ax.set_ylim(0, 330000)

    plt.show(block=True)

    return []


## This function runs vaccination campaign separately for IIV and LAIV.
def run_model_equations_weekly_vaccination_3_laiv(initial_SW=np.zeros(n_ages), amplitude = 0.3, R0 = 1.5, vaccine_uptake = 'pessimistic', vaccine_policy='status_quo',
                                             contact_matrix = np.ones((n_ages, n_ages))):
    ''' LAIV and IIV program run separately here so that LAIV can run during school term (6 weeks)
    vaccination campaign starts in week 10 (corresponding to 7 march)
    this change is implemented in response to Isaac's latest calibration for Wed 3 Sep 2025
    runs the ACEFA model for a specified amplitude, R0, and vaccine uptake (which is further divided into 3
    vaccine policy)

    This function differs from run_model_equations in the following ways:
    - Vaccination runs for 24 weeks starting from March.
    - t0 is March 1 here (i.e. phase_shift = 60 here)

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns a daily time series of (n_states + 3) and n_ages
    '''

    ## varied parameters
    # contact_matrix = load_contact_matrix(populations_by_fractions)
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    ## initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / 27_586_007  ### I CHANGED THIS!!!
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_SW * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of IU's
    initial_population[SW, :] = initial_SW * populations_by_fractions
    # print('Initial population = ', initial_population[:-3, :].sum())

    #### Set vaccine coverage by vaccine uptake and vaccine policy
    ## vaccine_uptake: 'pessimistic' OR 'mid' OR 'optimistic'
    ## vaccine_policy: 'status_quo' OR 'laiv_5_12' OR 'laiv_5_18'

    ## policy: status quo
    vc_laiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_LAIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)
    vc_iiv, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_IIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
    # parameters_outcome = [total_weeks, N, VE_SUS, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## Run model without vaccination for 10 weeks, then weeks 11 to 35 is the campaign, and weeks 36-52 is no vaccination again
    out_main = np.copy(initial_population).reshape((n_states + 3) * n_ages)  ## to be vstacked

    weeks_no_vacc = 12  ## initially no vaccination
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan[0:(weeks_no_vacc*7)],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out))

    old_init = out[-1].reshape((n_states + 3, n_ages))



    ## Set my weekly vaccination schedule (starting from week 10)
    # total_weeks = 52
    vacc_weeks = 24

    cum_weekly_vacc_progress, vacc_dates = get_weekly_vaccination_rates()
    cum_weekly_vacc_progress_laiv, _ = get_weekly_vaccination_rates_LAIV()

    cum_weekly_vaccination_schedule_iiv = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule_iiv = np.zeros((n_ages, vacc_weeks))

    cum_weekly_vaccination_schedule_laiv = np.zeros((n_ages, vacc_weeks + 1))
    new_weekly_vaccination_schedule_laiv = np.zeros((n_ages, vacc_weeks))

    vc_main = np.zeros(9)  ## used to check overall VC at the end


    for a in range(n_ages):
        cum_weekly_vaccination_schedule_iiv[a, 1:] = cum_weekly_vacc_progress * vc_iiv[a]
        new_weekly_vaccination_schedule_iiv[a] = cum_weekly_vaccination_schedule_iiv[a][1:] - \
                                             cum_weekly_vaccination_schedule_iiv[a][:-1]
        cum_weekly_vaccination_schedule_laiv[a, 1:] = cum_weekly_vacc_progress_laiv * vc_laiv[a]
        new_weekly_vaccination_schedule_laiv[a] = cum_weekly_vaccination_schedule_laiv[a][1:] - \
                                             cum_weekly_vaccination_schedule_laiv[a][:-1]


    # initial_population_endemic = np.copy(initial_population)
    # out_main = initial_population_endemic.reshape((n_states + 3) * n_ages)  ## to be vstacked

    # set initial condition with vaccination
    # old_init = out_main.reshape((n_states + 3, n_ages))
    my_t0 = 0  ## time counter
    my_t0 = int(tspan[0:(weeks_no_vacc*7)][-1] + 1)
    total_doses_frac = np.zeros(n_ages)

    for vacc_week in range(vacc_weeks):
        # print(vacc_week, my_t0)
        ## do IIV campaign
        vc_iiv = new_weekly_vaccination_schedule_iiv[:, vacc_week]
        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))  ## needs to be 18,9
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            # frac_to_vaccinate = vc_iiv * old_init[old_state]

            ## Adjust vc_iiv rate with denominator as the eligible unvaccinated S+R population
            unvaccinated_fraction_of_SR = (old_init[(SU, SW, RU, RW), :] * (vc_iiv > 0)).sum() / (populations_by_fractions * (vc_iiv > 0)).sum()
            vaccinated_fraction_of_SR = (old_init[(SV, RV), :] * (vc_iiv > 0)).sum() / (populations_by_fractions * (vc_iiv > 0)).sum()

            n_doses_actual = (populations_by_fractions * N) * vc_iiv  ## length n_ages
            vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * N))

            # do the vaccination with vc_weekly_actual
            frac_to_vaccinate = vc_weekly_actual * old_init[old_state]

            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
            total_doses_frac += frac_to_vaccinate

        ## do LAIV campaign
        vc_laiv = new_weekly_vaccination_schedule_laiv[:, vacc_week]
        if vc_laiv.sum() > 0:
            for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
                # frac_to_vaccinate = vc_laiv * old_init[old_state]

                ## Adjust vc_iiv rate with denominator as the eligible unvaccinated S+R population
                unvaccinated_fraction_of_SR = (old_init[(SU, SW, RU, RW), :] * (vc_laiv > 0)).sum() / (populations_by_fractions * (vc_laiv > 0)).sum()
                vaccinated_fraction_of_SR = (old_init[(SV, RV), :] * (vc_laiv > 0)).sum() / (populations_by_fractions * (vc_laiv > 0)).sum()

                n_doses_actual = (populations_by_fractions * N) * vc_laiv  ## length n_ages
                # vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * (vc_laiv > 0)).sum() * N)
                vc_weekly_actual = n_doses_actual / (unvaccinated_fraction_of_SR * (populations_by_fractions * N))

                # do the vaccination with vc_weekly_actual
                frac_to_vaccinate = vc_weekly_actual * init_new[old_state]

                init_new[old_state] = init_new[old_state] - frac_to_vaccinate
                init_new[new_state] += frac_to_vaccinate
                total_doses_frac += frac_to_vaccinate

        ## run simulation
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                           args=(parameters_ode,), full_output=True)

        ## save my time series
        out_main = np.vstack((out_main, out[1:]))

        ## update initial condition and time span
        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7
        vc_main += (vc_laiv + vc_iiv)
    total_doses = N * total_doses_frac

    ## run to week 52 (end of year)
    out, info = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                       args=(parameters_ode,), full_output=True)

    out_main = np.vstack((out_main, out[1:]))


    return out_main, total_doses

def run_ACEFA_scenarios_with_weekly_vacc_3_schoolterm():
    '''
    here t0 is Jan 1
    '''

    contact_matrix = load_contact_matrix(populations_by_fractions)

    ## fitted values from Wed 3 Sept, 4:47pm
    initial_SW = 0.15
    R0 = 1.258793
    amplitude = 0.061613
    phase_shift = 121.930872
    p_healthseeking = np.array([0.13440617, 0.13440617, 0.13440617, 0.13167023, 0.13167023,  ## mean from flutracking data
                                0.14591527, 0.14591527, 0.16041393, 0.16041393])

    # beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)


    tspan = np.linspace(0, 364 * 1, 364 * 1 + 1)
    total_weeks = int(len(tspan) / 7)

    # my_policies = ['laiv_5_12', 'laiv_5_18']
    my_vaccine_scenarios = ['status_quo', 'pessimistic', 'mid', 'optimistic']

    ## My loop through the vaccine policies and scenarios
    counter = 1
    scenarios_string = 'ABCDEFG'
    colors_colorblind = sns.color_palette("colorblind")
    scenario_colors = ['black', colors_colorblind[-3], colors_colorblind[-5], colors_colorblind[-1]]

    ## policy: status quo
    out_main, total_doses_sq = run_model_equations_weekly_vaccination_3_laiv(initial_SW=initial_SW, amplitude=amplitude, R0=R0,
                                           vaccine_uptake='pessimistic', vaccine_policy='status_quo', contact_matrix=contact_matrix)

    vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy='status_quo', vaccine_scenario='status_quo')
    # vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_IIV(vaccine_policy='status_quo', vaccine_scenario='status_quo')

    parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

    ## track my vaccinated population over time
    z = out_main.reshape((-1, n_states + 3, n_ages))
    z = z[:, 0:n_states, :]
    v = z[:, 5:10, :]

    my_vcolors = ['b', 'r', 'm']
    my_a = [0.5, 0.75, 1]
    # plt.figure(7)
    # plt.suptitle('Vaccinated population over time')
    # plt.plot(v.sum((1, 2)), label='status quo', color='k')
    # print('status quo', '   ', v.sum((1, 2))[-1].round(2))

    ## track all 15 states over time for each scenario
    z = out_main.reshape((-1, n_states + 3, n_ages))
    plt.figure(8, figsize=(14, 8), layout='tight')
    plt.suptitle('Status quo')
    for state in range(n_states):
        x = z[:, state, :].sum(1)
        plt.subplot(3, 5, state + 1)
        plt.title(state_names[state])
        plt.plot(x, label='status quo', color='black')
    # plt.show(block=True)

    ## actual vaccine coverage
    x = out_main[-1].reshape((n_states + 3, n_ages))
    x_vacc = x[5:10, :].sum(0) / populations_by_fractions
    print(f"Vaccination coverage by age group")
    print(f"status quo   : V: {x_vacc.round(2)}")


    (weekly_infections_sq, weekly_cases_sq,
     weekly_hosps_sq) = calculate_weekly_outcomes(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                  parameters=parameters_outcome)

    (weekly_infections_age_sq, weekly_cases_age_sq,
     weekly_hosps_age_sq) = calculate_weekly_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                             parameters=parameters_outcome)

    (total_infections_sq, total_cases_sq,
     total_hospitalizations_sq) = calculate_total_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                  parameters=parameters_outcome)


    total_doses_iiv_main = np.zeros((3, 3))  ## nrows = sq, laiv-12, laiv-18 // ncols = pes, mid, opt (so row 1 = sq, sq, sq)
    total_doses_iiv_main[0, :] = total_doses_sq.sum()

    total_doses_laiv_main = np.zeros((3, 3))  ## nrows = sq, laiv-12, laiv-18 // ncols = pes, mid, opt (so row 1 = sq, sq, sq)

    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    for fig_val, my_policy in enumerate(['laiv_5_12', 'laiv_5_18']):

        ## Set up arrays for model outcomes
        weekly_infections_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_cases_main = np.zeros((len(my_vaccine_scenarios), total_weeks))
        weekly_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), total_weeks))

        weekly_infections_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_cases_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))
        weekly_hospitalizations_age_main = np.zeros((len(my_vaccine_scenarios), total_weeks, n_ages))

        total_infections_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_cases_main = np.zeros((len(my_vaccine_scenarios), n_ages))
        total_hospitalizations_main = np.zeros((len(my_vaccine_scenarios), n_ages))

        ## fill in status quo values
        weekly_infections_main[0] = weekly_infections_sq
        weekly_cases_main[0] = weekly_cases_sq
        weekly_hospitalizations_main[0] = weekly_hosps_sq

        weekly_infections_age_main[0] = weekly_infections_age_sq
        weekly_cases_age_main[0] = weekly_cases_age_sq
        weekly_hospitalizations_age_main[0] = weekly_hosps_age_sq

        total_infections_main[0] = total_infections_sq
        total_cases_main[0] = total_cases_sq
        total_hospitalizations_main[0] = total_hospitalizations_sq

        for ival, my_vaccine_scenario in enumerate(['pessimistic', 'mid', 'optimistic']):
            # print('Scenario', scenarios_string[counter], my_policy.upper(), ',', my_vaccine_scenario)
            counter += 1

            out_main, total_doses = run_model_equations_weekly_vaccination_3_schoolterm(initial_SW=initial_SW, amplitude=amplitude, R0=R0,
                                                    vaccine_uptake=my_vaccine_scenario, vaccine_policy=my_policy, contact_matrix=contact_matrix)
            ## update my vaccine parameters
            # vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC(vaccine_policy=my_policy, vaccine_scenario=my_vaccine_scenario)
            vc, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_LAIV(vaccine_policy=my_policy, vaccine_scenario=my_vaccine_scenario)

            # parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VE_SUS, reduced_sus]
            parameters_outcome = [total_weeks, N, VE_GP, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, reduced_sus]

            ## record doses
            total_doses_laiv_main[fig_val + 1, ival] = total_doses[3]
            total_doses_iiv_main[fig_val + 1, ival] = total_doses.sum() - total_doses[3]

            if my_policy == 'laiv_5_18':
                total_doses_laiv_main[fig_val + 1, ival] += total_doses[4]
                total_doses_iiv_main[fig_val + 1, ival] = total_doses.sum() - total_doses_laiv_main[fig_val + 1, ival]

            ## record model outcomes
            (weekly_infections_main[ival + 1], weekly_cases_main[ival + 1],
             weekly_hospitalizations_main[ival + 1]) = calculate_weekly_outcomes(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                                 parameters=parameters_outcome)

            (total_infections_main[ival + 1], total_cases_main[ival + 1],
             total_hospitalizations_main[ival + 1]) = calculate_total_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                                      parameters=parameters_outcome)

            (weekly_infections_age_main[ival + 1], weekly_cases_age_main[ival + 1],
             weekly_hospitalizations_age_main[ival + 1]) = calculate_weekly_outcomes_by_age(y=out_main.reshape((-1, n_states + 3, n_ages)),
                                                                                     parameters=parameters_outcome)

            ## actual vaccine coverage
            x = out_main[-1].reshape((n_states + 3, n_ages))
            x_vacc = x[5:10, :].sum(0) / populations_by_fractions
            print(f"{my_policy} {my_vaccine_scenario[:3]}: V: {x_vacc.round(2)}")

            ## plot all the states

            z = out_main.reshape((-1, n_states + 3, n_ages))
            plt.figure(8, figsize=(14, 8), layout='tight')
            plt.suptitle('My scenarios')
            for state in range(n_states):
                x = z[:, state, :].sum(1)
                plt.subplot(3, 5, state + 1)
                plt.title(state_names[state])
                plt.plot(x, label=my_policy + ' ' + my_vaccine_scenario[:3], color=my_vcolors[fig_val + 1], alpha=my_a[ival])
                if state==0 and my_policy == 'laiv_5_18' and my_vaccine_scenario == 'optimistic': plt.legend()
    plt.show(block=True)

    #     ## Plot
    #     vaccine_strategy_names = my_vaccine_scenarios
    #     outcome_names = ['new infections', 'new cases', 'new hospitalisations']
    #     my_outcomes_weekly = [weekly_infections_main, weekly_cases_main, weekly_hospitalizations_main]
    #     # my_outcomes_total_9 = [total_infections_main, total_cases_main, total_hospitalizations_main]
    #
    #     my_outcomes_total = []  ## convert to 7 acefa age groups
    #     for A9 in [total_infections_main, total_cases_main, total_hospitalizations_main]:
    #         x1 = A9[:, :2].sum(1).reshape((4, 1))
    #         x18 = A9[:, 5:7].sum(1).reshape((4, 1))
    #         A7 = np.hstack((x1, A9[:, 2:5], x18, A9[:, -2:]))
    #         my_outcomes_total.append(A7)
    #
    #     A9 = weekly_cases_age_main
    #     x1 = A9[:, :, :2].sum(-1).reshape(((4, 52, 1)))
    #     x18 = A9[:, :, 5:7].sum((-1)).reshape((4, 52, 1))
    #     print(x1.shape, A9[:, :, 2:5].shape, x18.shape, A9[:, :, -2:].shape)
    #
    #     A7 = np.hstack((x1, A9[:, :, 2:5] , x18, A9[:, :, -2:]))
    #
    #
    #
    #     ### print a table of my outcomes
    #     # w = 8
    #     # ## PRINT OUTCOME TOTALS
    #     print('**************')
    #     print(my_policy.upper())
    #
    #     ## PRINT OUTCOME TOTALS BY TOTAL
    #     w = 8
    #     print('Totals')
    #     print('{message: <{width}}'.format(message='          SQ', width=17),
    #           '{message: <{width}}'.format(message='PES', width=w),
    #           '{message: <{width}}'.format(message='MID', width=w),
    #           '{message: <{width}}'.format(message='OPT', width=w))
    #     for (outcome_name, outcome) in zip(['inf ', 'case', 'hosp'], my_outcomes_total):
    #         x = np.round(outcome.sum(-1), 0)
    #         xd = np.round(outcome.sum(-1) - outcome.sum(-1)[0])
    #         print('  ', outcome_name, x)
    #
    #     ## PRINT OUTCOME TOTALS BY TOTAL
    #     print('Totals by age')
    #     print("        ", "    ".join(acefa_ages))
    #     # for outcome_name in ['inf ', 'case', 'hosp']:
    #
    #     for (outcome_name, outcome) in zip(outcome_names, my_outcomes_total):
    #         x = np.round(outcome, 0)
    #         for jval, scenario_name in enumerate(['SQ ', 'PES', 'MID', 'OPT']):
    #             print('  ', my_policy, scenario_name, x[jval], int(x[jval].sum()) )
    #
    #     months = 'JFMAMJJASONDJ'
    #
    #     plt.figure(1 + fig_val*10, figsize=(13, 3.5), layout='tight')
    #     plt.suptitle('Model outcomes for policy option ' + my_policy.upper(), fontsize=16)
    #     # print(1 + fig_val, my_policy.upper())
    #     # print('outcome_name [status_quo, pessimistic, mid, optimistic]')
    #     # print('outcome_name [outcomes prevented compared to status_quo]')
    #     for fval, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
    #         # print(outcome_name, np.round(outcome.sum(-1), 0))  ## outcome shape: (n_vaccine_scenarios, n_total_weeks)
    #         # print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
    #         plt.subplot(1, len(outcome_names), fval + 1)
    #         plt.title(outcome_name, fontsize=16)
    #         for jval in range( len(outcome) ):
    #             plt.plot(outcome[jval], label=vaccine_strategy_names[jval], color=scenario_colors[jval])
    #         if fval == 0:
    #             plt.ylabel('Individuals', fontsize=16)
    #             # plt.legend(fontsize=11)
    #         plt.xlabel('Time (month)', fontsize=16)
    #         plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
    #         plt.yticks(fontsize=16)
    #         # plt.legend()
    #     figname = 'plots/model_outcome_prototype_timeseries_weekly_campaign_' + my_policy.upper() + '_prelim.pdf'
    #     # plt.savefig(figname)
    #     # plt.show(block=True)
    #
    #     # plt.figure(2 + fig_val)
    #     barWidth = 0.2
    #     bar1 = np.arange(n_ages)
    #     bar2 = [x + barWidth for x in bar1]
    #     bar3 = [x + barWidth for x in bar2]
    #     bar4 = [x + barWidth for x in bar3]
    #     my_bars = [bar1, bar2, bar3, bar4]
    #
    #     barWidth = 0.2
    #     bar1 = np.arange(len(acefa_ages))
    #     bar2 = [x + barWidth for x in bar1]
    #     bar3 = [x + barWidth for x in bar2]
    #     bar4 = [x + barWidth for x in bar3]
    #     my_bars = [bar1, bar2, bar3, bar4]
    #
    #     plt.figure(2 + fig_val*10, figsize=(13, 4), layout='tight')
    #     plt.suptitle(my_policy.upper() + ': Total outcomes by age', fontsize=16)
    #     my_upper = [15000, 1500, 15]
    #     for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
    #         plt.subplot(1, len(outcome_names), ival + 1)
    #         plt.title(outcome_name[4:], fontsize=16)
    #         for jval in range(len(outcome)):
    #             plt.bar(my_bars[jval], outcome[jval], width=barWidth - 0.01, label=vaccine_strategy_names[jval],
    #                     color=scenario_colors[jval])
    #             plt.xticks(bar2, '')
    #             # if ival == 0:
    #                 # plt.legend(loc='upper left', fontsize=11)
    #             plt.xlabel('Age groups', fontsize=16)
    #             plt.xticks([r + barWidth for r in range(len(acefa_ages))], acefa_ages, rotation=45, fontsize=14)
    #             # plt.ylabel('Individuals', fontsize=16)
    #             plt.tick_params('both', length=6, width=1.5)
    #             plt.yticks(fontsize=18)
    #             # plt.ylim([0, my_upper[ival]])
    #     figname = 'plots/model_outcome_prototype_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '_prelim.pdf'
    #     # plt.savefig(figname)
    #     # plt.show(block=True)
    #
    #     ### horizontal 26/8/25
    #     ### Figure 2 (total outcomes by age) (1x3) RELATIVE CHANGE TO STATUS QUO
    #     plt.figure(3 + fig_val * 10, figsize=(13, 4), layout='tight')
    #     plt.suptitle(my_policy.upper() + ': Outcomes from policy divided by outcomes from status quo', fontsize=16)
    #     for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
    #         plt.subplot(1, len(outcome_names), ival + 1)
    #         plt.title(outcome_name[4:], fontsize=16)
    #         for jval in range(len(outcome)):
    #             # print(jval)
    #             relative_outcome = (outcome[0,:] - outcome[jval]) / outcome[0,:]
    #             # plt.bar(my_bars[jval], relative_outcome * 100, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
    #             #         color=scenario_colors[jval])
    #             if jval >0: ## so I don't plot the status quo
    #                 plt.bar(my_bars[jval], outcome[jval] / outcome[0,:], width=barWidth - 0.01, label=vaccine_strategy_names[jval],
    #                         color=scenario_colors[jval])
    #             plt.xticks(bar2, '')
    #             if ival == 0:
    #                 # plt.legend(loc='upper left', fontsize=11)
    #                 plt.ylabel('% prevented (relative)', fontsize=16)
    #             plt.xlabel('Age groups', fontsize=16)
    #             plt.xticks([r + barWidth for r in range(len(acefa_ages))], acefa_ages, rotation=45, fontsize=18)
    #             plt.tick_params('both', length=6, width=1.5)
    #             # plt.ylim([0, 55])
    #             plt.yticks(fontsize=18)
    #     figname = 'plots/model_outcome_prototype_by_age_and_scenario_with_ci_' + my_policy.upper() + '_horizontal_prelim_relative2.pdf'
    #     plt.savefig(figname)
    # # plt.show(block=True)
    #
    #     plt.figure(5 + fig_val*10, figsize=(6, 8), layout='tight')
    #     plt.suptitle(my_policy.upper() + ': Total outcomes by age (per thousand)', fontsize=16)
    #     pop1 = populations_by_fractions[:2].sum()
    #     pop18 = populations_by_fractions[5:7].sum()
    #     populations_by_fractions7 = np.hstack((pop1, populations_by_fractions[2:5], pop18, populations_by_fractions[-2:]))
    #     for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
    #         plt.subplot(len(outcome_names), 1, ival + 1)
    #         plt.title(outcome_name[4:], fontsize=16)
    #         ## print the outcomes
    #         # print(2 + fig_val, my_policy.upper())
    #         # print(' ', outcome_name)
    #         ## change from 9 to 7 age groups aqui
    #         # x1 = outcome[:, :2].sum(1).reshape((4, 1))
    #         # x18 = outcome[:, 5:7].sum(1).reshape((4, 1))
    #         # outcome_7 = np.hstack((x1, outcome[:, 2:5], x18, outcome[:, -2:]))
    #         for jval in range(len(outcome)):
    #             x = outcome[jval] / (populations_by_fractions7 * N) * 1000
    #             plt.bar(my_bars[jval], x, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
    #                     color=scenario_colors[jval])
    #             plt.xticks(bar2, '')
    #             if ival == 0:
    #                 plt.legend(loc='upper left', fontsize=11)
    #             if ival == 2:
    #                 plt.xlabel('Age groups', fontsize=16)
    #                 plt.xticks([r + barWidth for r in range(len(acefa_ages))], acefa_ages, rotation=45, fontsize=14)
    #             plt.ylabel('Individuals', fontsize=16)
    #             plt.yticks(fontsize=16)
    #     figname = 'plots/model_outcome_prototype_per_1000_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '_prelim.pdf'
    #     # plt.savefig(figname)
    #     # plt.show(block=True)
    #
    #     plt.figure(4 + fig_val*10, figsize=(6, 8), layout='tight')
    #     ymaxes = [10000, 3200, 3.1]  ## currently manually set because I'm lazy
    #     plt.suptitle(my_policy.upper() + ': Outcomes prevented by age', fontsize=16)
    #     ## create my status quo infections/cases/hospitalisations
    #     outcomes_prevented = np.zeros((len(outcome_names), len(vaccine_strategy_names), len(acefa_ages)))
    #     outcomes_status_quo = np.zeros((len(outcome_names), len(acefa_ages)))
    #     for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
    #         outcomes_status_quo[ival] = outcome[0]
    #         # print(np.round(outcome[0],0))  ## outcomes for status quo
    #         #
    #         plt.subplot(len(outcome_names), 1, ival + 1)
    #         plt.title(outcome_name, fontsize=16)
    #
    #         for jval in range(len(outcome)):
    #             outcome_prevented = outcome[jval] - outcomes_status_quo[ival]
    #             outcomes_prevented[ival, jval, :] = outcome_prevented
    #             plt.bar(my_bars[jval], -outcome_prevented, width=barWidth - 0.01, label=vaccine_strategy_names[jval],
    #                     color=scenario_colors[jval])
    #             plt.xticks(bar2, '')
    #             if ival == 0:
    #                 plt.legend(loc='upper left', fontsize=11)
    #             if ival == 2:
    #                 plt.xlabel('Age groups', fontsize=16)
    #                 plt.xticks([r + barWidth for r in range(len(acefa_ages))], acefa_ages, rotation=45, fontsize=14)
    #             plt.ylabel('Individuals', fontsize=16)
    #             plt.yticks(fontsize=16)
    #             # plt.ylim([0, ymaxes[ival]])
    #     figname = 'plots/model_outcomes_prevented_prototype_by_age_and_scenario_weekly_campaign_' + my_policy.upper() + '_prelim.pdf'
    #     # plt.savefig(figname)
    #     # plt.show(block=True)
    #
    #     plt.figure(3 + fig_val*10, figsize=(13, 4), layout='tight')
    #     plt.suptitle(my_policy.upper() + ': Cumulative outcomes by scenario', fontsize=16)
    #     for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
    #         # print('Cumulative', outcome_name[4:], np.round(outcome.sum(-1), 0))  ## outcome.shape (n_vaccine_scenarios, n_total_weeks) = (4, 54)
    #         # print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
    #         plt.subplot(1, len(outcome_names), ival + 1)
    #         plt.title('Cumulative ' + outcome_name[4:], fontsize=16)
    #         for jval in range( len(outcome) ):
    #             plt.plot((np.cumsum(outcome[jval]) / N) * 100, label=vaccine_strategy_names[jval],
    #                      color=scenario_colors[jval])
    #         if ival == 0:
    #             plt.ylabel('Population (%)', fontsize=16)
    #             plt.legend(fontsize=11)
    #         plt.xlabel('Time (Month)', fontsize=16)
    #         plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
    #         plt.yticks(fontsize=16)
    #     figname = 'plots/model_outcome_prototype_cumulative_weekly_campaign_' + my_policy.upper() + '_prelim.pdf'
    #     # plt.savefig(figname)

    return []


def plot_vaccine_strategies(VC_seasonal=np.array([0, 0.276, 0.276, 0.143, 0.166, 0.212, 0.333, 0.617, 0.617]),
                            VC_LAIV_5_12=np.array([0, 0.276, 0.276, 0.52, 0.166, 0.212, 0.333, 0.617, 0.617]),
                            VC_LAIV_5_18=np.array([0, 0.276, 0.276, 0.52, 0.52, 0.212, 0.333, 0.617, 0.617])):
    colors = sns.color_palette("colorblind")
    n_ages = 9

    ## Plot of vaccine coverage by vaccine strategy
    VC_S12 = np.copy(VC_seasonal)
    VC_S12[3] = 0

    VC_LAIV12 = np.zeros(n_ages)
    VC_LAIV12[3] = VC_LAIV_5_12[3]

    VC_LAIV18 = np.copy(VC_LAIV12)
    VC_LAIV18[3:5] = VC_LAIV_5_18[3:5]

    # barWidth = 0.25
    bar1 = np.arange(len(VC_seasonal))
    # bar2 = [x + barWidth for x in bar1]
    # bar3 = [x + barWidth for x in bar2]

    ## This disappears the plot box lines
    plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})

    ## FIGURE OF VACCINE POLICY OPTIONS
    # plt.figure(figsize=(2.5, 3), layout='tight')  ## vertical
    plt.figure(figsize=(3.5, 4), layout='tight')  ## vertical
    # plt.suptitle('Vaccine Policy Options')
    plt.subplot(3, 1, 1)
    plt.title('status quo: seasonal only', fontsize=8)
    plt.bar(bar1, VC_seasonal * 100, color=colors[-1], label='IIV')
    plt.bar(bar1, VC_seasonal * 0, color=colors[-2], label='LAIV')
    plt.legend(loc='lower right', ncol=1, fontsize=6)
    plt.xticks(bar1, '')
    plt.subplot(3, 1, 2)
    plt.title('seasonal + LAIV 5-12', fontsize=8)
    plt.bar(bar1, VC_seasonal * 100, color=colors[-1], label='IIV')
    plt.bar(bar1, VC_LAIV12 * 100, color=colors[-2], label='LAIV')
    plt.xticks(bar1, '')
    plt.ylabel('Vaccine coverage (%)', fontsize=8)
    plt.subplot(3, 1, 3)
    plt.title('seasonal + LAIV 5-18', fontsize=8)
    plt.bar(bar1, VC_seasonal * 100, color=colors[-1], label='IIV')
    plt.bar(bar1, VC_LAIV18 * 100, color=colors[-2], label='LAIV')
    plt.xticks(bar1, age_names, rotation=45, fontsize=8)
    plt.xlabel('Age groups (years)', fontsize=8)
    # plt.savefig('plots/vaccine_coverages.pdf')
    plt.show(block=True)

    return []