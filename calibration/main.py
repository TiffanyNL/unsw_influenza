## ACEFA Scenario Modelling: Team UNSW

import numpy as np
from scipy.integrate import odeint
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from acefa_functions import *
from plotnine import *

## state abbreviations
[SU, E1U, E2U, IU, RU, SV, E1V, E2V, IV, RV, SW, E1W, E2W, IW, RW, CaseIU, CaseIV, CaseIW] = range(18)
state_names = ['SU', 'E1U', 'E2U', 'IU', 'RU', 'SV', 'E1V', 'E2V', 'IV', 'RV', 'SW', 'E1W', 'E2W', 'IW', 'RW']
age_names = ['0-5M', '6-11M', '1-4', '5-11', '12-17', '18-49', '50-64', '65-79', '80+']
colors = sns.color_palette("colorblind")

if __name__ == '__main__':
    ## fixed parameters
    n_ages = 9
    n_states = 15
    N = 1_000_000
    amplitude = 0.3
    phase_shift = 120  # to be changed after we look at data
    populations_by_fractions = np.array(load_population_australia()['proportion'])
    contact_matrix = load_contact_matrix(populations_by_fractions)
    sigma = 1 / 1.2  ## latent period (Longini2005)
    gamma = 1 / 4.1  ## infectious period (Longini2005)
    omega = 1 / 120 ## duration of immunity (Tenforde2024b)

    ## Influenza epidemiological parameters (age-structured)
    p_asymptomatic = 0.19 * np.ones(n_ages) # Furuya-Kanamori2016
    p_healthseeking = np.array([0.296, 0.296, 0.296, 0.296, 0.296, 0.208, 0.219, 0.277, 0.277]) ##Baltrusaitis2022; 0.25 overall (Cheng2024)
    p_hospitalised = load_hospitalisation_rates() / 100_000
    rel_w = 0.3  ## relative protection for "waning" classes

    VESUS_TIV = np.array([0.6, 0.6, 0.6, 0.6, 0.6, 0.37, 0.37, 0.18, 0.18])  ## VE against any influenza (Frutos2025)
    VESUS_LAIV = np.array([0.8, 0.8, 0.8, 0.8, 0.8, 0.4, 0.4, 0.4, 0.4])  ## against matched strains, Mohn2018
    VE_HOSP = np.array([0., 0.71, 0.71, 0.49, 0.49, 0.51, 0.40, 0.36, 0.36])  ## TIV, Tenforde2024b
    VE_SYMP = np.zeros(n_ages)
    VE_healthseeking = np.array([0.32, 0.32, 0.32, 0.32, 0.32, 0.36, 0.36, 0.36, 0.36])  ## against medically attended influenza (Frutos2025)

    VESUS = np.copy(VESUS_TIV)
    VESUS[:2] = VESUS_LAIV[0:2]  ## VESUS for LAIV for 5-12YO, and for seasonal vaccine for other ages
    VEW = rel_w * VESUS  ## protection against infection from preexisting waning immunity

    ## varied parameters
    R0 = 1.5
    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    tspan = np.linspace(0, 364 * 1, 364 * 1 + 1)
    total_weeks = int(tspan[-1] / 7)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages, n_states, sigma, VESUS, VEW]
    parameters_outcome = [total_weeks, N, VE_SYMP, VE_healthseeking, VE_HOSP, p_asymptomatic, p_healthseeking, p_hospitalised, rel_w]

    #### Set initial condition
    initial_population = np.zeros((n_states + 3, n_ages))
    initial_i = 100 / 1_000_000
    initial_RU = np.repeat(0.1, 9)
    initial_population[SU, :] = np.ones(n_ages) * populations_by_fractions - initial_RU * populations_by_fractions
    initial_population[IU, 5] = initial_i
    initial_population[SU, 5] -= initial_i
    initial_population[RU, :] = initial_RU * populations_by_fractions
    initial_population[CaseIU, :] = initial_population[IU, :]  ## number of I's
    print('Initial population sums to: ', initial_population[:-3, :].sum())

    ##### Define vaccine policy options: Status quo, LAIV to 5-12 YOs, LAIV to 5-19 YOs #####
    VC_seasonal = np.array([0, 0.276, 0.276, 0.143, (0.143) * 4 / 6 + (0.212) * 2 / 6, 0.212, 0.333, 0.617, 0.617])  ## NCIRS for 2024
    VC_LAIV_5_12, VC_LAIV_5_18 = np.copy(VC_seasonal), np.copy(VC_seasonal)
    VC_LAIV_5_12[3] = 0.52
    VC_LAIV_5_18[3:5] = 0.52

    vaccine_strategies = [VC_seasonal, VC_LAIV_5_12, VC_LAIV_5_18]

    ##### Calculate model outcomes

    ## Set up arrays for model outcomes
    weekly_infections_main = np.zeros((len(vaccine_strategies), total_weeks))
    weekly_cases_main = np.zeros((len(vaccine_strategies), total_weeks))
    weekly_hospitalizations_main = np.zeros((len(vaccine_strategies), total_weeks))

    total_infections_main = np.zeros((len(vaccine_strategies), n_ages))
    total_cases_main = np.zeros((len(vaccine_strategies), n_ages))
    total_hospitalizations_main = np.zeros((len(vaccine_strategies), n_ages))

    ## transient run of 5 years with a fully susceptible population:
    tspan_transient = np.linspace(0, 364 * 5, 364 * 5 + 1)
    out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan_transient,
                       args=(parameters_ode,), full_output=True)


    initial_population_endemic = out[-1].reshape((n_states + 3, n_ages))
    print('yinit after burn-in'
          '\n  Unvaccinated SEEIR: ', np.round(initial_population_endemic[0:5, :].sum(1), 3),
          '\n    Vaccinated SEEIR: ', np.round(initial_population_endemic[5:10, :].sum(1), 3),
          '\n        Waning SEEIR: ', np.round(initial_population_endemic[10:15, :].sum(1), 3))

    # initial_population_endemic = np.copy(initial_population)

    ## Loop through the 3 vaccine strategies
    for ival, vc in enumerate(vaccine_strategies):
        # set initial condition with vaccination
        init_new = np.copy(initial_population_endemic)
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            frac_to_vaccinate = vc * initial_population_endemic[old_state]
            init_new[old_state] = initial_population_endemic[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate
        # print('%1f Total--Unvaccinated: %.2f, Vaccinated: %.2f, Waning: %.2f' %
        #       (init_new[:15, :].sum(), init_new[0:5, :].sum(), init_new[5:10, :].sum(), init_new[10:15, :].sum()) )

        # run the model
        out, info = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan,
                           args=(parameters_ode,), full_output=True)

        (weekly_infections_main[ival], weekly_cases_main[ival],
         weekly_hospitalizations_main[ival]) = calculate_weekly_outcomes(y=out.reshape((-1, n_states + 3, n_ages)),
                                                                         parameters=parameters_outcome)

        (total_infections_main[ival], total_cases_main[ival],
         total_hospitalizations_main[ival]) = calculate_total_outcomes_by_age(y=out.reshape((-1, n_states + 3, n_ages)),
                                                                              parameters=parameters_outcome)


    ############################################################################################################
    ############################################################################################################
    ###############                                 PLOTS                                #######################
    ############################################################################################################
    ############################################################################################################

    make_plots = False

    if make_plots:
        plot_vaccine_strategies()

        vaccine_strategy_names = ['IIV only', 'IIV + LAIV 5-12', 'IIV + LAIV 5-18']
        outcome_names = ['new infections', 'new cases', 'new hospitalisations']
        my_outcomes_weekly = [weekly_infections_main, weekly_cases_main, weekly_hospitalizations_main]
        my_outcomes_total = [total_infections_main, total_cases_main, total_hospitalizations_main]
        months = ['May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
        months = 'MJJASONDJFMAM'

        plt.figure(figsize=(13, 4), layout='tight')
        plt.suptitle('Prototype model outcomes after 5-year transient from S(0) = 1: R0 = 1.5, Phase shift = 120 d', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print(outcome_name, np.round(outcome.sum(-1), 0))
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(vaccine_strategies), ival + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot(outcome[jval], label=vaccine_strategy_names[jval])
            if ival == 0:
                plt.ylabel('Individuals', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Weeks', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
            # plt.legend()
        plt.show(block=True)

        barWidth = 0.20
        bar1 = np.arange(n_ages)
        bar2 = [x + barWidth for x in bar1]
        bar3 = [x + barWidth for x in bar2]
        my_bars = [bar1, bar2, bar3]

        plt.figure(figsize=(11, 8), layout='tight')
        plt.suptitle('Total outcomes by age', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(vaccine_strategies), 1, ival + 1)
            plt.title(outcome_name, fontsize=16)
            for jval in range(len(outcome)):
                plt.bar(my_bars[jval], outcome[jval], width=barWidth-0.01, label=vaccine_strategy_names[jval])
                plt.xticks([])
                plt.legend(loc='upper left', fontsize=11)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=16)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=14)
                plt.ylabel('Individuals', fontsize=16)
                plt.yticks(fontsize=16)
        # plt.savefig('plots/model_outcome_prototype_by_age.pdf')
        plt.show(block=True)

        plt.figure(figsize=(3, 5), layout='tight')  ## vertical
        # plt.suptitle('Total outcomes by age', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(len(vaccine_strategies), 1, ival + 1)
            plt.title('Total ' + outcome_name[4:], fontsize=10)
            for jval in range(len(outcome)):
                plt.bar(my_bars[jval], outcome[jval], width=barWidth - 0.01, label=vaccine_strategy_names[jval])
                plt.xticks([r + barWidth for r in range(n_ages)], [], rotation=45, fontsize=10)
                if ival == 0:
                    plt.legend(loc='center left', fontsize=8)
                if ival == 2:
                    plt.xlabel('Age groups', fontsize=10)
                    plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=10)
                plt.ylabel('Individuals', fontsize=10)
                plt.yticks(fontsize=10)
        # plt.savefig('plots/model_outcome_prototype_by_age_vertical.pdf')
        plt.show(block=True)

        plt.figure(figsize=(13, 4), layout='tight')  ## horizontal
        # plt.suptitle('Total outcomes by age', fontsize=16)
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_total)):
            plt.subplot(1, len(vaccine_strategies), ival + 1)
            plt.title('Total ' + outcome_name[4:], fontsize=16)
            for jval in range(len(outcome)):
                plt.bar(my_bars[jval], outcome[jval] / N, width=barWidth - 0.01, label=vaccine_strategy_names[jval])
                plt.xticks([r + barWidth for r in range(n_ages)], [], rotation=45, fontsize=16)
                plt.xlabel('Age groups (years)', fontsize=16)
                plt.xticks([r + barWidth for r in range(n_ages)], age_names, rotation=45, fontsize=16)
                plt.yticks(fontsize=16)
                if ival == 0:
                    plt.ylabel('Population (%)', fontsize=16)
                    plt.legend(loc='upper left', fontsize=11)
        # plt.savefig('plots/model_outcome_prototype_by_age_horizontal.pdf')
        plt.show(block=True)


        plt.figure(figsize=(13, 4), layout='tight')
        for ival, (outcome_name, outcome) in enumerate(zip(outcome_names, my_outcomes_weekly)):
            print(outcome_name, np.round(outcome.sum(-1), 0))
            print(outcome_name, np.round(outcome.sum(-1) - outcome.sum(-1)[0]))
            plt.subplot(1, len(vaccine_strategies), ival + 1)
            plt.title('Cumulative ' + outcome_name[4:], fontsize=16)
            for jval in range( len(outcome) ):
                plt.plot(np.cumsum(outcome[jval]) / N, label=vaccine_strategy_names[jval])
            if ival == 0:
                plt.ylabel('Population (%)', fontsize=16)
                plt.legend(fontsize=11)
            plt.xlabel('Time (Months)', fontsize=16)
            plt.xticks(np.linspace(0, total_weeks, int(total_weeks // 4.33 + 1)), months, fontsize=16)
            plt.yticks(fontsize=16)
        # plt.savefig('plots/model_outcome_prototype_cumulative.pdf')
        plt.show(block=True)



        ## Plot of individual states in the model

        ## run the model
        out, info = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages), tspan,
                           args=(parameters_ode,), full_output=True)

        ## reshape the simulation output
        y = out.reshape((-1, n_states + 3, n_ages))

        ## calculate the weekly outcomes
        weekly_infections, weekly_cases, weekly_hospitalizations = calculate_weekly_outcomes(y, parameters_outcome)
        print('Total: %d infections, %d cases, %d hospitalizations over %d weeks'
              % (weekly_infections.sum(), weekly_cases.sum(), weekly_hospitalizations.sum(), total_weeks))

        s_states = [SU, SV, SW]
        e_states = [E1U, E2U, E1V, E2V, E1W, E2W]
        i_states = [IU, IV, IW]
        r_states = [RU, RV, RW]

        plt.figure(2, figsize=(13, 3), layout='tight')
        plt.suptitle('SEIR states')
        for ival, (state_name, disease_states) in enumerate(zip(['S', 'E', 'I', 'R'], [s_states, e_states, i_states, r_states])):
            # print(ival, state_name, disease_states)
            my_sum = np.zeros(len(y))
            for state in disease_states:
                my_sum += y[:, state].sum(-1)
            print(state_name, my_sum[-1])
            plt.subplot(1, 4, ival + 1)
            plt.title(state_name)
            plt.plot(tspan, my_sum)
            # plt.rcParams.update({'axes.spines.top': False, 'axes.spines.right': False})
            plt.xlabel('Time (days)')
        plt.show(block=True)

        plt.figure(1, figsize=(13, 8), layout='tight')
        plt.suptitle('Time series of each state')
        for state, name in enumerate(state_names):
            plt.subplot(3, 5, state + 1)
            plt.title(name)
            plt.plot(tspan, np.sum(y[:, state], -1), label=name)
        plt.show(block=True)