### Scenario analysis for the age-structured influenza vaccination model.

import numpy as np
import pandas as pd
from scipy.integrate import odeint
import matplotlib
matplotlib.use('TkAgg')
import pickle
import time
import matplotlib.pyplot as plt
import seaborn as sns

from default_params_unsw_new import *
from functions_unsw_new import *

def run_model_equations_weekly_vaccination_3_laiv12_multirun(initial_population, amplitude = 0.3, R0 = 1.5,
                                                             vaccine_uptake = 'mid', vaccine_policy='status_quo',
                                                             contact_matrix = np.ones((9, 9))):
    '''
    Runs the flu model with weekly IIV and LAIV vaccination.

    LAIV and IIV rollouts are run over 12 and 24 weeks respectively.

    Runs the flu model for a specified amplitude, R0, and vaccine uptake

    vaccine_uptake :: 'pessimistic' OR 'mid' OR 'optimistic'
    vaccine_policy :: 'status_quo' OR 'iiv_65' OR 'iiv_50' OR 'laiv_5_12' OR 'laiv_5_18'

    Returns
    -------
    out_main : ndarray
        Daily model states, shape (365, (n_states + 3) * n_ages).
    total_doses : ndarray
        Total vaccine doses administered by age group.
    daily_iiv_doses : ndarray
        Daily IIV doses by age group.
    daily_laiv_doses : ndarray
        Daily LAIV doses by age group.
    doses_by_compartment : list
        IIV and LAIV doses administered to susceptible and recovered individuals.
    '''

    beta = findBeta(R0, contact_matrix, gamma, 0, populations_by_fractions)

    ## Vaccine effectiveness: fixed central values
    _, VE_SUS, VE_GP, VE_HOSP = load_VE_and_VC_LAIV(vaccine_policy=vaccine_policy, vaccine_scenario='mid')

    ## Vaccine coverage: varies by policy and vaccine uptake
    vc_laiv, _, _, _ = load_VE_and_VC_LAIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)
    vc_iiv,  _, _, _ = load_VE_and_VC_IIV(vaccine_policy=vaccine_policy, vaccine_scenario=vaccine_uptake)

    parameters_ode = [amplitude, beta, contact_matrix, gamma, omega, phase_shift, n_ages,
                      n_states, sigma, VE_SUS, reduced_sus]

    ## Simulation settings
    weeks_no_vacc = 12
    # vacc_weeks = 24
    n_days = 52 * 7

    initial_population = np.asarray(initial_population).reshape((n_states + 3) * n_ages)

    ## Run the model before vaccination begins
    out = odeint(model_equations_seir, initial_population.reshape((n_states + 3) * n_ages),
                 tspan[0:(weeks_no_vacc * 7)], args=(parameters_ode,))

    out_main = np.vstack((initial_population, out))
    old_init = out[-1].reshape((n_states + 3, n_ages))

    ## Weekly vaccination schedule
    cum_progress_iiv, cum_progress_laiv = get_weekly_vaccination_rates()

    cum_schedule_iiv = vc_iiv[:, None] * np.concatenate( ([0], cum_progress_iiv) )[None, :]
    cum_schedule_laiv = vc_laiv[:, None] * np.concatenate( ([0], cum_progress_laiv) )[None, :]

    weekly_schedule_iiv = np.diff(cum_schedule_iiv, axis=1)  ## cum_weekly_vaccination_schedule_iiv
    weekly_schedule_laiv = np.diff(cum_schedule_laiv, axis=1)  ## cum_weekly_vaccination_schedule_laiv

    # campaign data
    weeks_no_vacc = 12  ## initially no vaccination
    vacc_weeks = 24

    # set vaccination outputs
    my_t0 = int(tspan[0:(weeks_no_vacc*7)][-1] + 1)

    iiv_doses_given_to_S = np.zeros((vacc_weeks, n_ages))
    iiv_doses_given_to_R = np.zeros((vacc_weeks, n_ages))
    laiv_doses_given_to_S = np.zeros((vacc_weeks, n_ages))
    laiv_doses_given_to_R = np.zeros((vacc_weeks, n_ages))

    total_doses_frac = np.zeros(n_ages)

    daily_iiv_doses = np.zeros((n_days, n_ages))
    daily_laiv_doses = np.zeros((n_days, n_ages))

    iiv_doses_to_S = np.zeros((vacc_weeks, n_ages))
    iiv_doses_to_R = np.zeros((vacc_weeks, n_ages))
    laiv_doses_to_S = np.zeros((vacc_weeks, n_ages))
    laiv_doses_to_R = np.zeros((vacc_weeks, n_ages))


    for vacc_week in range(vacc_weeks):

        init_new = np.copy(old_init.reshape((n_states + 3, n_ages)))

        ## do IIV campaign
        vc_iiv = weekly_schedule_iiv[:, vacc_week]

        eligible_ages = vc_iiv > 0
        eligible_states = old_init[(SU, SW, RU, RW), :]
        susceptible_states = old_init[(SU, SW), :]
        recovered_states = old_init[(RU, RW), :]

        eligible_fraction = safe_divide((eligible_states * eligible_ages).sum(),
                                        (populations_by_fractions * eligible_ages).sum())

        susceptible_fraction = safe_divide((susceptible_states * eligible_ages).sum(),
                                           (eligible_states * eligible_ages).sum())

        recovered_fraction = safe_divide((recovered_states * eligible_ages).sum(),
                                         (eligible_states * eligible_ages).sum())

        iiv_doses = populations_by_fractions * vc_iiv

        iiv_doses_to_S[vacc_week] = susceptible_fraction * iiv_doses
        iiv_doses_to_R[vacc_week] = recovered_fraction * iiv_doses

        ## transform vc_iiv into vc_weekly_actual, the VC % of only the eligible population (those in S and R in eligible ages)
        iiv_doses_absolute = (populations_by_fractions * N) * vc_iiv  ## length n_ages
        vc_weekly_actual = safe_divide( iiv_doses_absolute, (eligible_fraction * (populations_by_fractions * N)) )

        ## record the iiv doses given to susceptibles and recovereds
        iiv_doses_given_to_S[vacc_week, :] = susceptible_fraction * iiv_doses_absolute
        iiv_doses_given_to_R[vacc_week, :] = recovered_fraction * iiv_doses_absolute

        ## Move eligible fractions to vaccinated compartments
        for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
            frac_to_vaccinate = vc_weekly_actual * old_init[old_state]

            init_new[old_state] = old_init[old_state] - frac_to_vaccinate
            init_new[new_state] += frac_to_vaccinate

        total_doses_frac += vc_iiv

        ## track doses by day and by age
        dose_day = (vacc_week + 1) * 7 + weeks_no_vacc * 7 - 1
        daily_iiv_doses[dose_day, :] = iiv_doses_absolute

        ## do LAIV campaign
        vc_laiv = weekly_schedule_laiv[:, vacc_week]

        if vc_laiv.sum() > 0:

            eligible_ages = vc_laiv > 0

            eligible_states = old_init[(SU, SW, RU, RW), :]
            susceptible_states = old_init[(SU, SW), :]
            recovered_states = old_init[(RU, RW), :]

            eligible_fraction = safe_divide((eligible_states * eligible_ages).sum(),
                                            (populations_by_fractions * eligible_ages).sum())

            susceptible_fraction = safe_divide((susceptible_states * eligible_ages).sum(),
                                               (eligible_states * eligible_ages).sum()
                                               )
            recovered_fraction = safe_divide((recovered_states * eligible_ages).sum(),
                                             (eligible_states * eligible_ages).sum()
                                             )
            laiv_doses_absolute = (populations_by_fractions * N) * vc_laiv  ## length n_ages
            vc_weekly_actual = safe_divide( laiv_doses_absolute, (eligible_fraction * (populations_by_fractions * N)) )

            ## record the iiv doses given to susceptibles and recovereds
            laiv_doses_given_to_S[vacc_week, :] = susceptible_fraction * laiv_doses_absolute
            laiv_doses_given_to_R[vacc_week, :] = recovered_fraction * laiv_doses_absolute


            for old_state, new_state in zip([SU, SW, RU, RW], [SV, SV, RV, RV]):
                frac_to_vaccinate = vc_weekly_actual * init_new[old_state]

                init_new[old_state] = init_new[old_state] - frac_to_vaccinate
                init_new[new_state] += frac_to_vaccinate

            total_doses_frac += vc_laiv

            ## track doses by week and by age (24 x 9)
            dose_day = (vacc_week + 1) * 7 + weeks_no_vacc * 7 - 1
            daily_laiv_doses[dose_day, :] = laiv_doses_absolute

        ## run simulation
        out = odeint(model_equations_seir, init_new.reshape((n_states + 3) * n_ages), tspan[my_t0:my_t0 + 8],
                     args=(parameters_ode,))

        out_main = np.vstack((out_main, out[1:]))

        old_init = out[-1].reshape((n_states + 3, n_ages))
        my_t0 += 7

    total_doses = total_doses_frac * (populations_by_fractions * N)

    ## run to week 52 (end of year)
    out = odeint(model_equations_seir, old_init.reshape((n_states + 3) * n_ages), tspan[my_t0:],
                 args=(parameters_ode,))

    out_main = np.vstack((out_main, out[1:]))

    doses_iiv = np.array([iiv_doses_given_to_S.sum(0), iiv_doses_given_to_R.sum(0)])
    doses_laiv = np.array([laiv_doses_given_to_S.sum(0), laiv_doses_given_to_R.sum(0)])

    return out_main, total_doses, daily_iiv_doses, daily_laiv_doses, [doses_iiv, doses_laiv]


def plot_weekly_and_cumulative_outcomes(weekly_outcome_arrays, policies, scenarios, total_weeks=52):
    """

    To be called within run_ACEFA_scenarios_with_weekly_vacc_3_multirun()

    Plot weekly and cumulative outcomes for each vaccination policy.

    Individual trajectories are plotted with transparency.
    """

    outcome_names = ["Infections", "Cases", "Hospitalisations", "Deaths"]

    scenario_labels = {"status_quo": "Baseline", "pessimistic": "Pessimistic", "mid": "Central", "optimistic": "Optimistic"}

    colors = sns.color_palette("colorblind")

    scenario_colors = {"status_quo": "black",
                       "pessimistic": colors[-3],
                       "mid": colors[-5],
                       "optimistic": colors[-1]}

    weeks = np.arange(1, total_weeks + 1)

    for policy_idx, policy in enumerate(policies):

        for cumulative in (False, True):

            fig, axes = plt.subplots(1, 4, figsize=(15, 3.5), layout="tight")

            plot_type = "Cumulative" if cumulative else "Weekly"

            fig.suptitle(f"{policy.upper()}: {plot_type} outcomes", fontsize=16)

            for outcome_idx, (outcome_name, weekly_meta) in enumerate(zip(outcome_names, weekly_outcome_arrays)):

                ax = axes[outcome_idx]

                # Shape: (n_sims, n_scenarios, total_weeks)
                policy_outcomes = weekly_meta[:, policy_idx, :, :]

                for scenario_idx, scenario in enumerate(scenarios):

                    # Skip because IIV 65 central is the same as baseline
                    if policy == "iiv_65" and scenario == "mid":
                        continue

                    trajectories = policy_outcomes[:, scenario_idx, :]
                    if cumulative:
                        trajectories = np.cumsum(trajectories, axis=1)

                    # Plot individual trajectories
                    for sim_idx, trajectory in enumerate(trajectories):
                        ax.plot(weeks, trajectory,
                                color=scenario_colors[scenario],
                                linewidth=0.5,
                                alpha=1.0 if scenario == "status_quo" else 0.33,
                                label=(scenario_labels[scenario] if sim_idx == 0 else None)
                                )

                ax.set_title(outcome_name, fontsize=14)
                ax.set_xlabel("Week of year", fontsize=12)
                ax.set_xlim(1, total_weeks)
                ax.set_xticks([1, 13, 26, 39, 52])

                ax.spines[["top", "right"]].set_visible(False)

                if outcome_idx == 0:
                    ax.set_ylabel("Cumulative individuals" if cumulative else "Individuals per week", fontsize=12)

            # Add legend
            axes[0].legend(loc="upper left", fontsize=10, frameon=False)
    plt.show()

    return


def run_ACEFA_scenarios_with_weekly_vacc_3_multirun():
    """
    Run the ACEFA model across fitted parameter sets and vaccination scenarios.

    Saves cumulative infections, cases, hospitalisations and deaths by age.

    Output array dimensions:
        (n_sims, n_policies, n_scenarios, n_ages)

    Policies:
        iiv_65, iiv_50, laiv_5_12, laiv_5_18

    Scenarios:
        status_quo, pessimistic, mid, optimistic

    The status-quo baseline is stored at scenario index 0 for
    every policy, using the same baseline simulation.
    """

    contact_matrix = load_contact_matrix()

    # baseline_healthseeking = np.array([0.093087, 0.093087, 0.093087, 0.091192, 0.091192, 0.101058,
    #                                    0.101058, 0.111099, 0.111099])

    # Load fitted parameter sets
    (initial_conditions, my_healthseekings, r0_values, amplitudes, phase_shifts, initial_sw, initial_i
    ) = read_parameters_sensitivity_analysis(baseline_healthseeking)

    total_weeks = 52

    policies = ["iiv_65", "iiv_50", "laiv_5_12", "laiv_5_18"]
    scenarios = ["status_quo", "pessimistic", "mid", "optimistic"]

    n_sims = initial_conditions.shape[0]
    n_policies = len(policies)
    n_scenarios = len(scenarios)

    # Initialise cumulative outcome arrays
    meta_shape = (n_sims, n_policies, n_scenarios, n_ages)

    total_infections_meta = np.zeros(meta_shape)
    total_cases_meta = np.zeros(meta_shape)
    total_hospitalizations_meta = np.zeros(meta_shape)
    total_deaths_meta = np.zeros(meta_shape)

    outcome_arrays = [total_infections_meta, total_cases_meta, total_hospitalizations_meta, total_deaths_meta]

    # Weekly outcomes across all ages
    weekly_shape = (n_sims, n_policies, n_scenarios, total_weeks)

    weekly_infections_meta = np.zeros(weekly_shape)
    weekly_cases_meta = np.zeros(weekly_shape)
    weekly_hospitalizations_meta = np.zeros(weekly_shape)
    weekly_deaths_meta = np.zeros(weekly_shape)

    weekly_outcome_arrays = [weekly_infections_meta,
                             weekly_cases_meta,
                             weekly_hospitalizations_meta,
                             weekly_deaths_meta]

    # Loop through fitted parameter sets
    for sim_idx in range(n_sims):

        initial_population = initial_conditions[sim_idx]
        p_healthseeking = my_healthseekings[sim_idx]
        r0 = r0_values[sim_idx]
        amplitude = amplitudes[sim_idx]
        phase_shift = phase_shifts[sim_idx]

        print(f'sim {sim_idx + 1}/{n_sims}: '
              f'| R0: {r0:.2f} | amplitude: {amplitude:.3f} '
              f'| phase shift: {phase_shift:.0f}')

        # Run baseline
        baseline_output, _, _, _, _ = run_model_equations_weekly_vaccination_3_laiv12_multirun(
                initial_population=initial_population,
                amplitude=amplitude,
                R0=r0,
                vaccine_uptake="mid",
                vaccine_policy="status_quo",
                contact_matrix=contact_matrix)

        baseline_y = baseline_output.reshape((-1, n_states + 3, n_ages))

        # Calculate hosps and deaths per infection from baseline
        hosp_per_infection = get_hospitalisations_per_infection(baseline_y, populations_by_fractions, p_hospitalised,
                                                                p_asymptomatic)

        death_per_infection = get_deaths_per_infection(baseline_y, populations_by_fractions, p_deaths, p_asymptomatic)

        _, ve_sus, ve_gp, ve_hosp = load_VE_and_VC_IIV(vaccine_policy='status_quo', vaccine_scenario='status_quo')

        outcome_parameters = [total_weeks, N, ve_sus, ve_gp, ve_hosp, p_asymptomatic, p_healthseeking,
                              hosp_per_infection, death_per_infection]

        # Baseline cumulative outcomes by age
        baseline_totals = calculate_total_outcomes_by_age(y=baseline_y, parameters=outcome_parameters)

        # Baseline weekly outcomes, aggregated across all ages
        baseline_weekly = calculate_weekly_outcomes(y=baseline_y, parameters=outcome_parameters)

        # Store baseline at scenario index 0 for all 4 policies
        for meta, baseline in zip(outcome_arrays, baseline_totals):
            meta[sim_idx, :, 0, :] = baseline

        # Store baseline weekly outcomes for all policies
        for meta, values in zip(weekly_outcome_arrays, baseline_weekly):
            meta[sim_idx, :, 0, :] = values

        # Run each vaccination policy
        for policy_idx, policy in enumerate(policies):

            # Run pessimistic, mid and optimistic scenarios
            for scenario_idx, uptake in enumerate(scenarios[1:], start=1):
                output, _, _, _, _ = run_model_equations_weekly_vaccination_3_laiv12_multirun(
                        initial_population=initial_population,
                        amplitude=amplitude,
                        R0=r0,
                        vaccine_uptake=uptake,
                        vaccine_policy=policy,
                        contact_matrix=contact_matrix)

                y = output.reshape((-1, n_states + 3, n_ages))

                # Calculate cumulative outcomes by age
                totals = calculate_total_outcomes_by_age(y=y, parameters=outcome_parameters)

                # Calculate weekly outcomes
                weekly = calculate_weekly_outcomes(y=y, parameters=outcome_parameters)

                # Store results
                for meta, values in zip(outcome_arrays, totals):
                    meta[sim_idx, policy_idx, scenario_idx, :] = values

                # Store weekly outcomes
                for meta, values in zip(weekly_outcome_arrays, weekly):
                    meta[sim_idx, policy_idx, scenario_idx, :] = values

    plot_weekly_and_cumulative_outcomes(weekly_outcome_arrays=weekly_outcome_arrays,
                                        policies=['iiv_65', 'iiv_50', 'laiv_5_12', 'laiv_5_18'],  # policies,
                                        scenarios=['status_quo', 'pessimistic', 'mid', 'optimistic'], # scenarios,
                                        total_weeks=total_weeks)

    # Save cumulative outcomes

    output_file = 'my_outputs_40.pkl'
    data_to_save = {"total_infections_meta": total_infections_meta,
                    "total_cases_meta": total_cases_meta,
                    "total_hospitalizations_meta": total_hospitalizations_meta,
                    "total_deaths_meta": total_deaths_meta,
                    "n_ages": n_ages,
                    "my_strategies": policies,
                    "my_vaccine_scenarios": scenarios}

    with open(output_file, "wb") as file:
        pickle.dump(data_to_save, file)

    print(f"Saved: {output_file}")

    return



def plot_averted_outcomes_relative_to_baseline():
    """
    Plot percentage reductions in cases, hospitalisations and deaths
    relative to the status-quo baseline.

    For each fitted trajectory, percentage reduction is calculated
    relative to its corresponding baseline. The plotted point estimate
    is the mean of these trajectory-specific percentage reductions.

    Error bars represent the minimum and maximum across trajectories.

    Strategies:
        IIV 65 optimistic
        IIV 50 central
        LAIV 5-12 central
        LAIV 5-18 central
    """

    my_file_name = 'my_outputs_40.pkl'

    # Load simulation results
    with open(my_file_name, "rb") as file:
        data = pickle.load(file)

    policies = list(data["my_strategies"])
    scenarios = list(data["my_vaccine_scenarios"])

    outcome_keys = {"Cases": "total_cases_meta",
                    "Hospitalisations": "total_hospitalizations_meta",
                    "Deaths": "total_deaths_meta"}

    # Select vaccination scenario for each policy
    policy_scenarios = {"iiv_65": "optimistic", "iiv_50": "mid", "laiv_5_12": "mid", "laiv_5_18": "mid"}

    strategy_labels = {"iiv_65": "IIV 65", "iiv_50": "IIV 50", "laiv_5_12": "LAIV 5-12", "laiv_5_18": "LAIV 5-18"}

    labels = [strategy_labels[p] for p in policy_scenarios]
    selected_policies = list(policy_scenarios)

    # Plot formatting
    colors = sns.color_palette("colorblind")
    color_all = colors[-5]  # brown
    color_65 = colors[-1]  # blue

    bar_width = 0.2
    bar_offset = bar_width + 0.05
    x = np.arange(len(selected_policies))

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), layout="tight")

    for idx, (outcome_name, outcome_key) in enumerate(outcome_keys.items()):
        outcomes = np.asarray(data[outcome_key])

        # Expected shape: (n_sims, n_policies, n_scenarios, n_ages)
        if outcomes.ndim != 4:
            raise ValueError(f"{outcome_key} must have four dimensions.")

        # Status-quo baseline is policy 0, scenario 0
        baseline = outcomes[:, policies.index("iiv_65"), scenarios.index("status_quo"), :]

        # Select intervention outcomes for each vaccination policy
        intervention_list = []
        for policy in selected_policies:
            # Find index of vaccination policy and vaccination scenario
            policy_idx = policies.index(policy)
            scenario = policy_scenarios[policy]
            scenario_idx = scenarios.index(scenario)

            # Get outcomes for all 40 trajectories and 9 age groups
            policy_outcomes = outcomes[:, policy_idx, scenario_idx, :]
            intervention_list.append(policy_outcomes)

        # Combine outcomes into a single array
        intervention = np.stack(intervention_list, axis=1)

        # Baseline and intervention outcomes: all ages
        baseline_all = baseline.sum(axis=-1)
        intervention_all = intervention.sum(axis=-1)

        # Baseline and intervention outcomes: ages 65+
        baseline_65 = baseline[:, -2:].sum(axis=-1)
        intervention_65 = intervention[:, :, -2:].sum(axis=-1)

        # Percentage reduction for each fitted trajectory
        with np.errstate(divide="ignore", invalid="ignore"):
            averted_all = ( (baseline_all[:, None] - intervention_all) / baseline_all[:, None] ) * 100
            averted_65 = ( (baseline_65[:, None] - intervention_65) / baseline_65[:, None] ) * 100

        if not (np.isfinite(averted_all).all() and np.isfinite(averted_65).all()):
            raise ValueError(f"Undefined percentage reductions for {outcome_name}. "
                             "Check for zero baseline outcomes.")

        # Mean, min and max across the 40 trajectories
        mean_all = averted_all.mean(axis=0)
        min_all = averted_all.min(axis=0)
        max_all = averted_all.max(axis=0)

        mean_65 = averted_65.mean(axis=0)
        min_65 = averted_65.min(axis=0)
        max_65 = averted_65.max(axis=0)

        # Asymmetric error bars
        error_all = np.array([mean_all - min_all, max_all - mean_all])
        error_65 = np.array([mean_65 - min_65, max_65 - mean_65])

        ax = axes[idx]

        ax.bar(x, mean_all, width=bar_width, yerr=error_all, color=color_all, label="All ages")
        ax.bar(x + bar_offset, mean_65, width=bar_width, yerr=error_65, color=color_65, label="65+")

        # Figure formatting
        ax.set_title(outcome_name, fontsize=14)
        ax.set_ylim(0, 21)
        ax.set_yticks(np.arange(0, 21, 5))
        ax.set_xticks(x + bar_offset / 2)
        ax.set_xticklabels(labels, fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)

        ax.text(0, 1.03, "ABC"[idx], transform=ax.transAxes, fontsize=16, fontweight="bold")

        if idx == 0:
            ax.set_ylabel("% reduction in event", fontsize=12)
            ax.legend(frameon=False, ncol=2, fontsize=11)

    return fig


