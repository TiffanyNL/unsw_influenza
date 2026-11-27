import numpy as np
from default_params import *
from plotnine import *
from calibration_functions import *
from acefa_functions import *
import pickle
import statistics
from scipy.stats import binomtest
import matplotlib.pyplot as plt
from mizani.formatters import percent_format
import h5py
import numdifftools as nd
from scipy.stats import norm
import pandas as pd
import numpy as np
import itertools
from plotnine import *

def gen_model_simulations(pars, data):

    R0, p_hs_scale_y, p_hs_scale_o, amplitude, phase_shift, init_SW, seed,  = pars

    model_runs = run_model_equations_weekly_vaccination(default_parameters=default_parameters,
                                                        status_quo_vac_parameters=status_quo_vac_parameters,
                                                        initial_SW= np.repeat(init_SW, n_ages), #vals_run['SW_vals'].item(),
                                                        seed = seed, #vals_run['seed'].item(),
                                                        amplitude= amplitude, #0, #vals_run['amplitude (A)'].item(),
                                                        R0= R0, #vals_run['R{}'.format('\u2080')].item(),
                                                        phase_shift= phase_shift# vals_run['phase shift (\N{GREEK SMALL LETTER PHI})'].item()
                                                        )
    model_runs = model_runs.reshape((-1, n_states + 3, n_ages))

    age_specific_cases = []
    age_specific_hospitalisations = []
    age_specific_infections = []
    years = np.unique(data['year'])

    #p_hs_scale_y = vals_run['\N{GREEK SMALL LETTER TAU}_child'].item()
    #p_hs_scale_o = vals_run['\N{GREEK SMALL LETTER TAU}_adult'].item()

    for i in range(0, len(years), 1):
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
                                       p_healthseeking_subset.loc[
                                           p_healthseeking_subset['age_group'] == '>65', 'p'] * p_hs_scale_o]).flatten()
        temp = calculate_weekly_outcomes_by_age_for_cal(model_runs, default_parameters, status_quo_vac_parameters, N_year_in, p_healthseeking_in, True)[1]
        temp = pd.DataFrame(temp)
        temp.columns = acefa_ages
        temp['week'] = temp.index
        temp['year'] = years[i]
        temp = pd.melt(temp, id_vars=['week', 'year'], value_vars=acefa_ages, var_name='age_group', value_name='cases')
        age_specific_cases.append(temp)

        temp_i = calculate_weekly_outcomes_by_age_for_cal(model_runs, default_parameters, status_quo_vac_parameters, N_year_in, p_healthseeking_in, True)[0]
        temp_i = pd.DataFrame(temp_i)
        temp_i.columns = acefa_ages
        temp_i['week'] = temp_i.index
        temp_i['year'] = years[i]
        temp_i = pd.melt(temp_i, id_vars=['week', 'year'], value_vars=acefa_ages, var_name='age_group', value_name='cases')
        age_specific_infections.append(temp_i)

        temp_h = calculate_weekly_outcomes_by_age_for_cal(model_runs, default_parameters, status_quo_vac_parameters, N_year_in, p_healthseeking_in, True)[2]
        temp_h = pd.DataFrame(temp_h)
        temp_h.columns = acefa_ages
        temp_h['week'] = temp_h.index
        temp_h['year'] = years[i]
        temp_h = pd.melt(temp_h, id_vars=['week', 'year'], value_vars=acefa_ages, var_name='age_group', value_name='cases')
        age_specific_hospitalisations.append(temp_h)


    age_specific_cases = pd.concat(age_specific_cases, axis=0)
    age_specific_cases['week'] = age_specific_cases['week'] + 1
    age_specific_infections = pd.concat(age_specific_infections, axis=0)
    age_specific_infections['week'] = age_specific_infections['week'] + 1
    age_specific_hospitalisations = pd.concat(age_specific_hospitalisations, axis=0)
    age_specific_hospitalisations['week'] = age_specific_hospitalisations['week'] + 1

    data['year'] = pd.Categorical(data['year'], ordered=True)
    age_specific_cases['year'] = pd.Categorical(age_specific_cases['year'], categories=years, ordered=True)
    age_specific_hospitalisations['year'] = pd.Categorical(age_specific_hospitalisations['year'], categories=years, ordered=True)
    age_specific_infections['year'] = pd.Categorical(age_specific_infections['year'], categories=years, ordered=True)

    data['age_group'] = pd.Categorical(data['age_group'], categories=acefa_ages, ordered=True)
    age_specific_cases['age_group'] = pd.Categorical(age_specific_cases['age_group'], categories=acefa_ages, ordered=True)
    age_specific_hospitalisations['age_group'] = pd.Categorical(age_specific_hospitalisations['age_group'], categories=acefa_ages, ordered=True)
    age_specific_infections['age_group'] = pd.Categorical(age_specific_infections['age_group'], categories=acefa_ages, ordered=True)

    year_age_o = np.repeat(np.unique(nndss_data['year']), 7).astype(str) + ', ' + np.tile(acefa_ages, 9).astype(str)
    data['facet_variable'] = data['year'].astype('string') + ', ' + data['age_group'].astype('string')
    data['facet_variable'] = pd.Categorical(data['facet_variable'], categories=year_age_o, ordered=True)

    age_specific_cases['facet_variable'] = age_specific_cases['year'].astype('string') + ', ' + age_specific_cases[
        'age_group'].astype('string')
    age_specific_cases['facet_variable'] = pd.Categorical(age_specific_cases['facet_variable'], categories=year_age_o,
                                                          ordered=True)
    age_specific_infections['facet_variable'] = age_specific_infections['year'].astype('string') + ', ' + age_specific_infections[
        'age_group'].astype('string')
    age_specific_infections['facet_variable'] = pd.Categorical(age_specific_infections['facet_variable'], categories=year_age_o,
                                                          ordered=True)
    age_specific_hospitalisations['facet_variable'] = age_specific_hospitalisations['year'].astype('string') + ', ' + age_specific_hospitalisations[
        'age_group'].astype('string')
    age_specific_hospitalisations['facet_variable'] = pd.Categorical(age_specific_hospitalisations['facet_variable'], categories=year_age_o,
                                                          ordered=True)

    data['year'] = data['year'].cat.remove_unused_categories()
    age_specific_cases['year'] = age_specific_cases['year'].cat.remove_unused_categories()
    age_specific_infections['year'] = age_specific_infections['year'].cat.remove_unused_categories()
    age_specific_hospitalisations['year'] = age_specific_hospitalisations['year'].cat.remove_unused_categories()

    age_specific_cases['seed'] = str(seed)
    age_specific_hospitalisations['seed'] = str(seed)
    age_specific_infections['seed'] = str(seed)
    age_specific_cases['init_SW'] = str(np.around(init_SW, decimals = 2))
    age_specific_hospitalisations['init_SW'] = str(np.around(init_SW, decimals=2))
    age_specific_infections['init_SW'] = str(np.around(init_SW, decimals=2))

    age_specific_cases['immunity-seed'] = 'prior immunity: ' + age_specific_cases['init_SW'] + ', seed: ' + age_specific_cases['seed']
    age_specific_hospitalisations['immunity-seed'] = 'prior immunity: ' + age_specific_hospitalisations['init_SW'] + ', seed: ' + \
                                          age_specific_hospitalisations['seed']
    age_specific_infections['immunity-seed'] = 'prior immunity: ' + age_specific_infections['init_SW'] + ', seed: ' + \
                                          age_specific_infections['seed']

    age_specific_cases['immunity-seed-year'] = 'prior immunity: ' + age_specific_cases['init_SW'] + ', seed: ' + \
                                          age_specific_cases['seed'] + ', year: ' + age_specific_cases['year'].astype(str)
    age_specific_hospitalisations['immunity-seed-year'] = 'prior immunity: ' + age_specific_hospitalisations['init_SW'] + ', seed: ' + \
                                               age_specific_hospitalisations['seed'] + ', year: ' + age_specific_hospitalisations['year'].astype(str)

    age_specific_infections['immunity-seed-year'] = 'prior immunity: ' + age_specific_infections['init_SW'] + ', seed: ' + \
                                                    age_specific_infections['seed'] + ', year: ' + \
                                                    age_specific_infections['year'].astype(str)

    return [data, age_specific_cases, age_specific_hospitalisations, age_specific_infections]

if __name__ == '__main__':

    ###################################################
    ##### loading in the required data and values #####
    ###################################################
    nndss_data = pd.read_pickle('results/nndss_data.pkl')
    nndss_data_plot = pd.read_pickle('results/nndss_data_plot.pkl')
    flutracking_data = pd.read_pickle('results/flutracking_data.pkl')
    p_healthseeking_all = pd.read_pickle('results/p_healthseeking_all.pkl')

    nndss_data['age_group'] = pd.Categorical(nndss_data['age_group'], categories=np.array(
        ['0-1', '1-4', '5-11', '12-17', '18-64', '65-79', '>80']), ordered=True)

    flutracking_data['age_group'] = pd.Categorical(flutracking_data['age_group'],
                                                   categories=np.array(['0-4', '5-17', '18-64', '>65']), ordered=True)

    with open('results/optimization_results_p_hs_scale_wo_a_ps_365.pkl', 'rb') as f:
        optimization_results = pickle.load(f)

    SW_vals = np.arange(0.05, 0.8, 0.1)
    seed_vals = [1E-8, 1E-7, 1E-6, 1E-5, 1E-4]
    combinations = pd.DataFrame(list(itertools.product(SW_vals, seed_vals)), columns=['SW_vals', 'seed'])

    combinations['R{}'.format('\u2080')] = np.ones(combinations.shape[0])
    combinations['\N{GREEK SMALL LETTER TAU}_child'] = np.ones(combinations.shape[0])
    combinations['\N{GREEK SMALL LETTER TAU}_adult'] = np.ones(combinations.shape[0])
    combinations['amplitude'] = np.ones(combinations.shape[0])
    combinations['phase-shift'] = np.ones(combinations.shape[0])
    combinations['log-likelihood'] = np.ones(combinations.shape[0])

    for i in range(0, combinations.shape[0]):
        combinations.loc[i, 'R{}'.format('\u2080')] = optimization_results[i]['x'][0]
        combinations.loc[i, '\N{GREEK SMALL LETTER TAU}_child'] = optimization_results[i]['x'][1]
        combinations.loc[i, '\N{GREEK SMALL LETTER TAU}_adult'] = optimization_results[i]['x'][2]
        combinations.loc[i, 'amplitude'] = optimization_results[i]['x'][3]
        combinations.loc[i, 'phase-shift'] = optimization_results[i]['x'][4]
        combinations.loc[i, 'log-likelihood'] = -optimization_results[i]['fun']

    combinations.to_csv('results/optimization_results_p_hs_scale_wo_a_ps_365.csv')

    parameter_plot = (
            ggplot(data = combinations.melt(id_vars=['SW_vals', 'seed'],
                    value_vars=['R{}'.format('\u2080'), '\N{GREEK SMALL LETTER TAU}_child', '\N{GREEK SMALL LETTER TAU}_adult', 'amplitude', 'phase-shift'],
                                            var_name='parameter', value_name='value')) +
            geom_point(aes(x='SW_vals', y='value', colour='factor(seed)'), size=3) +
            labs(x='Initial proportion with prior immunity',
                 y='Parameter value') +
            theme_classic(base_size=18) +
            facet_wrap('~parameter', scales="free_y") +
            theme(figure_size=(10, 7)) +
            scale_colour_brewer(type="qual", palette='Set2', name="initial\ninfection\nprevalence")
    )

    parameter_plot.show()

    # flutracking data
    flutracking_plot = (
            ggplot() +
            geom_pointrange(aes(x='year', y='p_healthseeking', colour = 'age_group', ymin='p_healthseeking_low', ymax='p_healthseeking_high'),
                            data=flutracking_data, size = 1) +
            geom_line(aes(x='year', colour = 'age_group', y='p'), data=p_healthseeking_all, size = 1, linetype = "dashed") +
            geom_line(aes(x='year', colour='age_group', y='p_original'), data=p_healthseeking_all, size=1) +
            labs(x='Year',
                 y='Proportion of self-identified people with\nsymptoms who received a flu test') +
            theme_classic(base_size=18) +
            scale_y_continuous(labels = percent_format(), limits = (0, 0.5)) +
            scale_x_continuous(breaks = np.arange(2015,2026,2)) +
            theme(figure_size=(10, 7)) +
            scale_colour_brewer(type="qual", palette='Set2', name = "age group")
    )

    flutracking_plot.show()

    flutracking_n_plot = (
            ggplot() +
            geom_point(aes(x='year', y='n_total', colour='age_group'),
                            data=flutracking_data, size=3) +
            labs(x='Year',
                 y='Total self-identified people with\nsymptoms who received a flu test') +
            theme_classic(base_size=18) +
            scale_x_continuous(breaks=np.arange(2015, 2026, 2)) +
            theme(figure_size=(10, 7)) +
            scale_colour_brewer(type="qual", palette='Set2', name="age group")
    )
    flutracking_n_plot.show()


    # running the model simulations for each

    model_simulations_o = gen_model_simulations([combinations.loc[0, 'R{}'.format('\u2080')],
                                                 combinations.loc[0, '\N{GREEK SMALL LETTER TAU}_child'],
                                                 combinations.loc[0, '\N{GREEK SMALL LETTER TAU}_adult'],
                                                 combinations.loc[0, 'amplitude'],
                                                 combinations.loc[0, 'phase-shift'],
                                                 combinations.loc[0, 'SW_vals'],
                                                 combinations.loc[0, 'seed']],
                                                nndss_data)

    nndss_data_plot = model_simulations_o[0]

    model_simulations = model_simulations_o[1]
    model_simulations_h = model_simulations_o[2]
    model_simulations_i = model_simulations_o[3]

    for i in range(1, np.shape(combinations)[0]):
        print(i)

        model_sims_temp = gen_model_simulations([combinations.loc[i, 'R{}'.format('\u2080')],
                                                 combinations.loc[i, '\N{GREEK SMALL LETTER TAU}_child'],
                                                 combinations.loc[i, '\N{GREEK SMALL LETTER TAU}_adult'],
                                                 combinations.loc[i, 'amplitude'],
                                                 combinations.loc[i, 'phase-shift'],
                                                 combinations.loc[i, 'SW_vals'],
                                                 combinations.loc[i, 'seed']],
                                                nndss_data)

        model_simulations = pd.concat([model_simulations, model_sims_temp[1]])
        model_simulations_h = pd.concat([model_simulations_h, model_sims_temp[2]])
        model_simulations_i = pd.concat([model_simulations_i, model_sims_temp[3]])

    nndss_age_fit_plot = (
            ggplot() +
            geom_line(aes(x = 'week', y = 'cases', group = "immunity-seed-year", colour='year'), data=model_simulations, size=0.25, alpha = 0.75) +
            geom_point(aes(x='week', y='cases', fill = 'year'), data=nndss_data_plot, size=1, alpha=0.75, stroke = 0) +
            geom_point(aes(x='week', y='cases'), data=nndss_data_plot, size=1, fill="none", alpha = 0.5) +
            labs(x='Week of year',
                 y='Total influenza cases') +
            theme_classic(base_size=19) +
            facet_wrap('age_group', scales="free_y") +
            theme(figure_size=(10, 7)) +
            scale_x_continuous(breaks=np.arange(0, 53, 10))
    )

    model_simulations.to_csv('results/notifications_total_by_age_year.csv')
    model_simulations_h.to_csv('results/hospitalisations_total_by_age_year.csv')
    model_simulations_i.to_csv('results/infections_total_by_age_year.csv')

    nndss_age_fit_plot.show()

    # all the age groups
    nndss_data_total = nndss_data_plot.groupby(['week', 'year'], observed=True).agg(cases=('cases', 'sum')).reset_index()

    nndss_data_total.to_csv('results/nndss_total_by_week.csv')

    model_simulations_total = model_simulations.groupby(['week', 'year', 'init_SW', 'immunity-seed', 'immunity-seed-year', 'seed'], observed=True).agg(cases=('cases', 'sum')).reset_index()
    model_simulations_h_total = model_simulations_h.groupby(['week', 'year', 'init_SW', 'immunity-seed', 'immunity-seed-year', 'seed'], observed=True).agg(cases=('cases', 'sum')).reset_index()
    model_simulations_i_total = model_simulations_i.groupby(['week', 'year', 'init_SW', 'immunity-seed', 'immunity-seed-year', 'seed'], observed=True).agg(cases=('cases', 'sum')).reset_index()

    nndss_total_fit_plot = (
            ggplot() +
            geom_line(aes(x='week', y='cases', group="immunity-seed-year", colour = "seed"), data=model_simulations_total,
                      size=0.25, alpha=0.75) +
            geom_point(aes(x='week', y='cases'), data=nndss_data_total, size=2.5, alpha=0.75, stroke=0) +
            labs(x='Week of year',
                 y='Total influenza cases') +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7)) +
            scale_x_continuous(breaks=np.arange(0, 53, 10)) +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            facet_wrap('year')
    )

    nndss_total_fit_plot.show()

    model_cases_2023 = model_simulations_total[model_simulations_total['year'].isin([2023])].copy()
    model_cases_2023['type'] = "notifications"

    model_infections_2023 = model_simulations_i_total[model_simulations_i_total['year'].isin([2023])].copy()
    model_infections_2023['type'] = "infections"

    model_hospitalisations_2023 = model_simulations_h_total[model_simulations_h_total['year'].isin([2023])].copy()
    model_hospitalisations_2023['type'] = "hospitalisations"

    model_total_2023 = pd.concat([model_cases_2023, model_infections_2023, model_hospitalisations_2023])

    nndss_data_total['type'] = "notifications"

    model_total_2023['type'] = model_total_2023['type'].astype('category')

    model_total_2023['type'] = model_total_2023['type'].cat.set_categories(['hospitalisations', 'notifications', 'infections'], ordered=True)

    nndss_data_total['type'] = nndss_data_total['type'].astype('category')
    nndss_data_total['type'] = nndss_data_total['type'].cat.set_categories(['hospitalisations', 'notifications', 'infections'], ordered=True)

    model_total_2023.to_csv('results/simulations_age_aggregated_2025_2024.csv')

    nndss_2023_all_plot = (
            ggplot() +
            geom_line(aes(x='week', y='cases', group="immunity-seed-year", colour="seed"), data= model_total_2023,
                      size=0.25, alpha=0.75) +
            geom_point(aes(x='week', y='cases'), data=nndss_data_total[nndss_data_total['year'] == 2025], size=2.5, alpha=0.75, stroke=0) +
            labs(x='Week of year',
                 y='Total influenza counts') +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7)) +
            scale_x_continuous(breaks=np.arange(0, 53, 10)) +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            facet_wrap('type', scales = "free_y")
    )

    nndss_2023_all_plot.show()

    model_total_2025_per_100 = model_total_2025.copy()

    model_total_2025_per_100 = model_total_2025_per_100.merge(N_all[['year', 'N']], on='year', how='left')

    model_total_2025_per_100['cases'] = (model_total_2025_per_100['cases'] / model_total_2025_per_100['N']) * 1000000

    model_total_2025_per_100.to_csv('results/simulations_age_aggregated_2025_2024_per_mil.csv')

    # total count plots
    nndss_data_total_per_year = nndss_data_plot.groupby(['year'], observed=True).agg(cases=('cases', 'sum')).reset_index()

    model_simulations_total_per_year = model_simulations.groupby(['year', 'init_SW', 'immunity-seed', 'immunity-seed-year', 'seed'], observed=True).agg(cases=('cases', 'sum')).reset_index()

    model_simulations_i_total_per_year = model_simulations_i.groupby(
        ['year', 'init_SW', 'immunity-seed', 'immunity-seed-year', 'seed'], observed=True).agg(
        cases=('cases', 'sum')).reset_index()

    np.max(model_simulations_i_total_per_year['cases'] / N_all.loc[N_all['year'] == 2025, 'N'].iloc[0] * 100)

    nndss_total_per_year_plot = (
            ggplot() +
            geom_bar(aes(x='year', y='cases'), data=nndss_data_total_per_year, alpha=0.5, stat = 'identity') +
            geom_point(aes(x='year', y='cases', group="immunity-seed", colour="seed"), data=model_simulations_total_per_year,
                      size=3, alpha=0.5) +
            geom_point(aes(x='year', y='cases', group="immunity-seed", colour="seed"),
                       data=model_simulations_total_per_year,
                       size=3, fill="none") +
            labs(x='Year',
                 y='Total influenza notifications') +
            theme_classic(base_size=19) +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            theme(figure_size=(10, 7))
    )

    nndss_total_per_year_plot.show()

    model_simulations_total_per_year_per_100 = model_simulations_total_per_year.copy()

    for i in range(0, np.shape(model_simulations_total_per_year_per_100)[0]):
        model_simulations_total_per_year_per_100.loc()[i, 'cases'] = (model_simulations_total_per_year_per_100.loc()[i, 'cases'] /N_all.loc[N_all['year'] == model_simulations_total_per_year_per_100.loc()[i, 'year'], 'N'].iloc[0]) * 100000

    nndss_total_per_year_per_100_plot = (
            ggplot() +
            geom_point(aes(x='year', y='cases', group="immunity-seed", colour="seed"),
                       data=model_simulations_total_per_year_per_100,
                       size=3, alpha=0.5) +
            geom_point(aes(x='year', y='cases', group="immunity-seed", colour="seed"),
                       data=model_simulations_total_per_year_per_100,
                       size=3, fill="none") +
            labs(x='Year',
                 y='Total influenza notifications\nper 100,000 people') +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7))
    )

    nndss_total_per_year_per_100_plot.show()


    ##### comparison with scenario status quo #####

    sim_check_h = []
    sim_check = []
    sim_check_i = []
    for i in range(1, 41):
        print(i)
        sim_check_t = pd.read_parquet('for_isaac_params_and_statusquo/for_isaac_statusquo/flunsw_laiv_5_12_temperate_' + str(i) + '.parquet', engine='fastparquet')
        sim_check_t = sim_check_t[sim_check_t['scenario'] == 'Status_quo']
        sim_check_t["week"] = np.floor(sim_check_t["horizon"] / 7) + 1
        sim_check_t = sim_check_t.groupby(['week', 'target', 'scenario'], observed=True).agg(value=('value', 'sum')).reset_index()
        sim_check_t['value'] = sim_check_t['value'] * 27586007.0 / 1000000
        sim_check_t['init_SW'] = combinations.loc[i-1, 'SW_vals']
        sim_check_t['seed'] = combinations.loc[i - 1, 'seed']
        sim_check_i.append(sim_check_t[sim_check_t['target'] == "infection_incidence"])
        sim_check_h.append(sim_check_t[sim_check_t['target'] == "admission_incidence"])
        sim_check.append(sim_check_t[sim_check_t['target'] == "disease_incidence"])

    sim_check_i = pd.concat(sim_check_i, axis = 0, ignore_index = True)
    sim_check = pd.concat(sim_check, axis=0, ignore_index=True)
    sim_check_h = pd.concat(sim_check_h, axis=0, ignore_index=True)

    model_hospitalisations_2025['init_SW'] = pd.to_numeric(model_hospitalisations_2025['init_SW'], errors='coerce').astype(float)
    model_hospitalisations_2025['seed'] = pd.to_numeric(model_hospitalisations_2025['seed'], errors='coerce').astype(float)
    sim_check_h['init_SW'] = np.round(sim_check_h['init_SW'], decimals=2)
    model_hospitalisations_2025['init_SW'] = np.round(model_hospitalisations_2025['init_SW'], decimals=2)

    model_infections_2025['init_SW'] = pd.to_numeric(model_infections_2025['init_SW'], errors='coerce').astype(float)
    model_infections_2025['seed'] = pd.to_numeric(model_infections_2025['seed'], errors='coerce').astype(float)
    sim_check_i['init_SW'] = np.round(sim_check_i['init_SW'], decimals=2)
    model_infections_2025['init_SW'] = np.round(model_infections_2025['init_SW'], decimals=2)

    model_cases_2025['init_SW'] = pd.to_numeric(model_cases_2025['init_SW'], errors='coerce').astype(float)
    model_cases_2025['seed'] = pd.to_numeric(model_cases_2025['seed'], errors='coerce').astype(float)
    sim_check['init_SW'] = np.round(sim_check['init_SW'], decimals=2)
    model_cases_2025['init_SW'] = np.round(model_cases_2025['init_SW'], decimals=2)

    check_h_plot = (
            ggplot() +
            geom_line(aes(x = 'week', y = 'value'), data = sim_check_h, colour = "skyblue") +
            geom_line(aes(x= 'week', y = 'cases'), data = model_hospitalisations_2025, linetype='dashed') +
            labs(x='Week',
                 y='Total influenza hospitalisations') +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            facet_grid('init_SW~seed', scales = "free") +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7))
    )

    check_h_plot.show()

    check_c_plot = (
            ggplot() +
            geom_line(aes(x='week', y='value'), data=sim_check, colour="skyblue") +
            geom_line(aes(x='week', y='cases'), data=model_cases_2025, linetype='dashed') +
            labs(x='Week',
                 y='Total influenza notifications') +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            facet_grid('init_SW~seed', scales="free") +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7))
    )

    check_c_plot.show()

    check_i_plot = (
            ggplot() +
            geom_line(aes(x='week', y='value'), data=sim_check_i, colour="skyblue") +
            geom_line(aes(x='week', y='cases'), data=model_infections_2025, linetype='dashed') +
            labs(x='Week',
                 y='Total influenza infections') +
            guides(colour=guide_legend(title="initial infection\nprevalence")) +
            facet_grid('init_SW~seed', scales="free") +
            theme_classic(base_size=19) +
            theme(figure_size=(10, 7))
    )

    check_i_plot.show()



    nndss_data = pd.read_pickle('results/nndss_data.pkl')

    df_plot = (ggplot(
        data=df_sum[np.isin(df_sum['target'], ["admission_incidence", "disease_incidence", "infection_incidence"])]) +
               geom_line(aes(x="week", y="value")) +
               facet_wrap('target', scales="free_y") +
               theme_bw()  # +
               # geom_point(aes(x = "week", y = "value"), data=nndss_data_plot)
               )

    df_plot.show()


    age_specific_cases = model_simulations
    age_specific_cases.rename(columns={'cases': 'sim_cases'}, inplace=True)

    af_data = pd.merge(nndss_data[['week', 'year', 'age_group', 'facet_variable', 'cases']],
                       age_specific_cases[['week', 'year', 'age_group', 'facet_variable', 'sim_cases']],
                       on=['week', 'year', 'age_group', 'facet_variable'], how='outer', ).dropna(subset=['cases'])

    af_data['year'] = pd.Categorical(af_data['year'], ordered=True)

    actual_fitted_plot = (
            ggplot(af_data, aes(x='cases', y='sim_cases', colour='year', group='year')) +
            geom_point(alpha=0.75, size = 1.5) +
            geom_point(alpha=0.05, size=1.5, colour='grey', fill='none') +
            labs(x='Sampled cases', y='Simulated cases') +
            theme_classic(base_size=20) +
            facet_wrap('~age_group', scales="free") +
            theme(figure_size=(10, 7)) +
            geom_abline(slope=1, intercept=0, colour='black', linetype='dashed', size=1)  # +
        # geom_smooth(formula = 'y~x', method = 'lm', se=False, size = 0.7)
    )

    actual_fitted_plot.show()

    (nndss_age_fit_plot / actual_fitted_plot)

    ##### calculating the 95% confidence intervals #####

    seed_in = combinations['seed'][combinations['log-likelihood'] == max(combinations['log-likelihood'])]
    SW_in = combinations['SW_vals'][combinations['log-likelihood'] == max(combinations['log-likelihood'])]

    def objective_fun(mle_pars):
        return log_likelihood(
            pars = mle_pars,
            df_cases = nndss_data,
            p_healthseeking_all = p_healthseeking_all,
            default_parameters = default_parameters,
            status_quo_vac_parameters = status_quo_vac_parameters,
            max_likelihood=True
        )

    mle_pars = combinations[['R{}'.format('\u2080'), '\N{GREEK SMALL LETTER TAU}_child', '\N{GREEK SMALL LETTER TAU}_adult']].loc[combinations['log-likelihood'] == max(combinations['log-likelihood'])].to_numpy()

    hessian_calculator = nd.Hessian(objective_fun, step=1e-5, method='central')
    information_matrix = hessian_calculator(mle_pars)
    cov_matrix = np.linalg.inv(information_matrix)
    variances = np.diag(cov_matrix)
    std_errors = np.sqrt(variances)
    z_score = norm.ppf(1 - 0.01 / 2)  # e.g., 1.95996... for 95% CI
    lower_ci = mle_pars - z_score * std_errors
    upper_ci = mle_pars + z_score * std_errors

    ml_vals = pd.DataFrame({'SW' : SW_vals,
                            'R{}'.format('\u2080') : R0_vals,
                            'amplitude (A)' : amplitude_vals,
                            #'size' : size_vals,
                            'phase shift (\N{GREEK SMALL LETTER PHI})' : phase_shift_vals,
                            '\N{GREEK SMALL LETTER TAU}_child' : p_hs_scale_y,
                            '\N{GREEK SMALL LETTER TAU}_adult' : p_hs_scale_o})

    ml_vals_run = ml_vals.loc[ml_vals['SW'] == SW_vals[2], :]

    ml_vals_run_SW = SW_vals[2]#ml_vals.loc[np.argmax(ll_vals), "SW"]

    ml_vals_long = pd.melt(ml_vals, id_vars=['SW'], value_vars=ml_vals.columns[[1, 2, 3, 4, 5]], var_name='parameter', value_name='value')

    ml_vals_plot = (
            ggplot() +
            geom_point(ml_vals_long, aes(x='SW', y='value'), size = 4, shape = 'o') +
            labs(x='Initial proportion with prior immunity',
                 y='Fitted parameter') +
            theme_classic(base_size=20) +
            facet_wrap('~parameter', scales="free_y") +
            scale_fill_brewer(name = "Data", type="qual", palette='Pastel1') +
            theme(figure_size=(10, 7)))

    ml_vals_plot.show()

    ml_vals_run = combinations[['R{}'.format('\u2080'), '\N{GREEK SMALL LETTER TAU}_child', '\N{GREEK SMALL LETTER TAU}_adult', 'SW_vals', 'seed']].loc[39]

