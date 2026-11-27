import numpy as np
from default_params import *
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statistics
from scipy.stats import binomtest
from calibration_functions import *


if __name__ == '__main__':
    ############################
    ##### calibration data #####
    ############################
    nndss_data = load_nndss_data(filepath="data/nndss/Daily_case_counts_20150101_to_20250917_NNDSSonly.csv")

    nndss_data_plot = nndss_data.groupby(['week', 'week_cm', 'year'], observed=True).agg(
        cases=('cases', 'sum')).reset_index()

    nndss_data = nndss_data[nndss_data['year'].isin([2020, 2021]) == False].reset_index()

    nndss_data_plot = nndss_data_plot[nndss_data_plot['year'].isin([2020, 2021]) == False].reset_index()

    conditions = [nndss_data['age_group'] == '0-1',
                  nndss_data['age_group'] == '1-4',
                  nndss_data['age_group'] == '5-11',
                  nndss_data['age_group'] == '12-17',
                  nndss_data['age_group'] == '18-64',
                  nndss_data['age_group'] == '65-79',
                  nndss_data['age_group'] == '>80']

    index_values = np.linspace(0, len(conditions) - 1, len(conditions))

    nndss_data['age_index'] = np.select(conditions, index_values)
    nndss_data['age_index'] = nndss_data['age_index'].astype(int)
    nndss_data['time_index'] = nndss_data['week'] - 1
    nndss_data['time_index'] = nndss_data['time_index'].astype(int)
    nndss_data['year_index'] = pd.factorize(nndss_data['year'])[0]

    nndss_data_pre_covid = nndss_data[nndss_data['year'] < 2020].reset_index()
    nndss_data_pre_covid['year_index'] = pd.factorize(nndss_data_pre_covid['year'])[0]
    nndss_data_post_covid = nndss_data[nndss_data['year'] > 2020].reset_index()
    nndss_data_post_covid['year_index'] = pd.factorize(nndss_data_post_covid['year'])[0]

    ###############################################
    ##### determining the ascertainment rate ######
    ###############################################\

    flutracking_data = pd.concat(
        [load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2015.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2016.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2017.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2018.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2019.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2022.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2023.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2024.csv'),
         load_flutracking_data(filepath='data/flutracking/WeeklySummary_AU_2025.csv')]
    )

    flutracking_data = flutracking_data.groupby(['year', 'age_group'], observed=True).agg(n_total=('CaseN', 'sum'),
                                                                         n_test=('CaseFluTestN', 'sum')).reset_index()

    flutracking_data['p_healthseeking'] = flutracking_data['n_test'] / flutracking_data['n_total']
    flutracking_data['n_no_test'] = flutracking_data['n_total'] - flutracking_data['n_test']

    for i in range(len(flutracking_data)):
        flutracking_data.loc[i, 'p_healthseeking_low'] = \
            binomtest(k=flutracking_data['n_test'][i], n=flutracking_data['n_total'][i]).proportion_ci()[0]
        flutracking_data.loc[i, 'p_healthseeking_high'] = \
            binomtest(k=flutracking_data['n_test'][i], n=flutracking_data['n_total'][i]).proportion_ci()[1]

    # Fit the model
    flutracking_pre_covid = flutracking_data[flutracking_data['year'] <= 2020]
    flutracking_post_covid = flutracking_data[flutracking_data['year'] > 2020]

    ascertainment_fit = smf.glm(formula='n_test + n_no_test ~ year + age_group', data=flutracking_pre_covid,
                                          family=sm.families.Binomial()).fit()

    #p_healthseeking_all = pd.DataFrame({'year': np.repeat(np.unique(flutracking_data['year']), len(np.unique(flutracking_data['age_group']))),
    #                                          'age_group': np.tile(np.unique(flutracking_data['age_group']), len(np.unique(flutracking_data['year'])))})

    #p_healthseeking_all = pd.concat([p_healthseeking_all, pd.DataFrame({'p': ascertainment_fit.predict(exog=p_healthseeking_all).to_numpy()})], axis=1)

    p_healthseeking_pre_covid = pd.DataFrame({'year': np.repeat(np.unique(flutracking_pre_covid['year']), 4),
                                              'age_group': np.tile(np.unique(flutracking_pre_covid['age_group']), 5)})

    p_healthseeking_pre_covid = pd.concat([p_healthseeking_pre_covid, pd.DataFrame({'p': ascertainment_fit.predict(exog=p_healthseeking_pre_covid).to_numpy()})], axis=1)

    p_healthseeking_post_covid = flutracking_post_covid.groupby(['age_group'], observed=True).agg(p=('n_test', lambda n_test: n_test.sum() / flutracking_post_covid.loc[n_test.index, 'n_total'].sum())).reset_index()

    p_healthseeking_post_covid = pd.DataFrame({'year' : np.repeat(np.unique(flutracking_post_covid['year']), 4),
                                               'age_group' : np.tile(p_healthseeking_post_covid['age_group'], 4),
                                              "p" : np.tile(p_healthseeking_post_covid['p'], 4)})

    ####################################################################################
    ##### adjusting the ascertainment rate to account for bias in flutracking data #####
    ####################################################################################

    # Eales, O., Teo, M., Price, D.J., Hao, T., Ryan, G.E., Senior, K.L., … Shearer, F.M.(2025).Temporal trends in test - seeking behaviour during the COVID - 19 pandemic.Mathematics in Medical and Life Sciences, 2(1).https: // doi.org / 10.1080 / 29937574.2025.2521858

    p_healthseeking_all = pd.concat([p_healthseeking_pre_covid, p_healthseeking_post_covid], axis=0)

    flutracking_prop_test_seeking_any_symptoms = 0.849
    NBS_prop_test_seeking_any_symptoms = 0.588
    scaling_flutracking = NBS_prop_test_seeking_any_symptoms / flutracking_prop_test_seeking_any_symptoms

    p_healthseeking_all['p_original'] = p_healthseeking_all['p']
    p_healthseeking_all['p'] = p_healthseeking_all['p'] * scaling_flutracking

    nndss_data.to_pickle('results/nndss_data.pkl')
    nndss_data_pre_covid.to_pickle('results/nndss_data_pre_covid.pkl')
    nndss_data_post_covid.to_pickle('results/nndss_data_post_covid.pkl')
    nndss_data_plot.to_pickle('results/nndss_data_plot.pkl')
    flutracking_data.to_pickle('results/flutracking_data.pkl')
    p_healthseeking_all.to_pickle('results/p_healthseeking_all.pkl')
    ascertainment_fit.save("results/ascertainment_fit.pkl")
