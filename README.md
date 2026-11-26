# Modelling the impact of live-attenuated influenza vaccination in Australia

is available at: https://www.medrxiv.org/content/10.64898/2026.11.23.26368629v1

Code for the age-structured models of seasonal influenza transmission and vaccination.


## Python files

**`default_params.py`**

Defines the default model parameters and functions for loading demographic data, contact matrices, vaccine coverage, 
vaccine effectiveness and fitted parameter sets.

**`functions_flu.py`**

Contains the influenza transmission equations, vaccination rollout schedules and functions for calculating 
epidemiological outcomes.

**`main_runmodel.py`**

Contains the functions for running vaccination scenarios across fitted parameter sets and generating plots of weekly, 
cumulative and averted influenza outcomes.


The main simulation function is:

```python
run_ACEFA_scenarios_with_weekly_vacc_3_multirun()
```

It runs the influenza model across 40 fitted parameter sets and evaluates the status-quo baseline and four vaccination policies under pessimistic, central and optimistic scenarios.

For each fitted parameter set, the model calculates cumulative infections, medically attended cases, hospitalisations and deaths.

