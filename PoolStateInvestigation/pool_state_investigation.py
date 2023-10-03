import pickle

import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Models import Models
from Samplers.eHMMSampler import eHMMSampler
from Tools.DiagnosticTools import *

scheme_order = ['NR', 'R']
pool_update_order = ['shift', 'autoregressive']

acceptance_low = 0.4
acceptance_high = 0.5
delta = 0.25
def adaptive_epsilon(**kwargs):
    """
    increases or decreases target epsilon based on acceptance rate
    then randomly samples from a uniform distribution U(epsilon - delta, epsilon + delta)
    """
    previous_epsilon = kwargs['previous_epsilon']
    current_acceptance_rate = kwargs['current_acceptance_rate']

    if previous_epsilon is None:  # for first iteration use a default
        return np.random.uniform(0.1, 0.4)
    else:
        if current_acceptance_rate < acceptance_low:
            target_epsilon = previous_epsilon * 0.95
            if delta == 0:
                return max(min(target_epsilon, 0.02), 1)

            bounds = (target_epsilon - delta, target_epsilon + delta)
            bounds = (max(bounds[0], 0.02), min(bounds[1], 1))  # don't let epsilon go below 0.02 or above 1

            random_epsilon = np.random.uniform(bounds[0], bounds[1])
            return random_epsilon
        elif current_acceptance_rate > acceptance_high:

            target_epsilon = previous_epsilon * 1.05
            if delta == 0:
                return max(min(target_epsilon, 0.02), 1)

            bounds = (target_epsilon - delta, target_epsilon + delta)
            bounds = (max(bounds[0], 0.02), min(bounds[1], 1))

            random_epsilon = np.random.uniform(bounds[0], bounds[1])
            return random_epsilon
        else:
            return previous_epsilon

def epsilon_func(**kwargs):
    return adaptive_epsilon(**kwargs)

n_obs = 250
model = Models.Model1()
model.generate_data(n_obs)

# overwrite with the fixed realisation of latent process used in Shestopaloff-Neal 2018
all_sheets = pd.read_excel('../source.xlsx', sheet_name=None, header=None)
x_df = all_sheets['x']
y_df = all_sheets['y']

model.true_hss = [x_df[col].values for col in x_df.columns]
model.observations = [y_df[col].values for col in y_df.columns]
model.n_observations = len(model.observations)

initial_hss = [np.zeros(model.parameter_dict['n_vars'], dtype=np.float64) for _ in range(n_obs)]

def n_pool_states_generator():
    """
    define a generator that returns a list of pool states for each timepoint
    """
    pass

pool_states = [50 for _ in range(n_obs)]
def n_pool_states_func():
    return pool_states

n_iters = 1000
n_samplers = 1

seed = 0
ehmm = eHMMSampler(initial_hss, model, n_pool_states_func, scheme_order, pool_update_order, epsilon_func)
ehmm.run(n_iters, seed)

chains = [ehmm.samples]
title_dict = {
    "time": dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "class": ehmm.__class__.__name__,
    "iters": n_iters,
    "\n": f"target rate: {acceptance_low} to {acceptance_high}, delta = {delta}",
    "random seed": seed,
    "avg_pool_states": round(np.mean(n_pool_states_func()), 2),
    "scheme": str(scheme_order),
    "updates": str(pool_update_order)}

time_per_sample = round(ehmm.time_per_sample, 3)
print(f"Time per sample: ", time_per_sample)

# use large burnin to allow epsilon to adapt
iters_per_sample = 2
run_diagnostic_suite(model, chains, time_per_sample, title_dict=title_dict, burnin_index=int(0.5 * n_iters * iters_per_sample), middle_timepoint=n_obs // 2, save_figs=True)

pickle_title = f"samples_{title_dict['class']}_{title_dict['time'].replace(':', '').replace(' ', '').replace('-', '')}.pkl"
pickle_dict = {'model':model, 'chains': chains, 'title_dict': title_dict, 'time_per_sample': time_per_sample}
with open(pickle_title, 'wb') as f:
    pickle.dump(pickle_dict, f)