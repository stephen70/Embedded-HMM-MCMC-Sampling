import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Models import Models
from Samplers.eHMMSampler import eHMMSampler
from Tools.DiagnosticTools import *

scheme_order = ['NR', 'R']
pool_update_order = ['shift', 'autoregressive']

np.random.seed(5)

n_obs = 250
model = Models.Model1()
model.generate_data(n_obs)
all_sheets = pd.read_excel('../source.xlsx', sheet_name=None, header=None)
x_df = all_sheets['x']
y_df = all_sheets['y']
initial_hss = [np.zeros(model.parameter_dict['n_vars'], dtype=np.float64) for _ in range(n_obs)]
model.true_hss = [x_df[col].values for col in x_df.columns]
model.observations = [y_df[col].values for col in y_df.columns]
model.n_observations = len(model.observations)


acceptance_low = 0.4
acceptance_high = 0.5
delta = 0.25

def epsilon_func(**kwargs):
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

basic_pool_states = [50 for _ in range(n_obs)]

def n_pool_states_func():
    return basic_pool_states

n_iters = 4000
n_samplers = 5
seeds = range(n_samplers)

ehmms = [eHMMSampler(initial_hss, model, n_pool_states_func, scheme_order, pool_update_order, epsilon_func) for _ in range(n_samplers)]

ehmms = Parallel(n_jobs=n_samplers, backend='multiprocessing')(delayed(ehmm.run)(n_iters, seed) for ehmm, seed in zip(ehmms, seeds))

chains = [ehmm.samples for ehmm in ehmms]

autoreg_acceptance_all = [sum(x) for x in zip(*[ehmm._ehmm_updater.autoreg_accepted for ehmm in ehmms])]
autoreg_performed_all = [sum(x) for x in zip(*[ehmm._ehmm_updater.autoreg_performed for ehmm in ehmms])]
shift_acceptance_all = [sum(x) for x in zip(*[ehmm._ehmm_updater.shift_accepted for ehmm in ehmms])]
shift_performed_all = [sum(x) for x in zip(*[ehmm._ehmm_updater.shift_performed for ehmm in ehmms])]
autoreg_acceptance = [round(100 * x / y, 0) if y != 0 else None for x, y in zip(autoreg_acceptance_all, autoreg_performed_all)]
shift_acceptance = [round(100 * x / y, 0) if y != 0 else None for x, y in zip(shift_acceptance_all, shift_performed_all)]
print("Autoreg acceptance rate: ", autoreg_acceptance)
print("Autoreg min, max: ", min([x for x in autoreg_acceptance if x is not None]), max([x for x in autoreg_acceptance if x is not None]))
print("Shift acceptance rate: ", shift_acceptance)
print("Shift min, max: ", min([x for x in shift_acceptance if x is not None]), max([x for x in shift_acceptance if x is not None]))

title_dict = {
    "time": dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "class": ehmms[0].__class__.__name__,
    "iters": n_iters,
    "\n": f"target rate: {acceptance_low} to {acceptance_high}, delta = {delta}",
    "pool_states": 50,
    "scheme": str(scheme_order),
    "updates": str(pool_update_order)}

time_per_sample = round(np.mean([ehmm.time_per_sample for ehmm in ehmms]), 3)
samples_per_iter = 2
run_diagnostic_suite(model, chains, time_per_sample, title_dict=title_dict, burnin_index=int(0.2 * n_iters * samples_per_iter), middle_timepoint=n_obs // 2, save_figs=True)
