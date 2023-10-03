import pickle
import pandas as pd
from joblib import Parallel, delayed

from Models import Models
from Samplers.MetropolisSampler import MetropolisSampler
from Tools.DiagnosticTools import *
import datetime as dt


def epsilon_generator():
    while True:
        yield 0.2
        yield 0.8

epsilon_generator = epsilon_generator()
def epsilon_func():
    return next(epsilon_generator)

np.random.seed(5)

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
n_iters = 20000
thinning_factor = 10

n_chains = 5
seeds = range(n_chains)  # seeds are preserved when run in parallel
metros = [MetropolisSampler(initial_hss, model, epsilon_func, thinning_factor) for _ in range(n_chains)]

metros = Parallel(n_jobs=n_chains, backend='multiprocessing')(delayed(metro.run)(n_iters, seed) for metro, seed in zip(metros, seeds))

autoreg_acceptance_all = [sum(x) for x in zip(*[metro.autoreg_accepted for metro in metros])]
autoreg_performed_all = [sum(x) for x in zip(*[metro.autoreg_performed for metro in metros])]
autoreg_acceptance = [round(100 * x / y, 0) if y != 0 else None for x, y in zip(autoreg_acceptance_all, autoreg_performed_all)]
print("Autoreg acceptance rate: ", autoreg_acceptance)
print("Autoreg min, max: ", min([x for x in autoreg_acceptance if x is not None]), max([x for x in autoreg_acceptance if x is not None]))

chains = [metro.samples for metro in metros]
title_dict = {
    "time": dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "class": metros[0].__class__.__name__,
    "\n": "",
    "iters": n_iters}

time_per_sample = round(np.mean([metro.time_per_sample for metro in metros]), 3)  # multiply by thinning factor to get actual time per sample
samples_per_iter = 1
run_diagnostic_suite(model, chains, time_per_sample, burnin_index=int(0.2 * n_iters * samples_per_iter), middle_timepoint=n_obs // 2, title_dict=title_dict, save_figs=True)

pickle_title = f"samples_{title_dict['class']}_{title_dict['time'].replace(':', '').replace(' ', '').replace('-', '')}.pkl"
pickle_dict = {'model': model, 'chains': chains, 'title_dict': title_dict, 'time_per_sample': time_per_sample}
with open(pickle_title, 'wb') as f:
    pickle.dump(pickle_dict, f)

