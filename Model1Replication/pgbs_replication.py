import pickle
import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Samplers.PGBSSampler import PGBSSampler
from Tools.DiagnosticTools import *
from Models import Models

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
n_particles = 250
n_iters = 14000

n_samplers = 5
seeds = range(n_samplers)
pfs = [PGBSSampler(initial_hss, model, n_particles) for _ in range(n_samplers)]
pfs = Parallel(n_jobs=n_samplers, backend='multiprocessing')(delayed(pf.run)(n_iters, seed) for pf, seed in zip(pfs, seeds))

chains = [pf.samples for pf in pfs]
title_dict = {
    "time": dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    "class": pfs[0].__class__.__name__,
    "\n": "",
    "iters": n_iters,
    "n_particles": n_particles}

time_per_sample = round(np.mean([pf.time_per_sample for pf in pfs]), 3)
samples_per_iter = 2
run_diagnostic_suite(model, chains, time_per_sample, burnin_index=int(0.2 * n_iters * samples_per_iter), middle_timepoint=n_obs // 2, title_dict=title_dict, save_figs=True)

pickle_title = f"samples_{title_dict['class']}_{title_dict['time'].replace(':', '').replace(' ', '').replace('-', '')}.pkl"
pickle_dict = {'model': model, 'chains': chains, 'title_dict': title_dict, 'time_per_sample': time_per_sample}
with open(pickle_title, 'wb') as f:
    pickle.dump(pickle_dict, f)

