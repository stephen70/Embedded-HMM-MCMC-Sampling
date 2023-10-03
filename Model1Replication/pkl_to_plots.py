import pickle

filename = 'samples_PGBSAutoregSampler_20230929162029.pkl'  # enter filename here
with open(filename, 'rb') as f:
    pickled_dict = pickle.load(f)

from Tools.DiagnosticTools import *

model = pickled_dict['model']
chains = pickled_dict['chains']
title_dict = pickled_dict['title_dict']
time_per_sample = pickled_dict['time_per_sample']

try:  # some samplers don't have these
    autoreg_acceptance = pickled_dict['autoreg_acceptance']
    shift_acceptance = pickled_dict['shift_acceptance']
except KeyError:
    pass

n_iters = title_dict['iters']
n_obs = len(model.observations)

samples_per_iter = 4
run_diagnostic_suite(model, chains, time_per_sample, burnin_index=int(0.2 * n_iters * samples_per_iter), middle_timepoint=n_obs // 2, title_dict=title_dict, save_figs=True)