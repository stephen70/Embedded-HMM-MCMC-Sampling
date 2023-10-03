import copy
import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Tools.DiagnosticTools import *
from Models import Models
from Samplers.AbstractSampler import AbstractSampler
from Updaters.eHMMUpdater import eHMMUpdater

"""
eHMM sampler from Shestopaloff-Neal 2018
"""


class eHMMSampler(AbstractSampler):
    def __init__(self, initial_hss, model, n_pool_states_func, scheme_order, pool_update_order, epsilon_func):
        super().__init__(initial_hss, model)
        self.n_pool_states_func = n_pool_states_func
        self._pool_update_list = pool_update_order
        self._scheme_list = scheme_order
        self.epsilon_func = epsilon_func

        self._ehmm_updater = eHMMUpdater(self, self.model, self.epsilon_func)

    def iterate(self, iter):
        """
        1 iteration = 1 forward pass, 1 forward pass on reversed sequence
        each iteration produces len(self._scheme_list) samples
        """
        self.n_pool_states_list = self.n_pool_states_func()  # get number of pool states to use for each timepoint in this iteration

        for scheme in self._scheme_list:
            if scheme == 'R':
                current_hss = self.current_hss[::-1]
                observations = self.model.observations[::-1]
            elif scheme == 'NR':
                current_hss = self.current_hss
                observations = self.model.observations
            else:
                raise ValueError(f"Unknown scheme value {scheme[0]}")

            next_hss = self._ehmm_updater.perform_forward_update(current_hss, observations, self.n_pool_states_list, iter)

            if scheme[0] == 'R':
                self.current_hss = copy.deepcopy(next_hss[::-1])
            else:
                self.current_hss = copy.deepcopy(next_hss)

            self.samples.append(copy.deepcopy(self.current_hss))
