import copy
import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Models import Models
from Tools.DiagnosticTools import *
from Samplers.AbstractSampler import AbstractSampler
from Updaters.MetropolisUpdater import MetropolisUpdater


class MetropolisSampler(AbstractSampler):
    def __init__(self, initial_hss, model, epsilon_func, thinning_factor):
        super().__init__(initial_hss, model)
        self._metrop_updater = MetropolisUpdater(self.model)
        self.epsilon_func = epsilon_func
        self.thinning_factor = thinning_factor

        self.autoreg_performed = [0 for _ in range(self.n_observations)]
        self.autoreg_accepted = [0 for _ in range(self.n_observations)]

    def iterate(self, iter):
        """
        1 update = update all timepoints in sequence from start to end
        1 iteration = update `thinning_factor` times, and only store last sequence as sample
        """
        for _ in range(self.thinning_factor):
            epsilon = self.epsilon_func()
            next_hss, acceptances = self._metrop_updater.perform_update(self.current_hss, self.model.observations, epsilon)

            for t in range(self.n_observations):
                self.autoreg_accepted[t] += acceptances[t]
                self.autoreg_performed[t] += 1

            self.current_hss = next_hss
        self.samples.append(copy.deepcopy(next_hss))

