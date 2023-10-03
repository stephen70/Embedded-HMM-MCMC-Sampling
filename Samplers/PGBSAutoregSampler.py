import copy
from joblib import Parallel, delayed
import pandas as pd
import datetime as dt

from Models import Models
from Samplers.AbstractSampler import AbstractSampler
from Updaters.MetropolisUpdater import MetropolisUpdater
from Updaters.PGBSUpdater import PGBSUpdater
from Tools.DiagnosticTools import *


class PGBSAutoregSampler(AbstractSampler):
    """
    PGBS + Metropolis sampler from Shestopaloff-Neal 2018
    assumes q_1 = initial x distribution, q_i = transition distribution
    """
    def __init__(self, initial_hss, model, n_particles, epsilon_func):
        super().__init__(initial_hss, model)
        self.n_particles = n_particles
        self.epsilon_func = epsilon_func

        self._pgbs_updater = PGBSUpdater(self.model, self.n_particles)
        self._metrop_updater = MetropolisUpdater(self.model)

    def iterate(self, iter):
        """
        1 iteration = 1 PGBS forward update, 10 Metropolis updates, 1 PGBS reverse update, 10 Metropolis updates
        each iteration generates four samples
        """
        next_hss = self._pgbs_updater.perform_update(self.current_hss, direction='NR')
        self.samples.append(copy.deepcopy(next_hss))

        for _ in range(10):
            epsilon = self.epsilon_func()
            next_hss, _ = self._metrop_updater.perform_update(next_hss, self.model.observations, epsilon)
        self.samples.append(copy.deepcopy(next_hss))

        next_hss = self._pgbs_updater.perform_update(next_hss, direction='R')
        self.samples.append(copy.deepcopy(next_hss))

        for _ in range(10):
            epsilon = self.epsilon_func()
            next_hss, _ = self._metrop_updater.perform_update(next_hss, self.model.observations, epsilon)
        self.samples.append(copy.deepcopy(next_hss))

        self.current_hss = next_hss

