import copy
from typing import List

import pandas as pd
import datetime as dt
from joblib import Parallel, delayed

from Tools.DiagnosticTools import *
from Models import Models
from Samplers.AbstractSampler import AbstractSampler
from Updaters.PGBSUpdater import PGBSUpdater


class PGBSSampler(AbstractSampler):
    """
    PGBS + Metropolis sampler from Shestopaloff-Neal 2018
    assumes q_1 = initial x distribution, q_i = transition distribution
    """
    def __init__(self, initial_hss, model, n_particles):
        super().__init__(initial_hss, model)
        self.n_particles = n_particles

        self._pgbs_updater = PGBSUpdater(self.model, self.n_particles)

    def iterate(self, iter):
        """
        1 iteration = 1 PGBS forward update, 1 PGBS reverse update
        each iteration generates two samples
        """
        next_hss = self._pgbs_updater.perform_update(self.current_hss, direction='NR')
        self.samples.append(copy.deepcopy(next_hss))
        self.current_hss = next_hss

        next_hss = self._pgbs_updater.perform_update(self.current_hss, direction='R')
        self.samples.append(copy.deepcopy(next_hss))
        self.current_hss = next_hss

