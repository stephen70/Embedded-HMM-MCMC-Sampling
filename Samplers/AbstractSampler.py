import copy
from abc import ABCMeta
import numpy as np
import time
from tqdm import tqdm

from Models.Models import Model


class AbstractSampler(metaclass=ABCMeta):
    """
    Abstract base class for samplers
    """
    def __init__(self, initial_hss: np.array, model: Model):
        self.initial_hss = initial_hss
        self.model = model

        self.n_observations = len(self.model.observations)
        self.current_hss = initial_hss
        self.samples = [copy.deepcopy(initial_hss)]
        self.n_iterations_performed = 0
        self.time_per_sample = None

        self.acceptance_rates_by_timepoint = [0 for _ in range(self.n_observations)]

    def calc_acceptance_rates_by_timepoint(self):
        if self.n_iterations_performed == 0:
            return [0 for _ in range(self.n_observations)]
        else:
            rates = [round(x * 100 / self.n_iterations_performed, 0) for x in self.acceptance_rates_by_timepoint]
            return rates

    def run(self, n_iterations, seed=None):
        if seed is not None:
            np.random.seed(seed)

        # self.samples could be preallocated here, but efficiency gain will be minimal
        start_time = time.time()
        for i in tqdm(range(n_iterations)):  # use tqdm to show progress bar for convenience
            self.iterate(i)
            self.n_iterations_performed += 1
        end_time = time.time()

        self.time_per_sample = np.round((end_time - start_time) / len(self.samples), 3)
        return self

    def iterate(self, iter):  # to be implemented by subclasses
        raise NotImplementedError()

    @staticmethod
    def normalise_log_likelihoods(log_likelihoods: np.array):
        """
        convert a list of log likelihoods to a normalised list of weights proportional to likelihoods
        """
        max_ll = max(log_likelihoods)
        shifted_lls = log_likelihoods - max_ll  # use max log likelihood trick to avoid numerical instability
        unnormalized_ls = np.exp(shifted_lls)

        total = np.sum(unnormalized_ls)
        normalised_ls = unnormalized_ls / total
        return normalised_ls

