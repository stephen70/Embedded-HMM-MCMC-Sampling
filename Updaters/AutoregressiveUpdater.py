import copy
import numpy as np


class AutoregressiveUpdater:
    def __init__(self, model):
        self.model = model
        self.covariance_matrix_initial = self.model.parameter_dict['covariance_matrix_initial']
        self.covariance_matrix = self.model.parameter_dict['covariance_matrix']
        self.phi = self.model.parameter_dict['phi']
        self.n_vars = self.model.parameter_dict['n_vars']
        self.n_obs = self.model.n_observations

        self._instantiate_memoisation_dict()

    def _instantiate_memoisation_dict(self):
        """
        pre-calculate the terms from supplementary material
        """
        self.memoisation_dict = {}

        self.memoisation_dict['phi_matrix'] = np.dot(self.phi, np.identity(self.n_vars))

        term1 = self.memoisation_dict['phi_matrix'] ** 2 + np.matmul(np.linalg.inv(self.covariance_matrix_initial), self.covariance_matrix)
        term1 = np.linalg.inv(term1)
        term1 = np.matmul(term1, self.memoisation_dict['phi_matrix'])
        self.memoisation_dict['term1'] = term1

        term2 = self.memoisation_dict['phi_matrix'] ** 2 + np.identity(self.n_vars)
        term2 = np.linalg.inv(term2)
        term2 = np.matmul(term2, self.memoisation_dict['phi_matrix'])
        self.memoisation_dict['term2'] = term2

        sigma1 = np.matmul(self.memoisation_dict['phi_matrix'], np.matmul(np.linalg.inv(self.covariance_matrix), self.memoisation_dict['phi_matrix'])) + np.linalg.inv(self.covariance_matrix_initial)
        sigma1 = np.linalg.inv(sigma1)
        cholesky1 = np.linalg.cholesky(sigma1)
        self.memoisation_dict['cholesky1'] = cholesky1

        sigma_i = np.matmul(self.memoisation_dict['phi_matrix'], np.matmul(np.linalg.inv(self.covariance_matrix), self.memoisation_dict['phi_matrix'])) + np.linalg.inv(self.covariance_matrix)
        sigma_i = np.linalg.inv(sigma_i)
        cholesky_i = np.linalg.cholesky(sigma_i)
        self.memoisation_dict['cholesky_i'] = cholesky_i

        cholesky_n = np.linalg.cholesky(self.covariance_matrix)
        self.memoisation_dict['cholesky_n'] = cholesky_n

        self.memoisation_dict['cholesky1_ehmm'] = copy.deepcopy(self.memoisation_dict['cholesky1']) * 2.294157339  # additional scaling for no-lookahead

    def perform_update(self, current_x, current_loglikelihood, expected_x, t, epsilon, observations, lookahead=False):
        if lookahead:  # Metropolis
            if t == 0:
                cholesky = self.memoisation_dict['cholesky1']
            elif t < self.n_obs - 1:
                cholesky = self.memoisation_dict['cholesky_i']
            else:
                cholesky = self.memoisation_dict['cholesky_n']
        else:  # eHMM
            if t == 0:
                cholesky = self.memoisation_dict['cholesky1_ehmm']
            else:
                cholesky = self.memoisation_dict['cholesky1']

        current_w = current_x - expected_x
        errors = np.random.normal(0, 1, self.n_vars)

        proposed_w = np.sqrt(1 - epsilon ** 2) * current_w + epsilon * np.matmul(cholesky, errors)

        proposed_x = proposed_w + expected_x

        proposed_loglikelihood = self.model.emission_loglikelihood(proposed_x, observations[t])
        delta = proposed_loglikelihood - current_loglikelihood

        if delta > 0:
            is_accepted = True
            return proposed_x, is_accepted, proposed_loglikelihood
        else:
            threshold = np.exp(delta)
            uniform_number = np.random.uniform()

            if uniform_number <= threshold:
                is_accepted = True
                return proposed_x, is_accepted, proposed_loglikelihood
            else:
                is_accepted = False
                return current_x, is_accepted, current_loglikelihood

