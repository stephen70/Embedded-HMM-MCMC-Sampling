import copy
import numpy as np

from Updaters.AutoregressiveUpdater import AutoregressiveUpdater


class MetropolisUpdater:
    def __init__(self, model):
        self.model = model
        self.covariance_matrix_initial = self.model.parameter_dict['covariance_matrix_initial']
        self.covariance_matrix = self.model.parameter_dict['covariance_matrix']
        self.phi = self.model.parameter_dict['phi']
        self.n_vars = self.model.parameter_dict['n_vars']
        self.n_obs = self.model.n_observations

        self._autoreg_updater = AutoregressiveUpdater(self.model)

    def perform_update(self, current_hss, observations, epsilon):  # one update = update all timepoints in sequence form start ot end
        current_hss = copy.deepcopy(current_hss)

        acceptances = [0 for _ in range(self.n_obs)]
        for t in range(self.n_obs):
            current_emission_loglikelihood = self.model.emission_loglikelihood(current_hss[t], observations[t])
            if t == 0:  # expected x given ancestor differs based on timepoint
                expected_x_given_ancestor = np.matmul(self._autoreg_updater.memoisation_dict['term1'], current_hss[1])
                next_x, is_accepted, current_emission_loglikelihood = self._autoreg_updater.perform_update(current_hss[t], current_emission_loglikelihood, expected_x_given_ancestor, t, epsilon, observations, lookahead=True)
                if is_accepted:
                    current_hss[t] = next_x
                    acceptances[t] += 1

            elif t < self.n_obs - 1:
                sum = current_hss[t - 1] + current_hss[t + 1]
                expected_x_given_ancestor = np.matmul(self._autoreg_updater.memoisation_dict['term2'], sum)
                next_x, is_accepted, current_emission_loglikelihood = self._autoreg_updater.perform_update(current_hss[t], current_emission_loglikelihood, expected_x_given_ancestor, t, epsilon, observations, lookahead=True)
                if is_accepted:
                    current_hss[t] = next_x
                    acceptances[t] += 1
            else:
                expected_x_given_ancestor = self.model.parameter_dict['phi'] * current_hss[t - 1]
                next_x, is_accepted, current_emission_loglikelihood = self._autoreg_updater.perform_update(current_hss[t], current_emission_loglikelihood, expected_x_given_ancestor, t, epsilon, observations, lookahead=True)
                if is_accepted:
                    current_hss[t] = next_x
                    acceptances[t] += 1

        return current_hss, acceptances



