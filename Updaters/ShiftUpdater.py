import numpy as np


class ShiftUpdater:
    def __init__(self, model):
        self.model = model
        self.covariance_matrix_initial = self.model.parameter_dict['covariance_matrix_initial']
        self.covariance_matrix = self.model.parameter_dict['covariance_matrix']
        self.phi = self.model.parameter_dict['phi']
        self.n_vars = self.model.parameter_dict['n_vars']
        self.n_obs = self.model.n_observations

    def perform_update(self, pools, n_pool_states_list, current_pool_state, current_loglikelihood, t, observations) -> ('PoolState', bool):
        from Updaters.eHMMUpdater import PoolState  # to avoid circular import

        proposed_index = np.random.randint(n_pool_states_list[t - 1])

        deviation = current_pool_state.x - self.phi * pools[t - 1][2][current_pool_state.index].x
        proposed_x = self.phi * pools[t - 1][2][proposed_index].x + deviation
        proposed_loglikelihood = self.model.emission_loglikelihood(proposed_x, observations[t])
        delta = proposed_loglikelihood - current_loglikelihood

        if delta > 0:
            is_accepted = True
            return PoolState(proposed_x, proposed_index), is_accepted, proposed_loglikelihood
        else:
            threshold = np.exp(delta)
            uniform_number = np.random.uniform()

            if uniform_number <= threshold:
                is_accepted = True
                return PoolState(proposed_x, proposed_index), is_accepted, proposed_loglikelihood
            else:
                is_accepted = False
                return current_pool_state, is_accepted, current_loglikelihood


