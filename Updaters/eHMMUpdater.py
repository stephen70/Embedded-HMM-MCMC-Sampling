import copy
from typing import List
import numpy as np

from Updaters.AutoregressiveUpdater import AutoregressiveUpdater
from Updaters.ShiftUpdater import ShiftUpdater


class PoolState:
    def __init__(self, x, index):
        self.x = x
        self.index = index

    def __repr__(self):
        return f"PoolState(x={self.x}, index={self.index})"


class Pool:
    def __init__(self, pool_states: List[PoolState]):
        self.pool_states = pool_states

    @property
    def x_values(self):
        return [pool_state.x for pool_state in self.pool_states]

    @property
    def indexes(self):
        return [pool_state.index for pool_state in self.pool_states]

    def add(self, pool_state: PoolState):
        self.pool_states.append(pool_state)

    def __repr__(self):
        return f"Pool({self.pool_states})"

    def __getitem__(self, item):
        return self.pool_states[item]

    def __len__(self):
        return len(self.pool_states)


class eHMMUpdater:
    def __init__(self, sampler, model, epsilon_func):
        self.sampler = sampler
        self.model = model
        self.epsilon_func = epsilon_func

        self.phi = self.model.parameter_dict['phi']
        self.n_obs = self.model.n_observations

        self._autoreg_updater = AutoregressiveUpdater(self.model)
        self._shift_updater = ShiftUpdater(self.model)
        self.previous_epsilons = [None for _ in range(self.n_obs)]

        self.autoreg_performed = [0 for _ in range(self.n_obs)]
        self.autoreg_accepted = [0 for _ in range(self.n_obs)]
        self.shift_performed = [0 for _ in range(self.n_obs)]
        self.shift_accepted = [0 for _ in range(self.n_obs)]

    def _generate_pool_states(self, current_state, n_desired_states, t, pools, observations, pool_update_list, n_pool_states_list, iter):
        """
        generates the set of pool states at timepoint `t` using the updates in `pool_update_list`
        uses the distinct state optimisation trick
        """
        pool_states_distinct = []
        pool_counts = []

        current_emission_loglikelihood = self.model.emission_loglikelihood(current_state.x, observations[t])
        for i in range(n_desired_states):

            state_changed = False  # keep track of whether a new state is obtained

            for update_type in pool_update_list:
                if update_type == 'autoregressive':  # string comparison is fast, strings are likely to be interned or otherwise compiler optimised

                    current_acceptance_rate = self.autoreg_accepted[t] / self.autoreg_performed[t] if self.autoreg_performed[t] > 0 else 0
                    previous_epsilon = self.previous_epsilons[t]

                    epsilon = self.epsilon_func(
                        current_acceptance_rate=current_acceptance_rate,
                        n_pool_states=n_pool_states_list[t],
                        previous_epsilon=previous_epsilon,
                        iter=iter)
                    self.previous_epsilons[t] = epsilon
                    ancestor_index = current_state.index

                    # if using f + F
                    if t == 0:
                        expected_x_given_ancestor = 0
                    else:
                        expected_x_given_ancestor = self.model.parameter_dict['phi'] * pools[t - 1][2][ancestor_index].x

                    current_x, is_accepted, current_emission_loglikelihood = self._autoreg_updater.perform_update(
                        current_state.x, current_emission_loglikelihood, expected_x_given_ancestor, t, epsilon, observations, lookahead=False)
                    current_state = PoolState(current_x, ancestor_index)

                    self.autoreg_performed[t] += 1
                    if is_accepted:
                        state_changed = True
                        self.autoreg_accepted[t] += 1

                elif update_type == 'shift':
                    current_state, is_accepted, current_emission_loglikelihood = self._shift_updater.perform_update(
                        pools, n_pool_states_list, current_state, current_emission_loglikelihood, t, observations)

                    self.shift_performed[t] += 1
                    if is_accepted:
                        state_changed = True
                        self.shift_accepted[t] += 1

                else:
                    raise ValueError(f"Unknown pool update {update_type}")

            # optimisation trick to evaluate only unique states
            if state_changed:
                pool_states_distinct.append(current_state)
                pool_counts.append(1)
            else:
                if i == 0:
                    pool_states_distinct.append(current_state)
                    pool_counts.append(1)
                else:
                    pool_counts[-1] += 1

        # store the repeated states for efficient indexing later
        repeated_states = [x for x, y in zip(pool_states_distinct, pool_counts) for _ in range(y)]
        return pool_states_distinct, pool_counts, repeated_states

    def perform_forward_update(self, current_hss, observations, n_pool_states_list, iter):
        pools = [np.nan for _ in range(self.n_obs)]

        for t in range(self.n_obs):  # construct pools for each time point
            starting_index = np.random.randint(n_pool_states_list[t])  # J from paper
            n_forward_updates = n_pool_states_list[t] - starting_index - 1
            n_backward_updates = n_pool_states_list[t] - n_forward_updates - 1

            if t == 0:
                current_state = PoolState(copy.deepcopy(current_hss[t]), -1)
                pool_update_list = ['autoregressive']  # at first timepoint use a single autoregressive update, as shift doesn't work
            else:
                deviations = np.array([self.model.parameter_dict['phi'] * pool_state.x - current_hss[t] for pool_state in pools[t - 1][0]])

                selection_densities = self.model.transition_loglikelihoods(deviations)
                selection_densities = [x for x, y in zip(selection_densities, pools[t - 1][1]) for _ in range(y)]
                selection_densities_normed = self.sampler.normalise_log_likelihoods(selection_densities)

                initial_ancestor_index = np.random.choice(range(n_pool_states_list[t - 1]), p=selection_densities_normed)
                current_state = PoolState(copy.deepcopy(current_hss[t]), initial_ancestor_index)

                pool_update_list = self.sampler._pool_update_list

            backwards_pool_states_distinct, backwards_state_counts, backwards_repeated_states = self._generate_pool_states(
                current_state, n_backward_updates, t, pools, observations, pool_update_list[::-1], n_pool_states_list, iter)
            forward_pool_states_distinct, forwards_state_counts, forwards_repeated_states = self._generate_pool_states(
                current_state, n_forward_updates, t, pools, observations, pool_update_list, n_pool_states_list, iter)

            pool_states_distinct = backwards_pool_states_distinct + [copy.deepcopy(current_state)] + forward_pool_states_distinct
            state_counts = backwards_state_counts + [1] + forwards_state_counts  # [1] for the initial state
            repeated_states = backwards_repeated_states + [copy.deepcopy(current_state)] + forwards_repeated_states

            pools[t] = [pool_states_distinct, state_counts, repeated_states]

        next_hss = self._sample_backwards(pools, n_pool_states_list)
        return next_hss

    def _sample_backwards(self, pools, n_pool_states_list):
        next_hss = [np.nan for _ in range(self.n_obs)]
        for t in reversed(range(0, self.n_obs)):

            if t == self.n_obs - 1:
                chosen_pool_state_index = np.random.randint(n_pool_states_list[self.n_obs - 1])
                next_hss[t] = pools[t][2][chosen_pool_state_index].x
            else:
                deviations = np.array([self.model.parameter_dict['phi'] * pool_state.x - next_hss[t + 1] for pool_state in pools[t][0]])

                selection_densities = self.model.transition_loglikelihoods(deviations)
                selection_densities = [x for x, y in zip(selection_densities, pools[t][1]) for _ in range(y)]
                selection_densities_normed = self.sampler.normalise_log_likelihoods(selection_densities)

                chosen_pool_state_index = np.random.choice(n_pool_states_list[t], p=selection_densities_normed)  # sample a pool state index proportional to transition likelihood
                next_hss[t] = pools[t][2][chosen_pool_state_index].x
        return next_hss
