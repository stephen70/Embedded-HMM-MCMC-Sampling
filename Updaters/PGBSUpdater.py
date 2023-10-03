import copy
from typing import List
import numpy as np


class Particle:
    def __init__(self, x, weight, ancestor):
        self.x = x
        self.logweight = weight
        self.ancestor_index = ancestor

    def __repr__(self):
        return f"Particle(x={self.x}, logweight={self.logweight}, ancestor={self.ancestor_index})"


class Pool:
    def __init__(self, particles: List[Particle]):
        self.particles = particles

    @property
    def normalised_weights(self) -> np.array:
        logweights = np.array([particle.logweight for particle in self.particles])
        return normalise_log_likelihoods(logweights)

    def __repr__(self):
        return f"Pool(particles={self.particles})"

    def __getitem__(self, item):
        return self.particles[item]

    def __len__(self):
        return len(self.particles)


def normalise_log_likelihoods(log_likelihoods: np.array) -> np.array:
    """
    convert a list of log likelihoods to a normalised list of weights proportional to likelihoods
    use max log likelihood trick to avoid numerical instability
    """
    max_ll = max(log_likelihoods)
    shifted_lls = log_likelihoods - max_ll
    unnormalized_ls = np.exp(shifted_lls)

    total = np.sum(unnormalized_ls)
    normalised_ls = unnormalized_ls / total
    return normalised_ls


class PGBSUpdater:
    def __init__(self, model, n_particles):
        self.model = model
        self.n_particles = n_particles

        self.n_vars = self.model.parameter_dict['n_vars']
        self.n_observations = self.model.n_observations

    def perform_update(self, current_hss, direction):
        current_hss = copy.deepcopy(current_hss)
        if direction == 'R':  # reverse update, so reverse the current hss and observations
            current_hss = current_hss[::-1]
            observations = self.model.observations[::-1]
        else:
            observations = self.model.observations

        pools = []

        # t = 0
        x_values = [current_hss[0]]
        x_values.extend(self.model.sample_initial_states(self.n_particles - 1))
        logweights = [self.model.emission_loglikelihood(x, observations[0]) for x in x_values]

        pool1 = Pool([Particle(x, logweight, ancestor) for x, logweight, ancestor in zip(x_values, logweights, [-1 for _ in range(self.n_particles)])])
        pools.append(pool1)

        # t > 0
        for t in range(1, self.n_observations):
            x_values = [current_hss[t]]
            ancestor_indexes = [0]

            sampled_indexes = np.random.choice(range(self.n_particles), size=self.n_particles - 1, p=pools[t - 1].normalised_weights, replace=True)
            ancestor_indexes.extend(sampled_indexes)

            ancestor_values = [pools[t - 1][ancestor_index].x for ancestor_index in ancestor_indexes]
            sampled_x_values = self.model.sample_transitions(ancestor_values)
            x_values.extend(sampled_x_values)

            logweights = [self.model.emission_loglikelihood(x, observations[t]) for x in x_values]
            pool = Pool([Particle(x, logweight, ancestor) for x, logweight, ancestor in zip(x_values, logweights, ancestor_indexes)])
            pools.append(pool)

        next_hss = self._sample_backwards(pools)

        if direction == 'R':  # undo the latent sequence reversal
            next_hss = next_hss[::-1]
        return next_hss

    def _sample_backwards(self, pools):
        next_hss = [np.nan for _ in range(self.n_observations)]

        # sample a final state proportional to weight
        final_state_index = np.random.choice(range(self.n_particles), p=pools[-1].normalised_weights, replace=True)
        next_hss[-1] = pools[-1][final_state_index].x

        for t in range(self.n_observations - 2, -1, -1):
            deviations = np.array([pool_state.x for pool_state in pools[t]])
            deviations *= self.model.parameter_dict['phi']
            deviations -= next_hss[t + 1]

            transition_loglikelihoods = self.model.transition_loglikelihoods(deviations)

            sampling_pmf = [particle.logweight + transition_loglikelihood for particle, transition_loglikelihood in zip(pools[t].particles, transition_loglikelihoods)]
            sampling_pmf = normalise_log_likelihoods(np.array(sampling_pmf))

            chosen_state_index = np.random.choice(range(self.n_particles), p=sampling_pmf, replace=True)
            sampled_state = pools[t][chosen_state_index].x
            next_hss[t] = sampled_state

        return next_hss
