from scipy.stats import multivariate_normal, Covariance
from scipy.special import factorial, gammaln
import numpy as np


class Model:
    """
    abstract base class for a Model
    a Model object represents a state space model, defined in terms of the below unimplemented methods
    first instantiate and generate a realisat. then entire class can be passed to a sampler
    """
    def __init__(self, parameter_dict):
        self.parameter_dict = parameter_dict

    def generate_data(self, n_observations):
        # sample a realisation of the latent process at each timepoint, as well as an observation

        true_hss = [self.sample_initial_states(1)[0]]
        observations = [self.sample_observation(true_hss[0])]

        for _ in range(n_observations - 1):
            true_hss.append(self.sample_transitions([true_hss[-1]])[0])
            observations.append(self.sample_observation(true_hss[-1]))

        self.true_hss = true_hss
        self.observations = observations
        self.n_observations = n_observations

        return true_hss, observations

    def initial_loglikelihood(self, current_x):
        # probability of being in state current_x at first timepoint
        raise NotImplementedError()

    def transition_loglikelihoods(self, deviations):
        # probability of transitioning from current_x to next_x, where the deviation is expected_x - next_x
        # deviations is a list of vectors
        raise NotImplementedError()

    def emission_loglikelihood(self, x, y):
        # likelihood of observing y given x
        raise NotImplementedError()

    def sample_initial_states(self, n_samples):
        # sample `n_samples` initial states at first timepoint
        raise NotImplementedError()

    def sample_transitions(self, current_x):
        # sample a transition from `current_x` to next_x
        raise NotImplementedError()

    def sample_observation(self, current_x):
        # sample an observation given `current_x`
        raise NotImplementedError()

# helper functions for generating the covariance matrices of Model 1
def get_covariance_matrix(n_vars, rho):
    get_value = lambda i, j: 1 if i == j else rho
    covariance_matrix = np.array([[get_value(row, column) for column in range(n_vars)] for row in range(n_vars)])
    return covariance_matrix

def get_covariance_matrix_initial(n_vars, rho, phi):
    get_value = lambda i, j: 1 / (1 - phi ** 2) if i == j else rho / (1 - phi ** 2)
    covariance_matrix_initial = np.array([[get_value(row, column) for column in range(n_vars)] for row in range(n_vars)])
    return covariance_matrix_initial


class Model1(Model):
    """
    Model 1 from Shestopaloff-Neal 2016
    """
    def __init__(self):
        rho = 0.7
        poisson_base = -0.4
        phi = 0.9
        sigma = 0.6
        n_vars = 10
        covariance_matrix = get_covariance_matrix(n_vars, rho)
        covariance_matrix_initial = get_covariance_matrix_initial(n_vars, rho, phi)
        chol = np.linalg.cholesky(covariance_matrix)
        chol_initial = np.linalg.cholesky(covariance_matrix_initial)
        normal_distribution = multivariate_normal(mean=np.zeros(n_vars), cov=Covariance.from_cholesky(chol))
        initial_normal_distribution = multivariate_normal(mean=np.zeros(n_vars), cov=Covariance.from_cholesky(chol_initial))

        model_param_dict = {
            'rho': rho,
            'poisson_base': poisson_base,  # c_j from paper
            'phi': phi,  # using numpy can get away with using a scalar instead of a matrix
            'sigma': sigma,
            'n_vars': n_vars,
            'covariance_matrix_initial': covariance_matrix_initial,
            'covariance_matrix': covariance_matrix,
            'normal_distribution': normal_distribution,
            'initial_normal_distribution': initial_normal_distribution,
            'cholesky_covariance_matrix': chol}

        super().__init__(model_param_dict)

    def initial_loglikelihood(self, current_x):
        return self.parameter_dict['initial_normal_distribution'].logpdf(current_x)

    def transition_loglikelihoods(self, deviations):
        logpdfs = self.parameter_dict['normal_distribution'].logpdf(deviations)
        return logpdfs

    def sample_transitions(self, current_xs):  # vectorised for efficiency
        current_xs = np.array(current_xs).transpose()

        next_means = np.dot(self.parameter_dict['phi'], current_xs)
        deviations = self.parameter_dict['normal_distribution'].rvs(size=current_xs.shape[1]).transpose()  # generate deviations
        next_xs = next_means + deviations

        list_of_column_vectors = [next_xs[:, i] for i in range(next_xs.shape[1])]
        return list_of_column_vectors

    def sample_observation(self, current_x):  # vectorised for efficiency
        lambda_val = np.exp(self.parameter_dict['poisson_base'] + self.parameter_dict['sigma'] * current_x)
        observation = np.random.poisson(lambda_val)
        return observation

    def emission_loglikelihood(self, x, y):
        lambda_values = np.exp(self.parameter_dict['poisson_base'] + self.parameter_dict['sigma'] * x)
        log_likelihoods = y * np.log(lambda_values) - lambda_values - gammaln(y + 1)  # perform using log likelihoods to avoid numerical instability
        total_log_likelihood = np.sum(log_likelihoods)
        return total_log_likelihood

    def sample_initial_states(self, n_samples):
        samples = self.parameter_dict['initial_normal_distribution'].rvs(size=n_samples).transpose()
        samples = [samples[:, i] for i in range(samples.shape[1])]
        return samples
