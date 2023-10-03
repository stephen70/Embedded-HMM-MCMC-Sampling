import matplotlib.pyplot as plt
import numpy as np


def traceplot_chains(chains, model, timepoint, title_dict, save_figs, limit_length=False):
    if limit_length and len(chains[0]) > 5000:
        chains = [chain[:5000] for chain in chains]
        print("Limiting traceplot length to 5000 points")

    n_samples = len(chains[0])
    n_vars = len(chains[0][0][0])

    grid_size = int(np.ceil(np.sqrt(n_vars)))  # find the smallest square grid that can fit all subplots
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(12, 12), dpi=200)

    axes = axes.flatten()

    scatter_size = 1.2
    for var_index in range(n_vars):
        var_timepoint_mean = round(np.mean([[hss[timepoint][var_index] for hss in chain] for chain in chains]), 2)
        for chain_index, chain in enumerate(chains):
            axes[var_index].scatter(range(n_samples), [hss[timepoint][var_index] for hss in chain], label=f"Chain {chain_index}", s=scatter_size)
            axes[var_index].axhline([model.true_hss[timepoint][var_index]], color='red', label="True value")
        axes[var_index].set_title(f"Var {var_index}, mean: {var_timepoint_mean}")

    # If there are more subplots than images, turn off extra subplots
    for j in range(n_vars, grid_size * grid_size):
        axes[j].axis('off')

    title_elems = [key + ": " + str(value) for key, value in title_dict.items()]
    title_str = ' - '.join(title_elems)
    fig.suptitle(f"Traceplot(s) at timepoint {timepoint} \n {title_str}", wrap=True)

    plt.legend()
    plt.tight_layout()
    if save_figs:
        plt.savefig(f"traceplot_{title_dict['class']}_{timepoint}_{title_dict['time'].replace(':', '').replace(' ', '').replace('-', '')}.png")
    plt.show()


def calc_autocorrelation_time_variable(values_by_chain, var_mean, lags_to_use=30):
    """
    calculate autocorrelation time of a single variable over `lags_to_use` lags
    """
    n_samples = len(values_by_chain[0])

    if n_samples < lags_to_use:
        lags_to_use = n_samples - 1

    def calc_gamma(k, hss):
        products = []
        for i in range(n_samples - k - 1):
            products.append((hss[i] - var_mean) * (hss[i + k] - var_mean))
        gamma_k = np.sum(products) / n_samples  # technically should divide by (n_timepoints - k), but cancels out when calculating rho_k
        return gamma_k

    covariances = []
    for k in range(lags_to_use):
        cov_ks = [calc_gamma(k, chain) for chain in values_by_chain]
        covariances.append(np.mean(cov_ks))

    rhos = [covariances[k] / covariances[0] for k in range(0, lags_to_use)]
    autocorrelation_time = 1 + 2 * np.sum(rhos)

    return autocorrelation_time

def calc_autocorrelation_times(chains):
    """
    estimates the autocorrelation time of each variable x_ij, where i is time and j is the variable index
    the mean(x_ij) is taken over all chains and samples
    autocovariance(x_ij) is first calculated for each chain, then averaged over all chains
    """
    print("Calculating autocorrelation times...")

    n_timepoints = len(chains[0][0])
    n_vars = len(chains[0][0][0])

    autocorr_times = []
    for t in range(n_timepoints):
        autocorr_times_current_timepoint = []
        for var_index in range(n_vars):
            values = [[hss[t][var_index] for hss in chain] for chain in chains]
            var_mean = np.mean(values)
            autocorr_time = calc_autocorrelation_time_variable(values, var_mean)

            autocorr_times_current_timepoint.append(autocorr_time)

        autocorr_times.append(autocorr_times_current_timepoint)

    return autocorr_times


def plot_prior_and_posterior_histogram(model, chains, is_latent_state_univariate, timepoint=1, limit_length=True):
    """
    for a single real-valued variable, plot the prior and posterior distributions
    prior is found by generating samples from the model latent process at the given timepoint
    posterior samples are taken from the chains
    """
    n_chains = len(chains)

    n_samples_per_chain = len(chains[0])
    if limit_length and n_samples_per_chain * n_chains > 1000:  # limit number of points to plot to 1000
        chains = [chain[:int(1000 / n_chains)] for chain in chains]
        print("Limiting number of points to plot to 1000")

    n_samples_per_chain = len(chains[0])
    n_timepoints = len(chains[0][0])

    n_vars = len(chains[0][0][0])
    max_vars_to_plot = 5

    fig, axes = plt.subplots(1, min(n_vars, max_vars_to_plot), figsize=(15, 15), dpi=200)

    prior_samples = [model.generate_data(n_timepoints)[0][timepoint] for _ in range(n_samples_per_chain * n_chains)]
    if is_latent_state_univariate:
        prior_samples = [[sample] for sample in prior_samples]

    posterior_samples = [hss[timepoint] for chain in chains for hss in chain]

    for var_index in range(n_vars):
        if var_index > max_vars_to_plot - 1:
            break

        if is_latent_state_univariate:
            current_axis = axes
        else:
            current_axis = axes[var_index]

        prior_samples_var = [sample[var_index] for sample in prior_samples]
        posterior_samples_var = [sample[var_index] for sample in posterior_samples]

        current_axis.hist(prior_samples_var, label="Prior", alpha=0.5, color="blue")
        current_axis.hist(posterior_samples_var, label="Posterior", alpha=0.5, color="orange")

        current_axis.set_title(f"Variable {var_index} prior and posterior")
        current_axis.legend()

    plt.tight_layout()
    plt.show()


def plot_autocorrelation_times(autocorrelation_times, time_per_sample, title_dict, save_figs):
    n_vars = len(autocorrelation_times[0])
    fig, axis = plt.subplots(figsize=(16, 12), dpi=200)
    for var_index in range(n_vars):
        axis.plot([autocorrelation_time[var_index] * time_per_sample for autocorrelation_time in autocorrelation_times], label="Var " + str(var_index))

    title_elems = [key + ": " + str(value) for key, value in title_dict.items()]
    title_str = ' - '.join(title_elems)

    fig.suptitle(f"seconds per sample: {time_per_sample}s - UAT: {round(np.mean(autocorrelation_times), 3)} - AAT: {round(np.mean(autocorrelation_times) * time_per_sample, 3)} \n {title_str}", wrap=True)
    plt.tight_layout()
    plt.legend()
    if save_figs:
        plt.savefig(f"autocorr_{title_dict['class']}_{title_dict['time'].replace(':', '').replace(' ', '').replace('-', '')}.png")
    plt.show()


def run_diagnostic_suite(model, chains, time_per_sample, burnin_index=0, middle_timepoint=3, title_dict={}, save_figs=False):
    """
    for first, middle and last timepoint make traceplots of all dimensions
    then plot autocorrelation at all timepoints for all dimensions
    """
    chains = [chain[burnin_index:] for chain in chains]  # remove burnin samples

    for timepoint in [0, middle_timepoint, model.n_observations - 1]:
        traceplot_chains(chains, model, timepoint, title_dict, save_figs)

    autocorrelation_times = calc_autocorrelation_times(chains)
    plot_autocorrelation_times(autocorrelation_times, time_per_sample, title_dict, save_figs)