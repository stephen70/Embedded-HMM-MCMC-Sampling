import pickle
import matplotlib.pyplot as plt
from Tools.DiagnosticTools import *
import numpy as np

plt.rcParams["mathtext.fontset"] = "cm"

# import adaptive samples
filename = ''  # enter filename here
with open(filename, 'rb') as f:
    pickled_dict = pickle.load(f)

model = pickled_dict['model']
chains = pickled_dict['chains']
title_dict = pickled_dict['title_dict']
n_iters = title_dict['iters']

samples_per_iter = 2
burnin_index = int(0.5 * n_iters * samples_per_iter)
samples_adapt = chains[0][burnin_index:]

# import non-adaptive samples
filename = ''  # enter filename here
with open(filename, 'rb') as f:
    pickled_dict = pickle.load(f)

chains = pickled_dict['chains']
title_dict = pickled_dict['title_dict']

samples_per_iter = 2
burnin_index = int(0.5 * n_iters * samples_per_iter)
samples_nonadapt = chains[0][burnin_index:]

n_samples = len(samples_adapt)

fig, axes = plt.subplots(4, 2, figsize=(8.27, 11.69), dpi=300)
# axes = axes.flatten()

timepoint = 249
scatter_size = 1.4
scatter_colour = "#001DF5"
line_colour = "#F5001D"
for var_index in [0, 1, 2, 3]:
    axes[var_index][0].scatter(range(n_samples), [sample[timepoint][var_index] for sample in samples_adapt], s=scatter_size, c=scatter_colour)
    axes[var_index][0].axhline([model.true_hss[timepoint][var_index]], color='red', c=line_colour)

    axes[var_index][1].scatter(range(n_samples), [sample[timepoint][var_index] for sample in samples_nonadapt], s=scatter_size, c=scatter_colour)
    axes[var_index][1].axhline([model.true_hss[timepoint][var_index]], color='red', c=line_colour)

    var_str = str(var_index + 1)
    timepoint_str = str(timepoint + 1)

    for ax in axes[var_index]:
        ax.set_xlabel("Iteration")
        ax.set_ylabel("$x_{" + var_str + ",\," + timepoint_str + "}$", fontsize=14)

        ax.tick_params(which='major', bottom=True, top=True, right=True, direction='in')

    if var_index == 0:
        underscore_i = "_{i}"
        axes[var_index][0].set_title(f"Column 1: Adaptive $\epsilon{underscore_i}$, $L=35$")
        axes[var_index][1].set_title(f"Column 2: $\epsilon \sim U(0.1, 0.4)$, $L=50$")

        axes[var_index][0].set_xlim(0, 1000)
        axes[var_index][1].set_xlim(0, 1000)

        axes[var_index][0].set_ylim(-5.2, 2.5)
        axes[var_index][1].set_ylim(-5.2, 2.5)
    elif var_index == 1:
        axes[var_index][0].set_xlim(0, 1000)
        axes[var_index][1].set_xlim(0, 1000)

        axes[var_index][0].set_ylim(-4.4, 3)
        axes[var_index][1].set_ylim(-4.4, 3)
    elif var_index == 2:
        axes[var_index][0].set_xlim(0, 1000)
        axes[var_index][1].set_xlim(0, 1000)

        axes[var_index][0].set_ylim(-0.9, 4)
        axes[var_index][1].set_ylim(-0.9, 4)
    elif var_index == 3:
        axes[var_index][0].set_xlim(0, 1000)
        axes[var_index][1].set_xlim(0, 1000)

        axes[var_index][0].set_ylim(-4, 2.5)
        axes[var_index][1].set_ylim(-4, 2.5)

title_elems = [key + ": " + str(value) for key, value in title_dict.items()]
title_str = ' - '.join(title_elems)

plt.tight_layout()
plt.savefig(f"traceplot_comparison.png")
plt.show()

