# Embedded-HMM-MCMC-Sampling
An implementation of the embedded HMM sampler for state space models, along with a flexible sampling framework.

To perform sampling, instantiating a Model object and overriding the inherited methods with the relevant transition densities, emission densities and so on.

Then either use a pre-defined Sampler object, or define your own. Each sampler updates using an Updater object.

For the eHMM sampler, the desired scheme and sequence of pool updates may be easily specified by modifying the relevant lists, and dynamic functions for ϵ and L may be passed into the sampler.
