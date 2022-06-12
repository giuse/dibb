# IMPORTANT: remember to register new wrappers in `__init__.py`

# Parameter guessing
# Simply generate random guesses from a uniform distribution.
# At the end, return the best found so far
#
# NOTE: this algorithm is deceptively effective, and constitutes a very
#       valuable baseline especially when evaluating randomized algorithms,
#       or to estimate the complexity of a task.

import numpy as np

class ParameterGuessing():
    """Parameter Guessing implementation (no need for wrapper)"""
    def __init__(self, init_mu, opts):
        self.best = init_mu # just not to be "None"
        self.best_fit = float('inf') # minimization as usual
        self.rseed = opts.pop('rseed', None)
        self.rng = np.random.default_rng(seed=self.rseed)
        self.ndims = len(init_mu)
        default_popsize = int(4 + 3 * np.log(self.ndims)) # CMA lower bound
        self.popsize = opts.pop('popsize', default_popsize)
        self.sampling_range = opts.pop('range', (-1, 1))
        self.es = self # unnecessary to distinguish wrapper from implementation
        assert not opts, f"Unrecognized option: {opts}"

    def stop(self):
        """Check if optimization has terminated"""
        # PG never needs stopping
        return False

    def ask(self):
        """Retrieve a population from the optimizer"""
        return self.rng.uniform(*self.sampling_range, (self.popsize, self.ndims))

    def tell(self, population, fit_values):
        """Send the population fitness to the optimizer, and execute an update step"""
        # PG simply keeps the best so far
        best_idx = np.argmin(fit_values) # minimization!
        new_best_fit = fit_values[best_idx]
        if new_best_fit < self.best_fit:
            self.best_fit = new_best_fit
            self.best = population[best_idx]

    # Not needed for this implementation
    # @property
    # def popsize(self):
    #     return self.es.popsize


    # Debug

    def mean(self):
        return self.best

    def dump(self, fname):
        with open(fname, 'wb') as fd:
            pickle.dump(self, fd)

    def load(self, fname):
        with open(fname, 'rb') as fd:
            self = pickle.load(fd)
