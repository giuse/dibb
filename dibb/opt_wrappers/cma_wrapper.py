# IMPORTANT: remember to register new wrappers in `__init__.py`

import numpy as np
# pip install cma -- https://pypi.org/project/cma
from cma import CMAEvolutionStrategy


# Silence CMA warnings
from sys import warnoptions
if not warnoptions:
    from warnings import simplefilter
    simplefilter("ignore")


class CmaWrapper():
    """Wrapper to match CMA-ES to the DiBB interface """
    def __init__(self, init_mu, opts):
        self.init_mu = init_mu
        self.ndims = len(init_mu)
        self.init_sigma = opts.pop('init_sigma', 0.5)
        self.log_cma = opts.pop('log_workers', False)
        self.rseed = opts.pop('rseed', None)
        if self.rseed: opts['seed'] = rseed
        if not opts.get('verbose', None):
            opts['verbose'] = -9
        self.es = CMAEvolutionStrategy(self.init_mu, self.init_sigma, opts)

    def stop(self):
        """Check if optimization has terminated"""
        return self.es.stop() or \
            (self.es.sm.condition_number > 1e11 and \
             {'condition_number' : self.es.sm.condition_number})

    def ask(self):
        """Retrieve a population from the optimizer"""
        return self.es.ask()

    def tell(self, population, fit_values):
        """Send the population fitness to the optimizer, and execute an update step"""
        # TODO: check about using `somenan` from CMA
        self.es.tell(population, fit_values)
        # In case we want to analyze the CMA trajectory later
        if self.log_cma: self.es.logger.add()

    @property
    def popsize(self):
        return self.es.popsize


    # Debug

    def mean(self):
        return self.es.mean

    def dump(self, fname):
        with open(fname, 'wb') as fd:
            pickle.dump(self.es, fd)

    def load(self, fname):
        with open(fname, 'rb') as fd:
            self.es = pickle.load(fd)
