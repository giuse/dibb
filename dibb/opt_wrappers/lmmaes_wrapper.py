# IMPORTANT: remember to register new wrappers in `__init__.py`

import numpy as np
# pip install lmmmaes -- https://pypi.org/project/lmmaes
from lmmaes import Lmmaes

class LmmaesWrapper():
    """Wrapper to match LM-MA-ES to the DiBB interface """
    def __init__(self, init_mu, opts):
        self.init_mu = init_mu
        self.ndims = len(init_mu)
        self.init_sigma = opts.pop('init_sigma', 0.5)
        self.popsize = opts.pop('popsize', None)
        # TODO: grab verbose from DiBB and make compatible with CMA
        self.es = Lmmaes(x=self.init_mu, sigma=self.init_sigma,
                         popsize=self.popsize, verbose=False)

    def stop(self):
        """Check if optimization has terminated"""
        # TODO: implement in lmmaes
        # return self.es.stop()
        return False

    def ask(self):
        """Retrieve a population from the optimizer"""
        return self.es.ask()

    def tell(self, population, fit_values):
        """Send the population fitness to the optimizer, and execute an update step"""
        # TODO: move to ask/tell interface on lmmaes?
        assert np.array_equal(population, self.es.population), \
            'Lost sync between ask and tell!'
        self.es.tell(fit_values)

    # Debug

    def mean(self):
        return self.es.x

    def dump(self, fname):
        with open(fname, 'wb') as fd:
            pickle.dump(self.es, fd)

    def load(self, fname):
        with open(fname, 'rb') as fd:
            self.es = pickle.load(fd)
