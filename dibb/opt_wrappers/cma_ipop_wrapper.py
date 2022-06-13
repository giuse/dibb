# IMPORTANT: remember to register new wrappers in `__init__.py`

# pip install cma -- https://pypi.org/project/cma
from dibb.opt_wrappers.cma_wrapper import CmaWrapper

# CMA-ES with IPOP
class CmaIpopWrapper(CmaWrapper):
    """Wrapper to match CMA-ES to the DiBB interface """

    DEFAULT_MAX_RESTARTS = 9 # Standard in other implementations, but be careful

    def __init__(self, init_mu, opts):
        self.max_restarts = opts.pop('max_restarts', self.DEFAULT_MAX_RESTARTS)
        super().__init__(init_mu, opts)
        self.nrestarts = 0
        self.opts = opts
        self.opts['popsize'] = self.es.popsize

    def stop(self):
        """Check if optimization has terminated"""
        converged = self.es.stop() or \
            (self.es.sm.condition_number > 1e11 and \
             {'condition_number' : self.es.sm.condition_number})
        if converged:
            if self.nrestarts < self.max_restarts:
                # If restarts are available, double the population size and restart
                self.opts['popsize'] *= 2
                self.nrestarts += 1
                print('\n\tconverged! new popsize:', self.opts['popsize'],\
                      '/ restart', self.nrestarts)
                # Reset the original parameters, keep only the new mean
                super().__init__(self.es.mean, self.opts)
                # Recur to ensure the checks are fine
                return self.stop()
            else:
                # Following previous implementations, terminate (with cause)
                converged['hit_max_restarts'] = self.nrestarts
        return converged
