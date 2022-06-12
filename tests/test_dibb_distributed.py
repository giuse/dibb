import unittest
import warnings, time
import numpy as np
import ray
from dibb import DiBB, fmin

# Monotonic with optimum at np.zeros() (careful with init_ind)
class Sphere:
    def __init__(self, ndims):
        self.ndims = ndims
        # to test passing fit_fn in string format
        self.evaluable_str = "lambda x: np.dot(x, x)"
        self.init_ind = np.random.randn(self.ndims)
        self.init_ind += 10 # shift search start away from 0

    # use the `evaluable_str` to define `__call__` for consistency
    def __call__(self, x):
        return eval(self.evaluable_str)(x)


class TestDibb(unittest.TestCase):

    ## Setup

    @classmethod
    def setUpClass(cls):

        # => ray up!

        np.random.seed(1) # for consistency
        cls.ndims = 10
        cls.fit_fn = Sphere(cls.ndims)
        cls.ngens = 150
        cls.nblocks = 3
        cls.expected_block_size = [4,3,3]

    # TODO: check if necessary/hurtful
    @classmethod
    def tearDownClass(cls):

        # => ray down!

        ray.shutdown()

    # Ignore Ray's logging `ResourceWarning`s (as decorator for tests)
    def ignore_resource_warnings(test_func):
        def do_test(self, *args, **kwargs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)
                test_func(self, *args, **kwargs)
        return do_test

    def convergence_check(self, dibb):
        fit_of_mean = self.fit_fn(dibb.mean)
        self.assertLess(fit_of_mean, 1e-4, "Should have gotten close to zero")
        sigmas = dibb._get_sigmas()
        self.assertTrue(np.all(sigmas < 1e-4), "Should have converged")

    ## Tests


    # => test with multiple machines


    @ignore_resource_warnings
    def test_dibb_with_opts(self):

        ## Launch Ray before initializing DiBB
        ray.shutdown()
        time.sleep(1) # wait from instances from previous tests to die
        self.assertFalse(ray.is_initialized())
        ray.init(**DiBB.defaults['ray_config']) # test pre-initialized ray
        self.assertTrue(ray.is_initialized())

        # Options galore
        config = {
            # initial candidate solution (flattened)
            'init_ind' : self.fit_fn.init_ind,
            # problem dimensions (alternative to init_ind)
            # 'ndims' : None, # defaults to `len(init_ind)`
            # number of blocks in which to split the computation
            'nblocks' : self.nblocks,
            # explicit list of sizes for each block
            # 'block_sizes' : None, # defaults to ~ `ndims/nblocks`
            # fitness function: either callable or string to eval()
            'fit_fn' : self.fit_fn.evaluable_str, # => string!
            # optimizer for the blocks, as string
            'optimizer' : 'cma', # 'lmmaes',
            # options to pass raw to the constructor of the optimizer
            'optimizer_options' : {'init_sigma':0.3},
            # machines available for distributing the computation/blocks
            # 'available_machines' : None, # If `None`: local run
            # whether it is a minimization task (otherwise maximization)
            # 'minim_task' : True, # we expect all `optimizer`s to minimize by default
            # number of fitness evaluators to spawn per block
            'nfitness_evaluators' : 5, # `None` for auto (ncpus), 0 to disable
            # number of fitness evaluations per ind
            'ntrials_per_ind' : 3, # use more for non-deterministic fitnesses
            # callable to aggregate the fitnesses from multiple trials (max, min, etc.)
            'fit_aggregator' : np.mean, # only meaningful for `ntrials_per_ind > 1`
            # verbosity
            'verbose' : True, # will be a numerical level in the future
            # interval to print run info
            'print_interval' : 0.1,
            # callable to print run info (the DiBB instance is passed as param)
            'print_fn' : lambda dibb: print(dibb.mean.sum(), flush=True),
            # hooks allow injecting custom code to be executed by DiBB actors
            # 'hooks' : {'BlockWorker' : None, 'FitnessEvaluator' : None},
            # dictionary of stopping conditions
            'stopping_conditions' : {'nevals':1e10}, # ngens should hit first
            # fix the random seed for reproducibility
            'rseed' : 1,
            # Options for launching Ray from DiBB (only on local machine)
            'ray_config' : {
                'logging_level': 20, # INFO=20, WARNING=30, ERROR=40
                'resources': {
                     # (Reserving one CPU) https://stackoverflow.com/a/55423170
                    'DedicatedMachine': 6
                },
            }
        }

        dibb = DiBB(**config)
        # self.assertEqual(dibb.block_sizes, self.expected_block_size)
        dibb.run(self.ngens) # = dibb.optimize(ngens=self.ngens)
        self.convergence_check(dibb)

if __name__ == '__main__':
    unittest.main()
