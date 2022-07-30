# DiBB -- A framework for Distributed Black Box {optimization, search, algorithms}
# Copyright (c) 2019 2020 2021 2022 Giuseppe Cuccu -- all rights reserved

import inspect
import datetime
from multiprocessing import cpu_count # only for local setup
import numpy as np
import ray

from dibb.remote_data import RemoteData
from dibb.block_worker import BlockWorker


class DiBB:
    """ A framework for Distributed Black Box optimization
    """

    ## Accepted parameters and their default values
    defaults = {
        # initial candidate solution (flattened)
        'init_ind' : None, # defaults to `np.array([0]*ndims`
        # problem dimensions (alternative to init_ind)
        'ndims' : None, # defaults to `len(init_ind)`
        # number of blocks in which to split the computation
        'nblocks' : None, # defaults to `len(block_sizes)`
        # explicit list of sizes for each block
        'block_sizes' : None, # defaults to ~ `ndims/nblocks`
        # fitness function: either callable or string to eval()
        'fit_fn' : None, # if none, it will be expected by `optimize`
        # optimizer for the blocks, as string
        'optimizer' : 'default', # see `dibb/opt_wrappers/__init__.py` for options
        # options to pass raw to the constructor of the optimizer
        'optimizer_options' : {}, # will be passed splatted `**optimizer_options`
        # whether it is a minimization task (otherwise maximization)
        'minim_task' : True, # we expect all `optimizer`s to minimize by default
        # number of fitness evaluators to spawn per block
        'nfitness_evaluators' : 0, # `None` for auto (ncpus), 0 to disable
        # number of fitness evaluations per ind
        'ntrials_per_ind' : 1, # use more for non-deterministic fitnesses
        # callable to aggregate the fitnesses from multiple trials (max, min, etc.)
        'fit_aggregator' : np.mean, # only meaningful for `ntrials_per_ind > 1`
        # Whether the evaluation on the BWs should be parallel (vs sequential)
        'parallel_eval' : True, # only used if `nfitness_evaluators==0`
        # verbosity
        'verbose' : True, # will be a numerical level in the future
        # interval to print run info
        'print_interval' : 1, # in seconds, 0 or None to disable
        # callable to print run info (the DiBB instance is passed as param)
        'print_fn' : lambda dibb: print(dibb.best_fit), # run every `print_interval`
        # hooks allow injecting custom code to be executed by DiBB actors
        'hooks' : None, # {'BlockWorker' : None, 'FitnessEvaluator' : None},
        # dictionary of stopping conditions
        'stopping_conditions' : None, # {'ngens' : 10, 'nevals' : 50, 'trg_fit' : 0.1}
        # fix the random seed for reproducibility
        'rseed' : None, # none for auto
        # Options for launching Ray locally, from DiBB (no distribution)
        # These are only used if no Ray cluster is found
        'ray_config' : {
            'logging_level': 40, # INFO=20, WARNING=30, ERROR=40
            'resources': {
                 # (Reserving one CPU) https://stackoverflow.com/a/55423170
                'DedicatedMachine': cpu_count() - 1
            },
        }
    }


    def __init__(self, init_ind=None, **kwargs):

        # `init_ind` is positional here just for compatibility
        kwargs['init_ind'] = init_ind

        if kwargs.get('rseed', None) is not None:
            raise ParameterError("Custom random seeds for reproducibility purpose are temporarily disabled because of an incompatibility issue. Please do not set them. Thank you.")
                # If you wish to help, try running this test:
                # tests.test_dibb_local.TestDibb.test_dibb_with_rseed

        ## Ensure no spurious parameters
        accepted_params = set(self.defaults.keys())
        received_params = set(kwargs.keys())
        unrecognized_params = received_params.difference(accepted_params)
        if unrecognized_params:
            raise AttributeError(f"Unrecognized params: `{unrecognized_params}`")

        # Aggregate all params in single config
        self.cfg = dict(self.defaults, **kwargs)

        # Verbosity parameters
        self.verbose = self.cfg['verbose']
        self.print_interval = self.cfg['print_interval']
        self.print_fn = self.cfg['print_fn']
        if self.print_fn is not None:
            assert callable(self.print_fn), "Please provide a callable `print_fn`"
            self.print_fn = self.print_fn

        # Set up number of dimensions for initial candidate solution
        assert (self.cfg['init_ind'] is None) ^ (self.cfg['ndims'] is None), \
            'ERROR: pass either `init_ind` or `ndims`'
        if self.cfg['ndims']:
            self.ndims = self.cfg['ndims']
            # NOTE: actual assignment of candidate solution is in reset
            self.cfg['init_ind'] = np.zeros(self.ndims)
        else:
            self.ndims = len(self.cfg['init_ind'])

        # Set up blocks
        assert not (self.cfg['block_sizes'] and self.cfg['nblocks']), \
            'ERROR: pass either `block_sizes` (explicit list) or `nblocks`'

        if self.cfg['block_sizes'] is None and self.cfg['nblocks'] is None:
            self.cfg['nblocks'] = 1 # default to 1 block if neither is passed

        if self.cfg['block_sizes']:
            assert type(self.cfg['block_sizes']) is list and\
                all(type(bs) is int for bs in self.cfg['block_sizes'])
            self.block_sizes = self.cfg['block_sizes']
            self.nblocks = len(self.block_sizes)
        else:
            assert type(self.cfg['nblocks']) is int and self.cfg['nblocks'] > 0
            self.nblocks = self.cfg['nblocks']
            self.block_sizes = self.calc_block_sizes()

        assert sum(self.block_sizes) == self.ndims, \
            f"Sum of block sizes must sum up to number of dimensions " +\
            f"(tried {len(self.blocks)} blocks totaling {sum(self.blocks)} " +\
            f"dimensions with {ndims} parameters)"

        if self.verbose and self.cfg['nblocks'] == 1:
            print(f"\nWARNING: instantiating DiBB with a single block. This is essentially the same as running the optimizer directly, while providing parallel fitness evaluation. You can pass parameter `nblocks` as a number (automatically partitioning your parameters set) or parameter `block_sizes` as an explicit list of sizes for each block.")

        # Fitness evaluators and aggregator
        self.nfes = self.cfg['nfitness_evaluators']
        assert callable(self.cfg['fit_aggregator']) # no need for remote eval here
        self.fit_aggregator = self.cfg['fit_aggregator']
        self.parallel_eval = self.cfg['parallel_eval']
        # If a Ray instance is already initialized, use it (run on cluster)
        # Else, initialize Ray for a local run, with config from `defaults`
        if ray.is_initialized():
            self.available_machines = \
                [node['NodeManagerAddress'] for node in ray.nodes()]
            assert len(self.available_machines) >= self.nblocks
            if self.verbose: print(f"Running on cluster with machines:", self.available_machines)
        else:
            self.available_machines = None
            if self.verbose: print(f"\nWARNING: launching DiBB locally (no cluster). To distribute the BlockWorkers to multiple machines, see file `ray_cluster_config.yaml`.")
            ray.init(**self.cfg['ray_config'])

        # This `dibb_id` allows for launching multipls DiBBs on the same machine
        posix_time = datetime.datetime.now().timestamp()
        self.dibb_id = str(posix_time).strip('.') # pseudo-random

        # These will be initialized / assigned by `reset`
        self.fit_fn = None
        self.hooks = None
        self.stopping_conditions = None
        # NOTE: the two communication structures below are currently kept
        # separated just for code legibility and to split the work load
        self.workers_comm = RemoteData.remote() # General communication
        self.remote_mean  = RemoteData.remote() # Shared state of the search

        # Now ready to spawn BWs
        self.spawn_block_workers()

        # Method `reset()` takes care of the rest
        reset_param_names = inspect.signature(self.reset).parameters
        reset_params = {k:v for (k,v) in self.cfg.items() if k in reset_param_names}
        self.reset(**reset_params)


    # Resetting DiBB is faster (saves on distribution overhead) for minor changes
    def reset(self,
            init_ind=None,
            fit_fn=None,
            optimizer=None,
            optimizer_options=None,
            minim_task=None,
            ntrials_per_ind=None,
            hooks=False,
            stopping_conditions=False,
            rseed=False,
        ):

        if init_ind is not None:
            # NOTE: if the number of blocks changes (e.g. because of the
            # number of dimensions changed), you should reinstantiate DiBB
            assert len(init_ind) == self.ndims
            self.init_ind = init_ind
            # `remote_mean` holds the shared, remote mean of the search
            # distribution, and is now set to the new init_ind
            split_init_ind = self.split_ind_by_block_sizes(init_ind)
            ray.get(self.remote_mean.reset.remote(split_init_ind))

        if fit_fn is not None:
            # If string, it will be `eval`ed in the BW or FE for local instantiation
            assert callable(fit_fn) or type(fit_fn) is str
            self.fit_fn = fit_fn

        if optimizer is not None:
            self.optimizer = optimizer

        if optimizer_options is not None:
            assert type(optimizer_options) is dict
            self.optimizer_options = optimizer_options

        if minim_task is not None:
            assert type(minim_task) is bool
            self.minim_task = minim_task

        if ntrials_per_ind is not None:
            assert type(ntrials_per_ind) is int and ntrials_per_ind > 0
            self.ntrials_per_ind = ntrials_per_ind

        # Handle hooks
        allowed_hooks = ['BlockWorker', 'FitnessEvaluator']
        # the default value `False` is used to detect a request for "no change"
        if hooks is not False:
            # `None` is used to request a reset: skip validation
            if hooks is not None:
                # we have custom hooks: validate the format
                assert type(hooks) is dict, \
                    f"Should be a dict, got `{hooks}`"
                for k, v in hooks.items():
                    assert k in allowed_hooks, \
                        f"Unrecognized hook handle `{k}` (key `{v}`)"
                    assert type(v) is list or callable(v), \
                        f"Unrecognized hook type `{v}`"
                    if type(v) is list:
                        for f in v:
                            assert callable(f), f"Hooks should be callable"
            # Ok ready to assign
            self.hooks = hooks

        # Handle stopping conditions
        allowed_stop_cond = ['ngens', 'nevals', 'trg_fit']
        # the default value `False` is used to detect a request for "no change"
        if stopping_conditions is not False:
            # `None` is used to request a reset: skip validation
            if stopping_conditions is not None:
                # we have custom stopping_conditions: validate the format
                assert type(stopping_conditions) is dict, \
                    f"Should be a dict `{stopping_conditions}"
                for k, v in stopping_conditions.items():
                    assert k in allowed_stop_cond, \
                        f"Unrecognized stopping condition `{k}`"
                    assert type(v) in [int, float], \
                        f"Value for `{k}` should be an integer or (convertible) float (received {v})"
                if type(v) is float: stopping_conditions[k] = int(v)
            # Ok ready to assign
            self.stopping_conditions = stopping_conditions

        # the default value `False` is used to detect a request for "no change"
        if rseed is not False: # None resets to unfixed seed
            if not (type(rseed) is int or rseed is None):
                raise AttributeError(f"Wrong type for parameter rseed: `{rseed}`")
            self.rseed = rseed

        # Reset communication dict and remote mean
        ray.get(self.workers_comm.reset.remote({
            'running' : [False]*self.nblocks, # workers=>dibb
            'terminate' : False, # dibb=>workers
            'ngens' : [0]*self.nblocks,
            'nevals' : [0]*self.nblocks,
            'best' : {
                'fit' : float('inf') if self.minim_task else float('-inf'),
                'ind' : None, },
        }))

        # Reset BWs (async call + sync wait)
        waiting_list = []
        for worker in self.block_workers:
            waiting_list.append(worker.reset.remote(
                optimizer=self.optimizer,
                optimizer_options=self.optimizer_options,
                minim_task=self.minim_task,
                fit_fn=self.fit_fn,
                ntrials_per_ind=self.ntrials_per_ind,
                fit_aggregator=self.fit_aggregator,
                rseed=self.rseed,
                hooks=self.hooks,
                stopping_conditions=self.stopping_conditions,
                verbose=self.verbose
            ))
        ray.wait(waiting_list, num_returns=len(waiting_list))


    # Spawns the BW actors, one for each block, on dedicated machines if available
    def spawn_block_workers(self):
        self.block_workers = []

        for block_id in range(self.nblocks):

            # Assign an available machine
            if self.available_machines:
                ip = self.available_machines[block_id]
            else:
                ip = None

            # Construct the parameter options to pass to Ray and BW respectively
            ray_opts = {
                'name' : f"dibb_{self.dibb_id}_block_worker_{block_id:03}",
                'resources' : {'DedicatedMachine' : 1},
            }
            bw_opts = {
                # ID of DiBB is used for the names of the actors (including FEs)
                'dibb_id' : self.dibb_id,
                # ID of the worker (and index of corresponding block)
                'block_id' : block_id,
                # Shared mean for all blocks (TODO refactor name)
                'remote_mean' : self.remote_mean,
                # BlockWorkers communication dict
                'comm' : self.workers_comm,
                # Machine on which to run the BW (and its FEs)
                'machine' : ip,
                # Whether the evaluation on the BWs should be parallel (vs sequential)
                'parallel_eval' : self.parallel_eval,
                # number FEs per block ("auto" for auto)
                'nfitness_evaluators' : self.nfes,
            }

            # Instantiate the block workers
            self.block_workers.append(
                BlockWorker.options(**ray_opts).remote(**bw_opts))


    def optimize(self, fit_fn=None, ngens=0, nevals=0, print_interval=None):
        # NOTE: it returns `self` at the end to enable `dibb=DiBB().run()`
        # `ngens` and `nevals` are used to increment the BW run budget (e.g. resume run)

        # Update fit_fn if passed
        assert fit_fn or self.fit_fn, "ERROR: pass your optimization function " +\
            f"as `fit_fn` parameter either to the DiBB constructor (preferred) " +\
            f"or directly here to `optimize`/`run` (allowed only for compatibility)."
        if fit_fn is not None:
            assert callable(fit_fn) or type(fit_fn) is str
            self.fit_fn = fit_fn

        # Update print_interval if passed
        if print_interval is not None:
            assert type(print_interval) is int and print_interval > 0
            self.print_interval = print_interval

        # Launch computation on all block workers
        self.workers_comm.set.remote('terminate', val=False)
        self.run_start_time = int(datetime.datetime.now().timestamp())
        # `ngens` and `nevals` are used to increment the BW run budget (e.g. resume run)
        rets = [bw.run.remote(ngens, nevals, fit_fn) for bw in self.block_workers]

        timeout = self.print_interval or 30 # how often to check on the BWs
        while ray.wait(rets, num_returns=len(rets), timeout=timeout)[1]:
            if self.print_fn: self.print_fn(self)
            # If all workers have paused, break out of the wait
            if not self.any_worker_running():
                if self.verbose: print("All workers are paused")
                break

        # Kindly ask workers to terminate (async, wait for clean break)
        self.workers_comm.set.remote('terminate', val=True)
        if self.verbose: print("\nTermination command sent to workers")
        ray.wait(rets, num_returns=len(rets))

        # End-of-run printing
        if self.verbose: print("All blocks terminated, end of run\n")

        return self # this allows `dibb = DiBB(**config).run()


    ## Block tools

    def calc_block_sizes(self):
        bsize = self.ndims // self.nblocks
        assert bsize >= 2, \
            f"Currently minimum block size supported is 2 " +\
            f"(tried {self.nblocks} blocks with {self.ndims} parameters)"
        rest = self.ndims % self.nblocks
        block_sizes = [bsize]*self.nblocks
        # distribute the rest of the division across the blocks
        for block_idx in range(self.nblocks):
            if rest > 0:
                block_sizes[block_idx] += 1
                rest -= 1
            else:
                break
        return block_sizes

    def split_ind_by_block_sizes(self, ind):
        split = []
        start = 0
        for idx, b_size in enumerate(self.block_sizes):
            end = start + b_size
            # TODO: verify if copy is necessary! if so, add why to this comment
            split.append(ind[start:end].copy())
            start = end
        return split

    ## Convenience and shorthand methods

    def run(self, ngens, **kwargs): # just for compatibility
        return self.optimize(ngens=ngens, **kwargs)

    def any_worker_running(self):
        return ray.get(self.workers_comm.get.remote('running', apply_fn=np.any))

    @property
    def mean(self): # TODO: refactor better names all around "remote_mean"
        return ray.get(self.remote_mean.get.remote(apply_fn=np.concatenate))

    @property
    def best_ind(self):
        return ray.get(self.workers_comm.get.remote('best', 'ind'))

    @property
    def best_fit(self):
        return ray.get(self.workers_comm.get.remote('best', 'fit'))

    @property
    def comm_dict(self):
        return ray.get(self.workers_comm.get.remote())

    @property
    def result(self):
        return {
            'mean'     : self.mean,
            'best_ind' : self.best_ind,
            'best_fit' : self.best_fit,
        }


    ## DEBUG
    # DiBB is _not_ a Ray actor, but we still define these for convenience/consistency
    def _get_opt_wrappers(self):
        # NOTE: this will give problems if opt is not serializable/picklable
        return ray.get([b._get_opt.remote() for b in self.block_workers])
    def _get_opts(self): return [wrapper.es for wrapper in self._get_opt_wrappers()]
    def _get_sigmas(self):
        # can be used to check for convergence if opt has a `sigma`
        return np.array([opt.sigma for opt in self._get_opts()])
    def _get_dibb_id(self): return self.dibb_id
