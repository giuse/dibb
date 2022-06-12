import os, sys, platform, time, inspect
import numpy as np
import ray
from dibb.fitness_evaluator import FitnessEvaluator
from dibb import opt_wrappers


@ray.remote(resources={'DedicatedMachine': 1})
class BlockWorker:
    """ Worker to run the optimization on a single block
    """

    def __init__(self,
                 # DiBB force-sets all of these parameters.
                 # See `dibb.defaults` for default values and documentation.
                 block_id,
                 remote_mean,
                 comm,
                 machine,
                 nfitness_evaluators,
                 parallel_eval,
                 dibb_id
        ):

        # Basic assignments
        self.block_id = block_id
        self.remote_mean = remote_mean
        self.comm = comm
        self.machine = machine
        # TODO: better values for auto assign (default) and disable
        if nfitness_evaluators is None: # `0` to disable them
            nfitness_evaluators = len(os.sched_getaffinity(0))
        self.nfitness_evaluators = nfitness_evaluators
        self.parallel_eval = parallel_eval
        self.dibb_id = dibb_id

        # Make sure that fitness evaluators are on the same machine as the block worker
        worker_ip = self.machine if self.machine else \
            ray.util.get_node_ip_address()
        resources = {f'node:{worker_ip}': 1}

        # Spawn fitness evaluators (if nfitness_evaluators > 0)
        self.f_evaluators = []
        for i in range(self.nfitness_evaluators):
            # Instantiate FitnessEvaluators on the same node and give them a meaningful name
            name = f'dibb_{self.dibb_id}_block_{self.block_id:03}_fitness_evaluator_{i:03}'
            self.f_evaluators.append(
                FitnessEvaluator.options(
                    name=name,
                    resources=resources,
                    num_cpus=0.1
                ).remote(
                    block_id=self.block_id,
                    dibb_id=self.dibb_id
                ))

        # No call to `self.reset()` here because DiBB takes care of it when needed


    # Allows resetting the BW without reinstantiating (saving Ray overhead)
    def reset(self,
             # DiBB force-sets all of these parameters.
             # See `dibb.defaults` for default values and documentation.
              optimizer,
              optimizer_options,
              minim_task,
              fit_fn,
              ntrials_per_ind,
              fit_aggregator,
              rseed,
              hooks,
              stopping_conditions,
              verbose
        ):
        # Basic assignments
        self.minim_task = minim_task
        if self.minim_task:
            self.comp_fn = np.less
            self.best_fn = min
            self.best_fit_so_far = float('inf')
        else:
            self.comp_fn = np.greater
            self.best_fn = max
            self.best_fit_so_far = float('-inf')

        if self.nfitness_evaluators == 0: # evaluate fit on BWs
            if type(fit_fn) is str:
                assert not self.parallel_eval, "You cannot run complex fitnesses (such as with initialization via string) in parallel on the BW, because all fitness functions would access the same (one) instantiated resource. If this is just a quick test, simply set the DiBB option `parallel_eval=False` to enforce sequential evaluation on the BW. More likely, you'll want to switch to using the FEs instead: each FE will launch your (string-fit_fn) setup independently, hence allocating/setting up the resources for your fitness evaluation. The BW will then request fitness evaluations to a pool of FEs, reusing the resources. If this is not a complex fitness, simply pass it as a pre-initialized callable."
                fit_fn = eval(fit_fn)
            if self.parallel_eval:
                # parallelize fit_fn using Ray
                this_machines_ip = ray.util.get_node_ip_address()
                res = {f'node:{this_machines_ip}': 1} # ensure local spawn
                fit_fn = ray.remote(fit_fn).options(resources=res).remote

        self.fit_fn = fit_fn
        self.ntrials_per_ind = ntrials_per_ind
        self.fit_aggregator = fit_aggregator
        self.rseed = rseed
        self.hooks = hooks.get('BlockWorker', None) if hooks else hooks
        self.verbose = verbose
        # Track execution for stopping criteria
        self.stopping_conditions = stopping_conditions or {}
        self.ngen = 0
        self.nevals = 0
        # Instantiate optimizer
        initial_mean = ray.get(self.remote_mean.get.remote(self.block_id))
        opt_class = opt_wrappers.load_opt(optimizer)
        self.optimizer = opt_class(initial_mean, optimizer_options)

        # Configure fitness evaluators
        # TODO: check after fixing default/auto values for it
        if self.nfitness_evaluators:
            rets = []
            for evaluator in self.f_evaluators:
                rets.append(evaluator.reset.remote(
                    fit_fn = self.fit_fn,
                    rseed = self.rseed,
                    hooks = hooks.get('FitnessEvaluator', None)) if hooks else hooks,
                )
            ray.wait(rets, num_returns=len(rets))

        # Set the BW as running we are ready to go
        ray.get(self.comm.set.remote('running', self.block_id, val=True))


    def eval_pop_on_fes(self, population):
        # Distribute split_genotype ind to FEs
        ray.get([fe.update_ind.remote(self.split_genotype) for fe in self.f_evaluators])
        # Prepare a list of "busy" evaluators (better code readability)
        busy_fes = {}
        for idx, fe in enumerate(self.f_evaluators):
            dummy = fe.dummy.remote() # does nothing
            busy_fes[dummy] = fe

        # Evaluate all individuals
        fitnesses = [None]*len(population)
        for ind_idx, ind in enumerate(population):

            # Each individual may require multiple evaluations
            fitnesses[ind_idx] = [None]*self.ntrials_per_ind
            for ntrial in range(self.ntrials_per_ind):
                # Obtain an available fitness estimator with wait()
                (done_fit,*_), _ = ray.wait(list(busy_fes.keys()))
                fe = busy_fes.pop(done_fit) # grab the actual FE
                # Queue to run one fitness evaluation
                fit = fe.evaluate.remote(ind)

                # Mark the FE as busy, until fit is done_fit from wait()
                busy_fes[fit] = fe
                # Save future result (obj_id: pull later with ray.get())
                fitnesses[ind_idx][ntrial] = fit

        # Wait for all computation to finish
        # ray.wait(list(busy_fes.keys()), num_returns=len(busy_fes))
        # NO NEED: we call ray.get() below
        # return ray.get(fitnesses)
        return [ray.get(x) for x in fitnesses]


    def eval_pop_on_bws(self, population):
        # TODO: make sure to keep this synced to FEs' code
        full_inds = []

        # Construct full inds
        split_genotype = ray.get(self.split_genotype)
        for ind in population:
            split_genotype[self.block_id] = ind
            full_ind = np.concatenate(split_genotype)
            full_inds.append(full_ind)

        # Compute fitnesses for full inds
        fitnesses = []
        for full_ind in full_inds:
            this_ind_fit = []
            for neval in range(self.ntrials_per_ind):
                if self.rseed is not None and \
                    'rseed' in inspect.signature(self.fit_fn).parameters:
                    fit = self.fit_fn(full_ind, rseed=self.rseed)
                else:
                    fit = self.fit_fn(full_ind)
                this_ind_fit.append(fit)
            fitnesses.append(this_ind_fit)

        return fitnesses


    def eval_fit_on_pop(self, population):
        """ Return a list of fitnesses for all individuals in the
            population, evaluated in parallel (if using FEs at least).
        """
        # TODO: check with new auto/default settings after refactoring
        if self.nfitness_evaluators:
            fitnesses = self.eval_pop_on_fes(population)
        else: # not using FitnessEvaluators: run on BW
            fitnesses = self.eval_pop_on_bws(population)
        if self.parallel_eval: fitnesses = [ray.get(fit) for fit in fitnesses]
        # Aggregate fit for each ind (because always `ntrials_per_ind>0`)
        fit_vals = [self.fit_aggregator(fit_lst) for fit_lst in fitnesses]
        # Keep track of the number of fitness evaluations
        # TODO: these should actually be reserved at `reasons_for_termination`
        self.nevals += len(fit_vals) * self.ntrials_per_ind
        self.comm.set.remote('nevals', self.block_id, val=self.nevals)

        return fit_vals


    # `ngens` and `nevals` are used to increment the BW run budget (e.g. resume run)
    def run(self, ngens, nevals, fit_fn):
        """ Main BW run
        """
        if ngens: self.stopping_conditions['ngens'] = \
            self.stopping_conditions.get('ngens', 0) + ngens
        if nevals: self.stopping_conditions['nevals'] = \
            self.stopping_conditions.get('nevals', 0) + nevals
        if fit_fn:
            if type(fit_fn) is str:
                self.fit_fn = eval(fit_fn)
            else:
                self.fit_fn = fit_fn

        # Execution loops until termination criteria are satisfied or
        # until the all BWs underlying ESs simultaneously convergence/stop
        while True:

            term_reasons = self.reasons_for_termination()
            if any(term_reasons.values()):
                # Add possible optimizer reasons
                term_reasons['opt_stop'] = self.convergence_check()
                break

            # Run all block worker hooks
            if self.hooks:
                for hook in list(self.hooks):
                    hook(self)

            # Obtain and evaluate population
            population = self.optimizer.ask()
            self.split_genotype = self.remote_mean.get.remote() # same for all inds
            fitnesses = self.eval_fit_on_pop(population)

            # Save best
            self.best_fit_so_far = ray.get(self.comm.get.remote('best', 'fit'))
            this_gens_best_fit = self.best_fn(fitnesses)
            if self.comp_fn(this_gens_best_fit, self.best_fit_so_far):
                # NOTE: not a copy, but we overwrite it next gen anyway
                local_best = population[fitnesses.index(this_gens_best_fit)]
                full_ind = ray.get(self.split_genotype)
                full_ind[self.block_id] = local_best
                best_ind = np.concatenate(full_ind)
                # Set new best on communications dict
                self.comm.set.remote('best', 'fit', val=this_gens_best_fit)
                self.comm.set.remote('best', 'ind', val=best_ind)

            # Run the optimizer's update step (it's local, no `wait()` needed)
            # NEGATING FITNESSES: always expect ES to min., we handle max. ourselves
            if not self.minim_task: fitnesses = [-f for f in fitnesses]
            self.optimizer.tell(population, fitnesses)
            # Increase number of (executed) generations
            self.ngen += 1
            self.comm.set.remote('ngens', self.block_id, val=self.ngen)
            # Propagate the update to the remote_mean
            self.remote_mean.set.remote(self.block_id, val=self.optimizer.mean())

        # End of run
        self.comm.set.remote('running', self.block_id, val=False)
        if verbose: print(f"Block {self.block_id} run terminated with " +\
                          f"reasons:\n\t{reasons_for_termination}", flush=True)


    def reasons_for_termination(self):
        # Looks weird but ends up being more readable than a deep `if` chain
        return {
            'remote_request' :
                ray.get(self.comm.get.remote('terminate')),
            'target_fit_reached' :
                self.stopping_conditions.get('trg_fit', False) and \
                self.comp_fn(self.best_fit_so_far, \
                             self.stopping_conditions['trg_fit']) and \
                self.best_fit_so_far,
            'out_of_gens' :
                self.stopping_conditions.get('ngens', False) and \
                self.ngen >= self.stopping_conditions['ngens'],
            'out_of_fevals' : # GLOBALLY among all BWs
                self.stopping_conditions.get('nevals', False) and \
                (   ray.get(self.comm.get.remote('nevals', apply_fn=sum)) +\
                    self.optimizer.popsize * self.ntrials_per_ind
                ) >= self.stopping_conditions['nevals'],
        }


    def convergence_check(self):
        # If this BW's optimizer has converged, just try to sleep a bit and
        # let the other block workers catch up: this updates the remote mean,
        # which  should get out of convergence because of the moving target
        done = self.optimizer.stop() # the ES typically provides a reason here
        if done:
            # Pause block: the *moving target* problem should allow resuming later
            if self.verbose:
                print(f"Pausing block {self.block_id}, gen: {self.ngen}, " +\
                      f"reason: {done}", flush=True)

            self.comm.set.remote('running', self.block_id, val=False)
            sleep_time = 5  # TODO (self)configurable sleep time
            check_delay = 0.1  # Prevents BlockWorker to still sleep after `reset`
            t0 = t1 = time.time()
            while t0 - t1 < sleep_time:
                if ray.get(self.comm.get.remote('terminate')):
                    # run is terminating, no reason for waiting any longer
                    return done
                time.sleep(check_delay)
                t1 = time.time()
            self.comm.set.remote('running', self.block_id, val=True)

            if self.verbose:
                print(f"Resuming block {self.block_id}")
        return done


    ## DEBUG
    # The BW is a Ray actor, we need `ray.get()` to access its instance variables
    def _get_opt(self): return self.optimizer
    def _get_fitness_evaluators(self): return self.f_evaluators
    def _get_dibb_id(self): return self.dibb_id
