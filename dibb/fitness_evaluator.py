import platform, inspect
import numpy as np
import ray

@ray.remote
class FitnessEvaluator:
    """ Runs one fitness evaluation on one individual.

        Note: an "individual" is in principle only a partial genotype,
        corresponding to the block of a BlockWorker. The full genotype
        to pass to the fitness function is constructed based on the
        `remote_mean`, the reference full genotype maintained across
        blocks, which is also the distribution's mean.
    """

    def __init__(self, block_id, dibb_id):
        self.block_id = block_id
        self.dibb_id = dibb_id

    def reset(self, fit_fn, rseed, hooks):
        self.hooks = hooks
        # Set fitness function depending on type passed
        if fit_fn is not None:
            if callable(fit_fn):
                self.fit_fn = fit_fn
            elif type(fit_fn) is str:
                self.fit_fn = eval(fit_fn)
            else:
                raise ValueError(f"Unknown fit_fn `{fit_fn}`")

        # TODO: double check with fresh mind but we should always set rseed
        # even if just as None
        # if rseed is not None:
        self.rseed = rseed

        # `self.split_genotype` is set by `update_ind()`

    def update_ind(self, split_genotype):
        # IMPORTANT: called by block_worker to sync same genotype to all FEs
        # TODO: is `np.copy()` needed? (LR: "pinned_in_memory")?
        self.split_genotype = split_genotype

    def dummy(self): pass # simplifies Ray FE pool management in BlockWorker

    def evaluate(self, ind):
        # Run all fitness evaluator hooks
        if self.hooks:
            for hook in list(self.hooks):
                hook(self)
        # TODO: is `np.copy()` needed? (LR: "pinned_in_memory")?
        self.split_genotype[self.block_id] = np.copy(ind)
        full_ind = np.concatenate(self.split_genotype)
        # allow for `fit_fn` that can/cannot handle parameter `rseed`
        if self.rseed is not None and \
            'rseed' in inspect.signature(self.fit_fn).parameters:
            fit = self.fit_fn(full_ind, rseed=self.rseed)
        else:
            fit = self.fit_fn(full_ind)

        return fit


    ## DEBUG
    # The FE is a Ray actor, we need `ray.get()` to access its instance variables
    def _get_fit_fn(self): return self.fit_fn if hasattr(self, 'fit_fn') else None
    def _get_block_id(self): return self.block_id
    def _get_dibb_id(self): return self.dibb_id
