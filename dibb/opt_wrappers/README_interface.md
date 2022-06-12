Wrappers need to implement the following "ask & tell" interface:

`__init__` => correctly initialize underlying optimizer (use `dibb.optimizer_options`)
`stop`     => boolean return, whether the optimizer decided to terminate its run
`ask`      => the optimizer returns a population of candidate solutions
`tell`     => send the population + fitnesses to the optimizer for its update step
`popsize`  => (@property) returns the (current) population size (allows for dynamic size)

Optionally, include `dump` and `load` for easier inspection.
Easiest to just duplicate, rename and customize an existing wrapper.

# IMPORTANT:

Remember to add new optimizers to the loader in `dibb/opt_wrappers/__init__.py`!
