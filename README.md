# DiBB

_Distributing Black-Box optimization_

## TL;DR

DiBB generates and runs a Partially-Separable, parallelized and distributed version of your favorite Black Box optimization algorithm -- including Evolutionary Algorithms such as Evolution Strategies for continuous optimization.

```bash
# also installs CMA-ES, a solid choice for the underlying BBO
pip install dibb[cma]`
```

```python
# Python
from dibb import DiBB
# If a Ray cluster with 3 machines is already running, the computation will be
# distributed as 2 block workers (5 dims each) and 1 head node
# If no Ray cluster is running, the three processes will spawn locally
dibb = DiBB(ndims=10, nblocks=2, fit_fn=lambda x: sum(x**2), verbose=True)
dibb.run(ngens=10)
```

Paper + bibtex are at the bottom of this file, or check out [Giuse's page](https://exascale.info/giuse).

**Powerusers:** don't miss out on the *Advanced Usage* section below.


## Installation

You can install DiBB from Pypi using `pip install dibb[cma]` or `dibb[lmmaes]`, which installs both DiBB and a solid underlying optimizer ([CMA-ES](https://github.com/CMA-ES/pycma) or [LM-MA-ES](https://github.com/giuse/lmmaes) respectively). Running `pip install dibb[all]` installs all currently available optimizers.  

If you just `pip install dibb`, it will install only DiBB (and Parameter Guessing); you can then install separately an optimizer of your choice, just make sure that
[there is already a wrapper available for it](dibb/opt_wrappers/__init__.py) -- if not, [writing your own](dibb/opt_wrappers/README_interface.md) is very easy.

To contribute, you can also clone the repo from GitHub
`git clone https://github.com/giuse/dibb`
then install it locally with
`pip install -e <path_to>/dibb`.

## Local usage
DiBB is compatible with several traditional workflows from the larger family of Black Box Optimization methods. Here's an example:

```python
# Let's define a simple function to optimize:
import numpy as np
def sphere(coords): np.dot(coords, coords)

# You can use the classic `fmin`, if you need
from dibb import fmin
fmin(sphere, np.random.rand(10))
# ...but that is not particularly interesting

# Let's fire up a couple of *blocks* instead:
from dibb import DiBB
dibb = DiBB(ndims=10, nblocks=2, fit_fn=sphere, verbose=True)
dibb.run(ngens=100)
```

When launching DiBB using the last command, you should notice the overhead of Ray starting up with multiple workers to handle the blocks. It is just a few seconds, and already allows you to play with dimensionality `ndims` much larger than you could with CMA-ES alone.
The algorithm used is more correctly the partially-separable version of CMA created by DiBB: *PS-CMA-ES*. [Check out the paper](https://exascale.info/assets/pdf/cuccu2022gecco.pdf) for more details.

## Cluster usage

To enable the full potential of DiBB you should get your hands on a few machines (one per each block, plus one for the head process), then:

- `pip install ray` on each of them
- Set up basic SSH key-pair authentication (here's a [quick script](https://github.com/giuse/devops/blob/master/pair_ssh_keys.sh)) 
- Mark down the IP addresses
- Customize a copy of [`ray_cluster_config.yaml`](ray_cluster_config.yaml)
(instructions inside)
- Initialize the cluster by running (locally) `ray up ray_cluster_config.yaml`

Now you can run the same code again, only this time it will find and utilize your new cluster rather than launching local processes. Notice the overhead is very small.

```python
# With a Ray cluster with 3 machines already running, this code will now
# run two blocks on two machines, plus the head node on a third machine
from dibb import DiBB
dibb = DiBB(ndims=10, nblocks=2, fit_fn=sphere, verbose=True)
dibb.run(ngens=100)
```

A more flexible way to call DiBB with multiple options is to use a `dict` (check `DiBB.defaults` for an example):

```python
from dibb import DiBB
dibb_config = {
    'ndims' : 10,
    'nblocks' : 2,
    'fit_fn' : sphere,
    'verbose' : True,
    'stopping_conditions' : {'ngens' : 100},
}
dibb = DiBB(**dibb_config).run()
```

You can find the complete list of accepted parameters, their descriptions and default values, in [DiBB's main file](dibb/dibb.py)

## Neuroevolution example in 5 minutess

*Neuroevolution* means training neural networks using evolutionary computation algorithms. This is not a supervised learning technique, so you need no labels and no differentiability. In a RL control problem for example, you can train a neural network to use as the policy (*Direct Policy Search*), and entirely ditch value functions, Markov chains and the whole classic RL framework.

First you will need to install the requirements (besides DiBB of course): 1. an optimizer, 2. a neural network and 3. a RL control environment.

```bash
pip install "dibb[cma] tinynet gym[classic_control]"
```
_(the quotes are here to escape the square parenthesis for `zsh`, which is currently the default shell on Mac)_

Then copy+paste the example below to a `.py` file and run it.
It should not take long, even on your local machine -- or you can launch a cluster of 3 machines first (using `ray_cluster_config.yaml`): the example below will run on the cluster with no further changes.

```python
# INSTALL REQUIREMENTS FIRST: dibb (with optimizer), neural network, RL environment:
# $ pip install "dibb[cma] tinynet gym[classic_control]"

import numpy as np
import ray
import tinynet  # https://github.com/giuse/tinynet/
import gym      # https://www.gymlibrary.ml/environments/classic_control/
import warnings # silence some gym warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gym')
from dibb import DiBB

# Set up the environment and network
env = gym.make("CartPole-v1")
nactions = env.action_space.n
ninputs = env.reset().size
# Just 2 neurons (for the Cartpole only has 2 actions: left and right)
# with linear activation `f(x)=x` are already enough
net = tinynet.FFNN([ninputs, nactions], act_fn=lambda x: x)

# The fitness function accumulates the episodic reward (basic gym gameplay)
def play_gym(ind, render=False):
    obs = env.reset(seed=1) # fix random seed: faster training but less generalization
    net.set_weights(ind) # set the weights into the network
    score = 0
    done = False
    while not done:
        if render: env.render() # you can watch it play!
        action = net.activate(obs).argmax() # pick action of neuron with highest act
        obs, rew, done, info = env.step(action)
        # With NE we ignore the availability of per-action reward (it's rarely 
        # available in nature anyway), and just accumulate it over the episode
        score += rew
    return score

###### THIS IS THE ACTUAL DIBB-SPECIFIC CODE ######

# Call `DiBB.defaults` for the full list, check `dibb/dibb.py` for descriptions
dibb_config = {
    'fit_fn' : play_gym,
    'minim_task' : False, # IMPORTANT: in this task we want to _maximize_ the score!
    'ndims' : net.nweights,
    'nblocks' : net.noutputs, # Let's use a block for each output neuron (2 here)
    'optimizer' : 'default', # CMA or LMMAES if installed, else Parameter Guessing
    # 'optimizer_options' : {'popsize' : 50} # Ray manages parallel fitness evals
}
dibb = DiBB(**dibb_config).run(ngens=15) # Cartpole is not a challenge for DPS/NE

###################################################

# Watch the best individual play!
best_fit = play_gym(dibb.best_ind, render=True)
print("Best fitness:", best_fit)

# You can even resume the run for a few more generations if needed:
# print("Resume training for 15 more generations")
# dibb.run(ngens=15)
# print("Best fitness:", play_gym(dibb.best_ind, render=True))

# Keep the console open at the end of the run, so you can play more with DiBB!
print("\n\nDropping into an interactive console now, feel free to inspect the `dibb` object at your leisure. For example, try running `dibb.defaults` to see all options and their default values; `dibb.cfg` for the values actually used for this run; `dibb.result` for a dict of the final mean and best individual found; `dibb.comm_dict` to see the communication dictionary with the BlockWorkers.\n")
import os; os.environ['PYTHONINSPECT'] = 'TRUE'
```

# Glossary

- **DiBB:** Distributed Black Box (optimization)
- **Block:** subset of parameters assumed to be correlated (intra-block); parameters in different blocks are assumed separable (inter-block). Notice that in most applications both full correlation and full separability between parameters are very unlikely. With DiBB you can better approximate real characteristics, and distribute the computation of blocks to different, dedicated machines as a bonus. Check the paper to know more.
- **Optimizer:** the underlying black box optimization algorithm that will be used on a block level. Notice that using an optimizer with DiBB automatically generates a partially-separable version of that algorithm: for example, using CMA-ES with DiBB you are actually running PS-CMA-ES.
- **Block Worker (BW):** the class, process and Ray actor that runs the base optimizer on one block. They are distributed to their own dedicated machines when a cluster is available. BWs always run asynchronously from each other, leveraging the assumption of block separability.
- **Fitness (function):** the objective function for the optimization. In classical Reinforcement Learning, the term "reward" indicates a reinforcement available per each interaction (denoted by states and actions). However, this is commonly not available in real-world processes. Black-box optimization instead uses a "fitness" evaluation, which is a score of proficiency of one solution (as it is a multi-agent setup) on the task. The quickest way to build a fitness from an environment coded with classic RL in mind is to accumulate the reward throughout the episode, though customizing a more proper definition of "fitness" beyond fixed extrinsic reward schemes can often improve performance quite drastically -- so keep that in mind.
- **Trial:** one evaluation of one individual on the task, running the fitness function. If the environment fitness is non-deterministic (e.g. random initial conditions), you can provide a better estimation of the individual's performance by averaging across multiple trials. DiBB has an option to do this automatically.
- **Fitness Evaluator (FE):** each BW can spawn a pool of FE, sized to the machine's computational resources (e.g. number of CPUs or GPUs). All fitness computation then runs through the FEs, as an effective way to manage (potentially complex and external) available resources. For example: load-scaling when the fitness function is resource intensive, such as a robotic control task running on a separate physics simulator.
- **Random seed:** setting a random seed for the (pseudo-)random number generator enforces reproducibility of the whole run. If tackling a non-deterministic problem using `ntrials>1`, each trial sets a separate seed out of a sequence (`[rseed, rseed+1, rseed+2, ... , rseed+ntrials-1]`).

## Instantiation hierarchy
- The cluster, if available, should be launched from a local machine, and include at least `nblocks+1` machines (one for each BW + one for the head/DiBB)
- If no cluster is available, the run is local: all "machines" mentioned below become one and the same (the very local machine), hosting all computation.
- DiBB runs on the head machine, which also hosts the Ray _object store_ (hosting the shared data: check out the [Ray documentation](https://docs.ray.io/en/latest/ray-core/objects/serialization.html#id1) for further details)
- At initialization, the DiBB object instantiates a list of BlockWorkers (BWs), one for each parameters block, each assigned to a dedicated machine on the Ray cluster (if available)
- At initialization, each BW creates one instance of the underlying optimizer, which is dedicated to the subset of parameters of its assigned block -- more precisely, it will instantiation the corresponding `opt_wrapper`, ensuring a compatible interface
- Each BW can be set up to execute the fitness function either on its own (either sequentially or in parallel), or instantiate a pool of FitnessEvaluators (FE), typically as many as the available computational resources (e.g. CPU cores or GPUs) -- see Advanced Usage below for more info
- If using the FEs, these run on the same machine of the BW that instantiated them: at any given moment, the machine resources are either dedicated to fitness evaluation or to (underlying) BBO update
- Further distributing the fitness computation to extra available machines is a planned feature for a future release

## Run hierarchy
- DiBB instantiates the BWs, then remains at watch, executing the function `print_fn(dibb)` (if enabled) at custom regular intervals
- If chosen, each BW instantiates immediately an independent, machine-local pool of FEs to evaluate the individuals generated by its optimizer
- The BW then immediately starts the generation loop of the underlying BBO, using a classic "ask&tell" interface:
    - `bbo.ask()` needs to return a population of individuals, which are then evaluated on the fitness function by the BW (either locally or on the FEs)
    - In case of multiple trials per individual, the BW aggregates the resulting episodic fitness scores using a customizable aggregation function
    - `bbo.tell(inds, fits)` then uses the individuals and fitness scores to execute one update step of the BBO algorithm
- The BWs run asynchronously from each other, leveraging the assumption of inter-block separability. Potentially they could even run on machines with different performance, and the blocks can differ in size, though special care is then necessary to ensure load balance
- The BWs can individually decide to pause their execution, typically temporarily if convergence is achieved in one block (as the _moving target_ problem should allow resuming the search shortly), or because of an inescapable termination criterion -- either way, this is communicated to the main DiBB process, which can itself request the BWs termination as needed, such as if they all reach termination.

## Advanced Usage

- The complete list of available options to DiBB is available directly in the code to avoid duplication and simplify maintenance; you can check the top of [the main DiBB file](dibb/dibb.py). You can ask directly the DiBB class (`DiBB.defaults`) or even the module (`from dibb import options`), but the source has more useful descriptions for each item.
- The `printfn` option accepts a callable object (function, method, lambda, even a class implementing `__call__()`) which should take a DiBB object as parameter: this is then executed at regular intervals by the main node while waiting for the workers to finish (run report)
- Termination conditions can be passed more explicitly using the `stopping_conditions` option (e.g. `DiBB(stopping_conditions={'nevals' : 10e5})`
- The `hooks` parameter accepts callables (either single or lists of them) which are executed before each generation (for the BW) or before each fitness evaluation (for the FE). This is often useful to customize the algorithm tracking and behavior, giving the user a direct "hook" into the algorithm's mechanics
- For reproducibility we use `pipenv` for development. If you wish to contribute a pull request (or simply to run the tests), do a `python3 -m pip install pipenv` followed by `python3 -m pipenv sync` to install the exact same environment we use for development (tip: don't use `pipenv install`, as it will update all libraries and edit the `Pipfile.lock`). You can then run your virtualenv shell with `python3 -m pipenv shell`, or run your code with `python3 -m pipenv run python your_experiment.py`
- To run DiBB on a cluster:
    1. Download the [Ray cluster configuration template](ray_cluster_config.yaml)
    2. Open it in a text editor and follow instructions: you need to replace all placeholders (format looks like this: `%replace_me%`) with the actual data of your cluster (machine IPs, ray command, etc.)
    3. Save, close, and keep on your local machine, which should be able to reach the network hosting the cluster machines
    4. Install DiBB, Ray and all libraries for your experiment Ray on all machines of your cluster, including any external resource you need for your fitness evaluation
    5. Launch the cluster by running `ray up ray_cluster_config.yaml` on your local machine
    6. Launch your code by using [Ray's job submission API](https://docs.ray.io/en/latest/cluster/job-submission.html#ray-job-cli). For a quick run (or to avoid problems e.g. with virtual environments through the job submission API) you can also simply connect to the head node with `ssh` and launch your code from there. Done!
- You can run fitness computation either: 1. on the BlockWorker, sequentially; 2. on the BlockWorker, in parallel (the default); 3. on the FitnessEvaluator pool, ideal for complex fitnesses such as needing to launch external simulators on each machine in the cluster
- To support complex fitness evaluation, especially on the FEs, and specifically to avoid the problems in pickling (serialization) complex fitness objects for distribution on the cluster: you can pass the `fit_fn` as a *string*, in which case it will be sent as a string and then `eval()`ed on the target machine. To fully leverage this, try the following:
    1. Create a `Fitness` class
    2. In its `__init__`, launch the simulator or connect to external resources
    3. Define a `__call__(self, ind)` method to take an individual and return its fitness (on one run; use the `ntrials` option for nondeterministic fitnesses)
    4. Pass to DiBB a string with a call to the constructor: `DiBB(fit_fn="Fitness()")`
    5. DiBB will send the string to the remote machines, which will then run`fitness = eval("Fitness()")`, thus creating locally the exact environment required for individual evaluation
- Debugging is best done following [Ray's guidelines and tools](https://docs.ray.io/en/latest/ray-observability/ray-debugging.html). Particularly, be aware that Ray silently collects uncaught exceptions on the remote machines, and upon errors it does not flush the local stdout buffers to the main node. 

This means that if your code breaks a remote machine in the cluster, it will look like DiBB is failing silently and without support. There is nothing we can do about it (yet). Please be patient, either trust DiBB and simplify your fitness, or carefully debug and submit a fix as pull request if it's our fault. Thank you!

Here's a snippet to help:

```python
# Classic printf debugging, add as many as needed
import inspect
lineno = inspect.getframeinfo(inspect.currentframe()).lineno
print(f"\n\n\tline {lineno} of file {__file__}\n\n" )
```

then check the logs ("workers" here are machines in a Ray cluster):

```bash
for log in $(find /tmp/ray/session_latest/logs -name 'worker*'); do
    echo -e "\n\n-------- $log --------\n"
    cat $log
done
```

## Acknowledgements

This work has been published at GECCO 2022, The 24th Genetic and Evolutionary Computation Conference.
[[Download the paper here]](https://exascale.info/assets/pdf/cuccu2022gecco.pdf), bibtex here:

```bibtex
@inproceedings{cuccu2022gecco,
  author    = {Cuccu, Giuseppe and Rolshoven, Luca and Vorpe, Fabien and Cudr\'{e}-Mauroux, Philippe and Glasmachers, Tobias},
  title     = {{DiBB}: Distributing Black-Box Optimization},
  year      = {2022},
  isbn      = {9781450392372},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  doi       = {10.1145/3512290.3528764},
  booktitle = {Proceedings of the Genetic and Evolutionary Computation Conference},
  pages     = {341–349},
  numpages  = {9},
  keywords  = {parallelization, neuroevolution, evolution strategies, distributed algorithms, black-box optimization},
  location  = {Boston, Massachusetts},
  series    = {GECCO '22},
  url       = {https://exascale.info/assets/pdf/cuccu2022gecco.pdf}
}
```

The experiment code to reproduce our COCO results is [available here](https://github.com/eXascaleInfolab/dibb_coco), created and maintained by Luca [(@rolshoven)](https://github.com/rolshoven).

Since 2021 -- initially as part of his Master thesis, then out of his personal quest for excellence -- Luca Sven Rolshoven [(@rolshoven)](https://github.com/rolshoven) has contributed to this project with engaging discussions, reliable debugging, the original "hooks" feature, [fundamental testing](https://github.com/eXascaleInfolab/dibb_coco), and managing our cluster of 25 decommissioned office computers :)  
The git history of this repository has been wiped at publication for privacy concerns, but his contribution should not go unacknowledged nor underestimated. Thanks Luca!
