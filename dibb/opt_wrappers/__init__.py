import sys

# Register names (and different spellings) for available BBO wrappers
all_opt_names = {
    'default' : ['lmmaes', 'cma', 'parameter_guessing'], # customizable
    'cma' : ['cma-es', 'cma', 'cmaes'],
    'cma-ipop' : ['cma-es-ipop', 'cma-ipop', 'cmaes-ipop'],
    'lmmaes' : ['lmmaes', 'lm-ma-es'],
    'parameter_guessing' : ['parameter_guessing', 'guess', 'guessing', 'pg'],

    # Register new optimizers here (and in `load_opt()` below)

}

# Load correct wrapper class based on name passed as parameter
def load_opt(opt_name):
    if opt_name == 'default':
        for opt in all_opt_names['default']:
            try:
                opt_cls = load_opt(opt)
                print(f"Using `{opt}` as default optimizer")
                return opt_cls
            except ImportError as exc:
                pass # just try the next one
        print(f"ERROR: could not find an installed/available default optimizer")
        raise exc # exception of the last available default option
    elif opt_name in all_opt_names['cma']:
        from dibb.opt_wrappers.cma_wrapper import CmaWrapper
        return CmaWrapper
    elif opt_name in all_opt_names['cma-ipop']:
        from dibb.opt_wrappers.cma_ipop_wrapper import CmaIpopWrapper
        return CmaIpopWrapper
    elif opt_name in all_opt_names['lmmaes']:
        from dibb.opt_wrappers.lmmaes_wrapper import LmmaesWrapper
        return LmmaesWrapper
    elif opt_name in all_opt_names['parameter_guessing']:
        from dibb.opt_wrappers.parameter_guessing import ParameterGuessing
        return ParameterGuessing

        # Register new optimizers here (and in `all_opt_names` above)

    else:
        raise ParameterError(
            f"Unknown optimizer: `{opt_name}` -- " + \
            "Check available options in `dibb.opt_wrappers.all_opt_names`")
