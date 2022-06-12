import sys

# Register names (and different spellings) for available BBO wrappers
all_opt_names = {
    'cma' : ['cma-es', 'cma', 'cmaes'],
    'cma-ipop' : ['cma-es-ipop', 'cma-ipop', 'cmaes-ipop'],
    'lmmaes' : ['lmmaes', 'lm-ma-es'],

    # Register new optimizers here (and below)

    # Leave PG last to be used as `default`
    'parameter_guessing' : ['parameter_guessing', 'guess', 'guessing', 'pg'],
}

# Load correct wrapper class based on name passed as parameter
def load_opt(opt_name):
    try:
        if opt_name in all_opt_names['cma']:
            # pip install cma -- https://pypi.org/project/cma
            from dibb.opt_wrappers.cma_wrapper import CmaWrapper
            return CmaWrapper
        elif opt_name in all_opt_names['cma-ipop']:
            # based on CMA (see above)
            from dibb.opt_wrappers.cma_ipop_wrapper import CmaIpopWrapper
            return CmaIpopWrapper
        elif opt_name in all_opt_names['lmmaes']:
            # pip install lmmmaes -- https://pypi.org/project/lmmaes
            from dibb.opt_wrappers.lmmaes_wrapper import LmmaesWrapper
            return LmmaesWrapper
        elif opt_name in all_opt_names['parameter_guessing']:
            # PM is directly implemented in the "wrapper"
            from dibb.opt_wrappers.parameter_guessing import ParameterGuessing
            return ParameterGuessing

        # Register new optimizers here (and above)

        elif opt_name == 'default':
            # Check what is installed and available
            for opt in all_opt_names.keys():
                try:
                    opt_cls = load_opt(opt)
                    print(f"Using {opt} as default optimizer\n")
                    return opt_cls
                except ImportError:
                    pass # just try the next one
            return load_opt('should not reach here')
        else:
            raise ValueError(
                f"Unknown optimizer: `{opt_name}` -- " + \
                "Check available options in `dibb.opt_wrappers.all_opt_names`")

    # Missing modules will be caught here
    except ImportError as exc:
        print(f"\n\n\tFAILED TO IMPORT MODULE `{opt_name}`, please make sure it is installed on the system.\n")
        import sys
        sys.exit()
