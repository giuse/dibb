from dibb.dibb import DiBB
options = defaults = DiBB.defaults # Syntactic sugar

# Classic fmin interface (just for compatibility)
def fmin(fit_fn, init_ind):
    dibb = DiBB(init_ind, fit_fn=fit_fn, \
                print_interval=0.01,
                print_fn=lambda dibb: print(dibb))
    dibb.optimize()
    return (dibb.best_ind, dibb)
