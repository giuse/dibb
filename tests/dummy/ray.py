import inspect

# BEWARE: this is NOT A COMPLETE INTERFACE,
# but still helpful to catch a majority of bugs

print('\n\n\t DUMMY RAY LOADED \n\n')

def init(*args, **kwargs): pass
def wait(*args, **kwargs): return [*args,[]]
def get(arg): return arg
def nodes(): return {'NodeManagerHostname' : 'localhost'}

class DummyFunction:

    def __init__(self, original_function, caller_object):
        self.original_function = original_function
        self.caller_object = caller_object

    # Calling `remote()` on an actor method
    def remote(self, *args, **kwargs):
        return self.original_function(self.caller_object, *args, **kwargs)
    __call__ = remote


# `@remote` decorator for ray actors
def remote(orig_class=None, **kwargs):

    # Recurse if calling `ray.remote(resources)`
    if orig_class is None: return remote

    # Dummy class to wrap original
    class DummyClass(orig_class):
        def options(*args, **kwargs): return DummyClass

        # Initialization with `classname.remote()`
        def remote(*args, **kwargs):
            obj = DummyClass(*args, **kwargs)
            for name, function in orig_class.__dict__.items():
                if inspect.isfunction(function):
                    setattr(obj, name, DummyFunction(function, obj))
            return obj

    print("Wrapping", orig_class.__name__)
    return DummyClass
