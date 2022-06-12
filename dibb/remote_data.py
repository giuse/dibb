import ray

@ray.remote
class RemoteData:
    """Shared data structure on the objects store.

       Allows for efficient (partial) getting and setting.
       Mostly used for communication and stats tracking.
    """

    def __init__(self, init_data=None):
        self.reset(init_data)

    def reset(self, data):
        self.data = data

    def get(self, *keys, apply_fn=None):
        # Allows deep getting by providing multiple keys,
        # and server-side data aggregation by using an apply_fn
        ret = self.data
        keys = list(keys)
        while keys: ret = ret[keys.pop(0)]
        if apply_fn: ret = apply_fn(ret)
        return ret

    def set(self, *keys, val):
        # Allows nested value setting by providing multiple keys
        keys = list(keys)
        trg = self.data
        last_key = keys.pop()
        for key in keys: trg = trg[key]
        trg[last_key] = val
