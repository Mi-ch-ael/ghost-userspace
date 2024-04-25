import numpy as np
from gymnasium import spaces

def generate_zeroed_sample(space):
        if isinstance(space, spaces.Box):
            return np.zeros(space.shape, dtype=space.dtype)
        elif isinstance(space, spaces.Discrete):
            return 0
        elif isinstance(space, spaces.MultiDiscrete):
            return np.zeros(space.nvec, dtype=space.dtype)
        elif isinstance(space, spaces.MultiBinary):
            return np.zeros(space.n, dtype=int)
        elif isinstance(space, spaces.Tuple):
            return tuple(generate_zeroed_sample(subspace) for subspace in space.spaces)
        elif isinstance(space, spaces.Dict):
            return {key: generate_zeroed_sample(subspace) for key, subspace in space.spaces.items()}
        else:
            raise NotImplementedError(f"Space type {type(space)} is not supported")