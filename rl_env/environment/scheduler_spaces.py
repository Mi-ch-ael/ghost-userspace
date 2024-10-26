import numpy as np
from gymnasium import spaces

def generate_zeroed_sample(space):
        if isinstance(space, spaces.Box) and space.shape == (1,):
             return np.array([0.0], dtype=np.float32)
        elif isinstance(space, spaces.Discrete):
            return np.int64(0)
        elif isinstance(space, spaces.Tuple):
            return tuple(generate_zeroed_sample(subspace) for subspace in space.spaces)
        elif isinstance(space, spaces.Dict):
            return {key: generate_zeroed_sample(subspace) for key, subspace in space.spaces.items()}
        else:
            raise NotImplementedError(f"Space type {type(space)} is not supported")