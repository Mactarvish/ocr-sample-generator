import random
import numpy as np


def parse_random(r):
    if isinstance(r, tuple):
        assert len(r) > 0, r 
        assert len(r) == 2 and r[0] <= r[1], r
        if isinstance(r[0], int) and isinstance(r[1], int):
            return np.random.randint(*r)
        else:
            return np.random.uniform(*r)
    elif isinstance(r, list):
        assert len(r) > 0, r
        return random.sample(r, 1)[0]
    elif isinstance(r, (int ,float)):
        return r
    else:
        raise ValueError(f"{r} {type(r)}")

