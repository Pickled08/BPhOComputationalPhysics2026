import numpy as np
from numba import njit
from numba import cuda
import math

maxSteps = 1000_000
stepSize = 1.0

@njit(fastmath=True)
def random_walk_numba(kappa):
    x = np.empty(maxSteps, dtype=np.float64)
    y = np.empty(maxSteps, dtype=np.float64)

    xn = 0.0
    yn = 0.0
    angle = np.random.uniform(0.0, 2.0 * math.pi)

    for i in range(maxSteps):
        # Von Mises via rejection sampling (Numba-compatible)
        angle += np.random.normal(0.0, 1.0 / (kappa + 1e-9))
        xn += stepSize * math.cos(angle)
        yn += stepSize * math.sin(angle)
        x[i] = xn
        y[i] = yn

    return x, y


@njit(parallel=True)
def NRandomWalks_numba(n_walks, kappa):
    xs = np.empty((n_walks, maxSteps))
    ys = np.empty((n_walks, maxSteps))

    for i in range(n_walks):
        x, y = random_walk_numba(kappa)
        xs[i] = x
        ys[i] = y

    return xs, ys

def pack_results(xs, ys):
    results = []
    for i in range(xs.shape[0]):
        results.append((
            xs[i].tolist(),
            ys[i].tolist(),
            i + 1
        ))
    return results

def NRandomWalks(a, b):
    xs, ys = NRandomWalks_numba(a, b)  # unpack
    return pack_results(xs, ys)
