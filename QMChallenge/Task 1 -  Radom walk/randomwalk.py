import numpy as np
from numba import njit
from numba import cuda
import math

maxSteps = 1000_000
stepSize = 1.0

@njit(fastmath=True)
def random_walk_numba(kappa, res, maxSteps, stepSize):
    
    record_interval = max(1, int(1 / res))
    n_points = maxSteps // record_interval

    x = np.empty(n_points, dtype=np.float64)
    y = np.empty(n_points, dtype=np.float64)

    xn = 0.0
    yn = 0.0
    angle = np.random.uniform(0.0, 2.0 * math.pi)

    j = 0

    for i in range(maxSteps):
        angle += np.random.normal(0.0, 1.0 / (kappa + 1e-9))
        xn += stepSize * math.cos(angle)
        yn += stepSize * math.sin(angle)

        if i % record_interval == 0:
            x[j] = xn
            y[j] = yn
            j += 1

    return x, y


@njit(fastmath=True)
def NRandomWalks_numba(n_walks, kappa, res, maxSteps, stepSize):
    
    record_interval = max(1, int(1 / res))
    n_points = maxSteps // record_interval

    xs = np.empty((n_walks, n_points))
    ys = np.empty((n_walks, n_points))

    for i in range(n_walks):
        x, y = random_walk_numba(kappa, res, maxSteps, stepSize)
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

def NRandomWalks(n_walks, kappa, res, maxSteps, stepSize):
    xs, ys = NRandomWalks_numba(n_walks, kappa, res, maxSteps, stepSize)  # unpack
    return pack_results(xs, ys)
