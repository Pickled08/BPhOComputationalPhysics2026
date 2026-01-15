import numpy as np
from numba import njit
import math

#Inital parameters
m = 0.5 #mass of ball kg
g = 9.81 #gravity ms^-2
c = 0.5 #coefficient of restitution
u = 0 #initial velocity ms^-1
h = 10 #initial height m
dt = 0.01 #time step s
nBounce = 0 #number of bounces
maxBounces = 10 # max number of bounces
simulationTime = 10 # sim time s

t = 0.0 #Time s
x=0 #Displacement  from ground m
v=0 #Velocity
timeIterations = int(simulationTime/dt)


@njit(fastmath=True)
def bounce(h, v0, g, dt, timeIterations, c):
    xarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations)

    x = h
    v = v0
    t = 0.0
    nBounce = 0

    for i in range(timeIterations):
        # update velocity and position
        v = v + g * dt
        x = x - v * dt
        t += dt

        # bounce
        if x < 0.0:
            x = 0.0
            v = -c * v
            nBounce += 1

        xarr[i] = x
        tarr[i] = t
        varr[i] = v

    return tarr, xarr
def pack_results(tarr,xarr):
    ltarr= []
    lxarr = []
    for i in range(timeIterations):
        ltarr.append(tarr[i])
        lxarr.append(xarr[i])
    results = [(ltarr, lxarr, 1)]

    return results

def bounceGraph():
    tarr, xarr = bounce(h=h,v0=v,g=g,dt=dt,timeIterations=timeIterations,c=c)  # unpack
    return pack_results(tarr, xarr)


