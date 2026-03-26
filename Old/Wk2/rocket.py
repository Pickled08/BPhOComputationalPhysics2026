import numpy as np
from numba import njit
from numba import cuda
import math

#Initial conditions
mr = 50000            #mass of rocket kg (dry mass + payload)
mf = 518500           #mass of fuel kg 
g = 9.81              #gravity ms^-2
thrust = 7607000      #thrust initial N 
burnTime = 162.0      #burn time s 
rho0 = 1.225          #kg/m^3, sea level air density
Cd = 0.3              #drag coefficient
A = 10.52             #m^2, cross-sectional area of rocket
dt = 0.001           #time step s
simulationTime = 600.0 #sim time s
timeIterations = int(simulationTime/dt)


t = 0.0 #Time s
x=0 #Displacement  from ground m
v=0 #Velocity
mt = mr + mf #total mass

@njit(fastmath=True)
def rocket(mt, g, thrust, burnTime, dt, timeIterations, t, x , v, rho0, Cd, A, mf):
    xarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)

    for i in range(timeIterations):

        if t <= burnTime:
            Thr = thrust - 2
        else:
            Thr = 0.0
        
        D = 0.5*rho0*(v*abs(v))*A*Cd

        acceleration = ((Thr - D) / mt) - g 

        v = v + acceleration*dt

        x = x + v * dt

        if t <= burnTime:
            mt -= (mf / burnTime) * dt


        if x < 0:
            x = 0
            v = 0
            acceleration = 0

        xarr[i] = x
        tarr[i] = t
        varr[i] = v
        aarr[i] = acceleration

        t += dt

    return tarr, xarr, varr, aarr 

def pack_results(tarr, xarr, varr, aarr):
    ltarr= []
    lxarr = []
    lvarr = []
    laarr = []
    for i in range(timeIterations):
        ltarr.append(tarr[i])
        lxarr.append(xarr[i])
        lvarr.append(varr[i])
        laarr.append(aarr[i])
    results = [(ltarr, lxarr, 1), (ltarr, lvarr, 2), (ltarr, laarr, 3)]
    return results

def rocketGraph():
    tarr, xarr, varr, aarr = rocket(mt, g, thrust, burnTime, dt, timeIterations, t, x , v, rho0, Cd, A, mf)
    return pack_results(tarr,xarr, varr, aarr)