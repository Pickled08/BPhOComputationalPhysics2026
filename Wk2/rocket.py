import numpy as np
from numba import njit
import math

#Innital conditions
mr = 50.0          #mass of rocket kg
mf = 25.0          #mass of fuel kg
g = 9.81           #gravity ms^-2
thrust = 5000.0    #thrust initial N
burnTime = 5.0     #burn time s
rho0 = 1.225       #kg/m^3, sea level air density
H = 8500           #m, scale height of atmosphere
Cd = 0.75          #drag coefficient
A = 0.02           #m^2, cross-sectional area of rocket
dt = 0.01          #time step s
simulationTime = 60.0 #sim time s
timeIterations = int(simulationTime/dt)



t = 0.0 #Time s
x=0 #Displacement  from ground m
v=0 #Velocity
mt = mr + mf #total mass

#@njit(fastmath=True)
def rocket(mt, g, thrust, burnTime, dt, timeIterations, t, x , v, rho0, H, Cd, A, mf):
    xarr = np.empty(timeIterations)
    tarr = np.empty(timeIterations)
    varr = np.empty(timeIterations) 
    aarr = np.empty(timeIterations)

    
    for i in range(timeIterations):

        if t <= burnTime:
            T = thrust
        else:
            T = 0.0

        rho = rho0 * math.exp(-x / H)   
        D = 0.5 * rho * Cd * A * v * abs(v)
        acceleration = (T - np.sign(v) * D) / mt - g
        
        v = v + acceleration * dt

        x = x + v * dt

        if t <= burnTime:
            mt -= (mf / burnTime) * dt


        if x < 0:
            x = 0
            v = 0

        xarr[i] = x
        tarr[i] = t
        varr[i] = v
        aarr[i] = acceleration

        t += dt

        if abs(t - burnTime) < dt:
            print("Burnout at t =", t, " altitude =", x, " mass =", mt)

        if i == timeIterations - 1:
            print("Final mass =", mt)

        if abs(v) > 5000:
            print("Unphysical velocity!", v, "at t =", t)


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
    tarr, xarr, varr, aarr = rocket(mt, g, thrust, burnTime, dt, timeIterations, t, x , v, rho0, H, Cd, A, mf)
    return pack_results(tarr,xarr, varr, aarr)



