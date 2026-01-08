import random
import numpy as np
maxSteps = 10000

stepSize = 1

def randomWalk():
    x, y = [], []
    xn, yn = 0, 0 
    for _ in range(maxSteps):
        angle = random.uniform(0, 2 * 3.141592653589793)
        xn += stepSize * np.cos(angle)
        yn += stepSize * np.sin(angle)
        x.append(xn); y.append(yn)
    return x, y
    

def NRadomWalks(max):

    listOfPlots = []

    for i in range(max):
        x, y = randomWalk()
        resultsTuple = (x,y,i)
        listOfPlots.append(resultsTuple)
    
    return listOfPlots
    