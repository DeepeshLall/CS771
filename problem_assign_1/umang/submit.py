import numpy as np
import random as rnd
import time as tm
import math
from matplotlib import pyplot as plt
import sys

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

#TODO: MOVE THIS TO main
gradW = None
gradB = None
X = None
y = None
C = 1
n = None
d = None
eta = 0.0014
xw = None	#dot product of x_i and w
randpermInner = -1
randperm = None
cyclicPermInner = 0

def getObj( X, y, w, b ):
    hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
    return 0.5 * w.dot( w ) + C * hingeLoss.dot(hingeLoss)

def getRandpermCoord():
    global randperm, randpermInner
    if randpermInner >= d or randpermInner < 0:
        randpermInner = 0
        randperm = np.random.permutation( d + 1 )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]

def getCyclicCoord():
    global cyclicPermInner
    if cyclicPermInner >= d or cyclicPermInner < 0:
        cyclicPermInner = 0
    else:
        cyclicPermInner += 1
    return cyclicPermInner
    

def getCDGrad(w,b,t):
    xwby = np.multiply(xw,y)
    loss10 = np.zeros((n,))
    loss10[xwby < 1] = -1
    if t < d:
        gradW = w[t]
        yDotX_t = np.multiply(X[:,t], y*2*C)
        diff1 = np.add(-xwby,1)
        c2ydiff1 = np.multiply(diff1, yDotX_t)
        gradW += loss10.dot(c2ydiff1)
        return gradW
    else:
        gradB = 0
        diff1 = np.add(-xwby, 1)
        c2ydiff1 = np.multiply(diff1,2*C*y)
        gradB = loss10.dot(c2ydiff1)
        return gradB


def getStepLength( t ):
    return eta/math.sqrt(t+1)
    # return eta/math.sqrt(math.sqrt(math.sqrt(t+1)))
    # return eta/math.sqrt(t//(d+1) + 1)



################################
# Non Editable Region Starting #
################################
def solver( X1, y1, C1, timeout, spacing ):
    global X,y,C,n,d,xw, gradW, gradB, eta
    X = X1
    y = y1
    C = C1
    (n1, d1) = X.shape
    n = n1;d = d1
    t = 0
    totTime = 0

    # w is the normal vector and b is the bias
    # These are the variables that will get returned once timeout happens
    w = np.zeros( (d,) )
    w_run = np.zeros( (d,))
    b = 0
    b_run = 0

    xw = np.dot(X,w)
    xw = np.add(xw, b)

    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
    t = 0
    # You may reinitialize w, b to your liking here
    # You may also define new variables here e.g. eta, B etc

################################
# Non Editable Region Starting #
################################

    while True:
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                print("gradb = ",gradB)
                for i in range(d):
                    print(getCDGrad(w_run,b_run,i))
                    

                return (w_run, b_run, totTime)
            else:
                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
            print(t,getObj(X,y,w_run,b_run),file=sys.stderr)
        # jt = rnd.randint(0,d) #jt = d represents bias term
        # jt = getRandpermCoord()
        jt = getCyclicCoord()
        if(jt < d):
            gradW = getCDGrad(w_run,b_run,jt)
            old = w_run[jt]
            w_run[jt] = w_run[jt] - gradW * getStepLength(t)
            deltaXw = (w_run[jt] - old) * X[:,jt]
            xw = xw + deltaXw
        else:
            gradB = getCDGrad(w_run,b_run,jt)
            b_run_old = b_run
            b_run = b_run - gradB * getStepLength(t)
            xw = np.add(xw,b_run-b_run_old)
        t = t + 1


        # Write all code to perform your method updates here within the infinite while loop
        # The infinite loop will terminate once timeout is reached
        # Do not try to bypass the timer check e.g. by using continue
        # It is very easy for us to detect such bypasses - severe penalties await

        # Please note that once timeout is reached, the code will simply return w, b
        # Thus, if you wish to return the average model (as we did for GD), you need to
        # make sure that w, b store the averages at all times
        # One way to do so is to define two new "running" variables w_run and b_run
        # Make all GD updates to w_run and b_run e.g. w_run = w_run - step * delw
        # Then use a running average formula to update w and b
        # w = (w * (t-1) + w_run)/t
        # b = (b * (t-1) + b_run)/t
        # This way, w and b will always store the average and can be returned at any time
        # w, b play the role of the "cumulative" variable in the lecture notebook
        # w_run, b_run play the role of the "theta" variable in the lecture notebook

    return (w_run, b_run, totTime) # This return statement will never be reached
