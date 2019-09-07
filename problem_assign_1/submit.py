import numpy as np
import random as rnd
import time as tm
import math

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

RMS_step = None
RMS_grad = None
eta = 0.001
xw = None	#dot product of x_i and w + b
randpermInner = -1
randperm = None

def getObj( X, y, w, b, C ):
    hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
    return 0.5 * w.dot( w ) + C * hingeLoss.dot(hingeLoss)

def getRandpermCoord(d):
    global randperm, randpermInner
    if randpermInner >= d or randpermInner < 0:
        randpermInner = 0
        randperm = np.random.permutation( d + 1 )
        return randperm[randpermInner]
    else:
        randpermInner = randpermInner + 1
        return randperm[randpermInner]


def getCDGrad(X,y,C,w,b,t):
    xwby = np.multiply(xw,y)
    (n,d) = X.shape
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

def getStepLength( jt ):
    return math.sqrt(eta*eta + RMS_step[jt])/math.sqrt(eta*eta + RMS_grad[jt])

################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
    (n, d) = X.shape
    t = 0
    totTime = 0
    
    # w is the normal vector and b is the bias
    # These are the variables that will get returned once timeout happens
    w = np.zeros( (d,) )
    b = 0
    tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

    # You may reinitialize w, b to your liking here
    # You may also define new variables here e.g. eta, B etc
    global xw , RMS_grad,RMS_step
    randPermInner = -1
    rho = 0.95
    RMS_step = np.zeros((d+1,))
    RMS_grad = np.zeros((d + 1,))
    xw = np.dot(X,w)
    xw = np.add(xw, b)
    



################################
# Non Editable Region Starting #
################################
    while True:
        t = t + 1
        if t % spacing == 0:
            toc = tm.perf_counter()
            totTime = totTime + (toc - tic)
            if totTime > timeout:
                return (w, b, totTime)
            else:
                tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

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

        jt = getRandpermCoord(d)
        if(jt < d):
            gradW = getCDGrad(X,y,C,w,b,jt)
            RMS_grad[jt] = rho * RMS_grad[jt] + (1 - rho) * (gradW*gradW)
            step = gradW * getStepLength(jt)
            RMS_step[jt] = rho * RMS_step[jt] + (1 - rho)* (step*step)
            old = w[jt]
            w[jt] = w[jt] - step
            xw = xw + (w[jt] - old) * X[:,jt]
        else:
            gradB = getCDGrad(X,y,C,w,b,jt)
            RMS_grad[jt] = rho* RMS_grad[jt] + (1 - rho) * (gradB*gradB)
            step = gradB * getStepLength(jt)
            RMS_step[jt] = rho*RMS_step[jt] + (1 - rho) * (step*step)
            b_old = b
            b = b - step
            xw = np.add(xw,b-b_old)
    return (w, b, totTime) # This return statement will never be reached