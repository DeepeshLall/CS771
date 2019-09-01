import numpy as np
import random as rnd
import time as tm
import math
from matplotlib import pyplot as plt

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

X = None
y = None
C = 1
n = None
d = None
eta = 0.0002
xw = None	#dot product of x_i and w
randpermInner = -1
randperm = None

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


def getCDGrad(w,b,t):
	if t < d:
		gradW = w[t]
		for i in range(n):
			gradW +=  (2*C*y[i]*(X[i][t])*(1-(xw[i]+b)*y[i])*(0 if (y[i]*(xw[i]+b) >= 1) else -1))
		return gradW
	else:
		gradB = 0
		for i in range(n):
			gradB +=  (2*C*y[i]*(1-(xw[i]+b)*y[i])*(0 if (y[i]*(xw[i]+b) >= 1) else -1))
		return gradB


def getStepLength( t ):
    return eta/math.sqrt(t+1)
    # return eta/math.sqrt(math.sqrt(t+1))
	# return eta/math.sqrt(t//(d+1) + 1)



################################
# Non Editable Region Starting #
################################
def solver( X1, y1, C1, timeout, spacing ):
	global X,y,C,n,d,xw
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

	xw = np.dot(X,w)

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
				return (w, b, totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
			# print(t,getObj(X,y,w,b))
		# jt = rnd.randint(0,d) #jt = d represents bias term
		jt = getRandpermCoord()
		w_run[:] = w[:]
		b_run = b
		if(jt < d):

			gradW = getCDGrad(w_run,b_run,jt)
			w_run[jt] = w_run[jt] - gradW * getStepLength(t)
			old = w[jt]
			for i in range(d):
				w[i] = (w[i]*(t) + w_run[i])/(t+1)
			for i in range(n):
				xw[i] += (w[jt] - old) * X[i][jt]
		else:
			gradB = getCDGrad(w_run,b_run,jt)
			b_run = b_run - gradB * getStepLength(t)
			b = (b*(t) + b_run)/(t+1)

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

	return (w, b, totTime) # This return statement will never be reached
