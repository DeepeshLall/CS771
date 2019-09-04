import numpy as np
import random 
import time as tm

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def getObj( X, y,C, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

# Get a stochastic gradient for the CSVM objective
# Choose a random data point per iteration
def getCSVMSGrad( X, y,C,theta ):
	w = theta[0:-1]
	b = theta[-1]
	n = y.size
	i = random.randint( 0, n-1 )
	x = X[i,:]
	discriminant = ((x.dot( w ) + b) * y[i])**2
	g = 0
	if discriminant < 1:
		g = -1
	delb = C * g * y[i]
	delw = w + C * n * (x * g) * y[i]
	return np.append( delw, delb )

# Given a gradient oracle, a step length oracle, an initialization,
# perform GD for a specified number of steps (horizon)
# An "oracle" is a fancy name for a function that does a certain job perfectly
def doGD( X,y,C,gradFunc, stepFunc, init, horizon = 1 ):
	#objValSeries = np.zeros( (horizon,) )
	#timeSeries = np.zeros( (horizon,) )
	#totTime = 0
	theta = init
	cumulative = init
	for t in range( horizon ):
		#tic = tm.perf_counter()
		delta = gradFunc( X ,y,C, theta )
		theta = theta - stepFunc( delta, t ) * delta
		cumulative = cumulative + theta
		#toc = tm.perf_counter()
		#totTime = totTime + (toc - tic)
		#objValSeries[t] = getCSVMObjVal( X,y,C,cumulative/(t+2) )
		#timeSeries[t] = totTime
	return (cumulative/(horizon+1))

def mySVM( X ):
	return X.dot(w) + b

# Quite standard for strongly convex but non-smooth objectives like CSVM
def getStepLength( grad, t ):
	return 0.1/(t+1)

# Get the CSVM objective value in order to plot convergence curves
def getCSVMObjVal( X,y,C,theta ):
	w = theta[0:-1]
	b = theta[-1]
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * np.sum( hingeLoss )


################################
# Non Editable Region Starting #
################################
def solver( X, y, C, timeout, spacing ):
	(n, d) = X.shape
	t = 0
	totTime = 0
	
	# w is the normal vector and b is the bias
	# These are the variables that will get returned once timeout happens
	w = np.ones( (d,) )
	b = 0
	tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

	# You may reinitialize w, b to your liking here
	# You may also define new variables here e.g. eta, B etc
	eta = 2
	b = 1
	C = 0.1
	theta = np.append(w,b)
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

		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		theta_SGD = doGD( X, y,C,getCSVMSGrad, getStepLength, theta, horizon = 1 )
		#print (w)
		w_run = theta_SGD[0:-1]
		b_run = theta_SGD[-1]
		theta = theta_SGD
		if(getObj(X,y,C,w_run,b_run) < getObj(X,y,C,w,b)):
			w = w_run
			b = b_run
		print(getObj(X,y,C,w,b))
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
	return (w, b, totTime)