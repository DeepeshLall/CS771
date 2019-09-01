import numpy as np
import random as rnd
import time as tm

C = 1
# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def getCyclicCoord( currentCoord, n ):
	if currentCoord >= n-1 or currentCoord < 0:
		return 0
	else:
		return currentCoord + 1
def getRandpermCoord( currentCoord, y,n):
	randperm = np.random.permutation( y.size )
	randpermInner = -1
	if randpermInner >= n-1 or randpermInner < 0 or currentCoord < 0:
		randpermInner = 0
		randperm = np.random.permutation( y.size )
		return randperm[randpermInner]
	else:
		randpermInner = randpermInner + 1
		return randperm[randpermInner]
def getObj( X, y, w, b ):
	hingeLoss = np.maximum( np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )
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
	horizon = 10
	dualObjValSeries = np.zeros((horizon,))
	timeSeries = np.zeros( (horizon,) )
	alpha =  np.ones((n,))
	alphay = np.multiply(alpha,y)
	w = X.T.dot(alphay)
	b = alpha.dot(y)
	# Norm = q
	normSq = np.square( np.linalg.norm( X, axis = 1 ) ) + 1
	# You may also define new variables here e.g. eta, B etc
	i = -1
	randperm = np.random.permutation( y.size )
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
		print(getObj(X,y,w,b))
		# Write all code to perform your method updates here within the infinite while loop
		# The infinite loop will terminate once timeout is reached
		# Do not try to bypass the timer check e.g. by using continue
		# It is very easy for us to detect such bypasses - severe penalties await
		# i = getRandpermCoord(i,y,n)
		i = getCyclicCoord(i,n)
		x = X[i,:]
		newAlphai = (1 - (y[i]*(x.dot(w)+b)) + (alpha[i] * normSq[i]))/( (0.5/C) + (normSq[i]) )
		if(newAlphai < 0):
			newAlphai = 0
		w = w + (newAlphai - alpha[i]) * y[i] * x
		b = b + (newAlphai - alpha[i]) * y[i]
		alpha[i] = newAlphai 
		if t < horizon:
			dualObjValSeries[t] = getObj(X,y,w,b)
			timeSeries[t] = totTime
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
