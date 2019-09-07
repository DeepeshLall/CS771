import numpy as np
import random as rnd
import time as tm
import matplotlib.pyplot as plt
import math

def getCyclicCoord(i, n):
	j = (i + 1) % n
	if (j == 0):
		return 1
	return j

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length
def getObj( X, y, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + 1 * hingeLoss.dot( hingeLoss )

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

	#add the bias to the dataset, i.e. a column of ones before the 1st column
	X = np.insert(X, 0, 1, axis=1)

	#randomly initialize alpha
	#alpha = np.ones( (n,) )
	alpha = np.zeros( (n,) )
	for i in range(0, n):
		alpha[i] = 0.00

	#increase the dimension of w by 1 to account for the bias term
	w_run = np.zeros( (d+1,) )
	w = np.zeros( (d+1,) )
	for i in range(0, n):
		w_run += alpha[i] * y[i] * X[i][:]
	eta0 = 1.0
	i = 1
	var_min = 0.1
	var = var_min+1
	obj_min = 100000000000000000
	#converged = False

	objseries = []
	timeseries = []

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			#print(objseries[-1])
			#print(getObj(X[:,1:], y, w_run[1:], w_run[0]))
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				#plt.plot(timeseries, objseries)
				#plt.show()
				return (w_run[1:], w_run[0], totTime, timeseries, objseries)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################
		#print("doing scd")
		#doing stochastic coordinate descent
		timeseries.append(t)
		objseries.append(getObj(X[:,1:], y, w_run[1:], w_run[0]))
		#print(objseries[-1])
		#if (objseries[-1] < obj_min):
		#	obj_min = objseries[-1]
		#	w = w_run
		#check for convergence
		#if len(objseries) > 10:
		#	var = np.var(objseries[-10:])
		#if var < var_min:
		#	converged = True

		step_length = eta0 / math.sqrt(t)
		i = getCyclicCoord(i, n)
		#i = np.random.randint(0, n)

		alphai = alpha[i]
		grad = 1 - alphai/(2*C) - y[i] * np.dot(w_run, X[i][:])
		new_alphai = alphai + step_length * grad
		if (new_alphai < 0):
			new_alphai = 0
		w_run += (new_alphai - alphai) * y[i] * X[i][:]
		#only the ith coordinate of alpha will change
		alpha[i] = new_alphai
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