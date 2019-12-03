import numpy as np
import random as rnd
import time as tm
from math import sqrt

# You may define any new functions, variables, classes here
# For example, functions to calculate next coordinate or step length

def getCyclicCoord(i, n):
	j = (i + 1) % n
	if (j == 0):
		return 1
	return j

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

	#initialize alpha
	alpha = np.zeros( (n,) )

	#increase the dimension of w by 1 to account for the bias term
	w = np.zeros( (d+1,) )
	for i in range(0, n):
		w += alpha[i] * y[i] * X[i][:]
	
	eta0 = 1.0
	i = 1

################################
# Non Editable Region Starting #
################################
	while True:
		t = t + 1
		if t % spacing == 0:
			toc = tm.perf_counter()
			totTime = totTime + (toc - tic)
			if totTime > timeout:
				return (w[1:], w[0], totTime)
			else:
				tic = tm.perf_counter()
################################
#  Non Editable Region Ending  #
################################

		#doing cyclic dual coordinate ascent

		step_length = eta0 / sqrt(t)
		i = getCyclicCoord(i, n)

		alphai = alpha[i]
		grad = 1 - alphai/(2*C) - y[i] * np.dot(w, X[i][:])
		new_alphai = alphai + step_length * grad
		if (new_alphai < 0):
			new_alphai = 0
		w += (new_alphai - alphai) * y[i] * X[i][:]
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