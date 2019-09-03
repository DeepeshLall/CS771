import numpy as np
from submit import solver

def getObj( X, y, w, b ):
	hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

def getAccuracy (X, y, w, b):
	#returns (false positive, true positives, false negatives, true negatives)
	I = np.ones((X.shape[0],))
	y_pred = X.dot(w) + b * I
	fp, tp, fn, tn = 0, 0, 0, 0
	for i in range(0, X.shape[0]):
		if y_pred[i] > 0 and y[i] > 0:
			tp += 1
		elif y_pred[i] > 0 and y[i] < 0:
			fp += 1
		elif y_pred[i] < 0 and y[i] < 0:
			tn += 1
		else:
			fn += 1
	#return (fp, tp, fn, tn)
	return (tp + tn) / (tp + tn + fp + fn)

Z = np.loadtxt( "data" )
print(Z.shape)
#y = Z[:,0]
#X = Z[:,1:]
i1, i2, i3 = 11000, 14000, 17000
y = Z[:i1,0]
X = Z[:i1,1:]
y_val1 = Z[i1+1:i2,0]
y_val2 = Z[i2+1:i3,0]
y_val3 = Z[i3+1:,0]
X_val1 = Z[i1+1:i2,1:]
X_val2 = Z[i2+1:i3,1:]
X_val3 = Z[i3+1:,1:]
C = 1

avgTime = 0
avgPerf0 = 0
avgPerf1 = 0
avgPerf2 = 0
avgPerf3 = 0

# To avoid unlucky outcomes try running the code several times
numTrials = 1
# 30 second timeout for each run
timeout = 30
# Try checking for timeout every 100 iterations
spacing = 100


for t in range( numTrials ):
	#solve the problem on the training data
	(w, b, totTime, timeseries, objseries) = solver( X, y, C, timeout, spacing )
	avgTime = avgTime + totTime
	avgPerf0 = avgPerf1 + getObj( X, y, w, b )
	avgPerf1 = avgPerf1 + getObj( X_val1, y_val1, w, b )
	avgPerf2 = avgPerf2 + getObj( X_val2, y_val2, w, b )
	avgPerf3 = avgPerf3 + getObj( X_val3, y_val3, w, b )
	#print(getAccuracy(X, y, w, b))
	#print(getAccuracy(X_val1, y_val1, w, b))
	#print(getAccuracy(X_val2, y_val2, w, b))
	#print(getAccuracy(X_val2, y_val2, w, b))

print( avgPerf0/numTrials )
print( avgPerf1/numTrials )
print( avgPerf2/numTrials )
print( avgPerf3/numTrials )
print( avgTime/numTrials )

