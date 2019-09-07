import numpy as np
from submit import solver as solver_ay
from submit_adadelta import solver as solver_um
import matplotlib.pyplot as plt

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
y0 = Z[:,0]
X0 = Z[:,1:]
i1, i2, i3 = 11000, 14000, 17000
y = Z[:i1,0]
X = Z[:i1,1:]
y_val1 = Z[i1+1:i2,0]
y_val2 = Z[i2+1:i3,0]
y_val3 = Z[i3+1:,0]
X_val1 = Z[i1+1:i2,1:]
X_val2 = Z[i2+1:i3,1:]
X_val3 = Z[i3+1:,1:]
C = 10000

avgTime = 0
avgPerf0 = 0
avgPerf1 = 0
avgPerf2 = 0
avgPerf3 = 0
avgPerf4 = 0

# To avoid unlucky outcomes try running the code several times
numTrials = 1
# 30 second timeout for each run, 15, 20, 25, 30 , 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2. 2.4
#timeouts = [30]
timeout = 30
#etas = 1.0
#etas = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
#table = np.zeros((len(etas), len(timeouts)))
# Try checking for timeout every 100 iterations
spacing = 100


'''
for j in range(len(timeouts)):
	for k in range(numTrials):
		(w, b, totTime) = solver( X0, y0, C, timeouts[j], spacing )
		table[i][j] += getObj(X0, y0, w, b)
	table[i][j] /= numTrials

	#print(table[i][j])

print(table)
'''
#for t in range( numTrials ):
	#solve the problem on the training data
(w, b, totTime, timeseries_ay, objseries_ay) = solver_ay( X0, y0, C, timeout, spacing )
(w, b, totTime, timeseries_um, objseries_um) = solver_um( X0, y0, C, timeout, spacing )
plt.plot(timeseries_ay, objseries_ay)
plt.plot(timeseries_um, objseries_um)
plt.show()
#	avgTime = avgTime + totTime
#	avgPerf0 += getObj( X, y, w, b )
#	avgPerf1 += getObj( X_val1, y_val1, w, b )
#	avgPerf2 += getObj( X_val2, y_val2, w, b )
#	avgPerf3 += getObj( X_val3, y_val3, w, b )
#	avgPerf4 += getObj( X0, y0, w, b)
#	print(getAccuracy(X, y, w, b))
#	print(getAccuracy(X_val1, y_val1, w, b))
#	print(getAccuracy(X_val2, y_val2, w, b))
#	print(getAccuracy(X_val2, y_val2, w, b))


#print( avgPerf0/numTrials )
#print( avgPerf1/numTrials )
#print( avgPerf2/numTrials )
#print( avgPerf3/numTrials )
#print( avgPerf4/numTrials)
#print( avgTime/numTrials )
