import numpy as np
from submit_ayush import solver as solver1
from matplotlib import pyplot as plt

def getObj( X, y, w, b ):
	hingeLoss = np.maximum( np.multiply( (X.dot( w ) + b), y ), 0 )
	return 0.5 * w.dot( w ) + C * hingeLoss.dot( hingeLoss )

Z = np.loadtxt( "data" )

y = Z[:,0]
X = Z[:,1:]
C = 1

avgTime = 0
avgPerf = 0

# To avoid unlucky outcomes try running the code several times
numTrials = 1
# 30 second timeout for each run
timeout = 30
# Try checking for timeout every 100 iterations
spacing = 100

global timeSeries1
global ObjSeries1

for t in range( numTrials ):
	(w, b, totTime, timeSeries1, ObjSeries1) = solver1( X, y, C, timeout, spacing )
	print(getObj(X,y,w,b))
	avgTime = avgTime + totTime
	avgPerf = avgPerf + getObj( X, y, w, b )

plt.plot(timeSeries1, ObjSeries1)
plt.show()

# for t in range( numTrials ):
# 	(w, b, totTime) = solver2( X, y, C, timeout, spacing )
# 	print(getObj(X,y,w,b))
# 	avgTime = avgTime + totTime
# 	avgPerf = avgPerf + getObj( X, y, w, b )

# for t in range( numTrials ):
# 	(w, b, totTime) = solver3( X, y, C, timeout, spacing )
# 	print(getObj(X,y,w,b))
# 	avgTime = avgTime + totTime
# 	avgPerf = avgPerf + getObj( X, y, w, b )

# print( avgPerf/numTrials, avgTime/numTrials )
