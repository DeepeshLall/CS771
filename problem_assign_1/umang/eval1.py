import numpy as np
from submit import solver
import sys

def getObj( X, y, w, b ):
    hingeLoss = np.maximum( 1 - np.multiply( (X.dot( w ) + b), y ), 0 )
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

for t in range( numTrials ):
    (w, b, totTime) = solver( X, y, C, timeout, spacing )

    Z = np.loadtxt( "test" )
    y1 = Z[:,0]
    X1 = Z[:,1:]
    (n1,d1) = X1.shape
    pos = 0
    totalpos=0;totalneg=0;actualpos=0;actualneg=0;
    for i in range(n1):
        cat = (1 if (X1[i].dot(w) + b >= 0) else -1)
        pos += (1 if cat == y1[i] else 0)
        # print(X1[i].dot(w) + b, file=sys.stderr)
        if(X1[i].dot(w) + b >= 0):
            actualpos += 1
        else:
            actualneg += 1
        if y[i] == 1:
            totalpos += 1
        else:
            totalneg += 1

    print("objective = ", getObj(X,y,w,b))
    print("obj on validation = ", getObj(X1,y1,w,b))
    print("accuracy= ", pos/n1)
    print(totalpos," = totalpos and totalneg = ",totalneg)
    print(actualpos," = actualpos and actualneg = ",actualneg)

    avgTime = avgTime + totTime
    avgPerf = avgPerf + getObj( X, y, w, b )



print( avgPerf/numTrials, avgTime/numTrials )

