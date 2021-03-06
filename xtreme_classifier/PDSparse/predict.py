#predict.py
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import dump_svmlight_file
from scipy import sparse as sps
import numpy as np
from numpy import random as rand
import subprocess
import time
import Predictor

def load_data(filename, d, L):
	X, _ = load_svmlight_file( "%s.X" % filename, multilabel = True, n_features = d, offset = 1 )
	#y, _ = load_svmlight_file( "%s.y" % filename, multilabel = True, n_features = L, offset = 1 )
	return X

def dump_data(X, filename):
	(n, d) = X.shape
	#(n1, L) = y.shape
	#assert n1 == n, "Mismatch in number of feature vectors and number of label vectors"
	dummy = sps.csr_matrix( (n, 1) )
	dump_svmlight_file( X, dummy, "%s.X" % filename, multilabel = True, zero_based = True, comment = "%d, %d" % (n, d) )
	#dump_svmlight_file( y, dummy, "%s.y" % filename, multilabel = True, zero_based = True, comment = "%d, %d" % (n, L) )

def getReco(X, k):
	#X = load_data("data", 16385, 3400)
	#print(myX)
	#mmwrite(Xfile, X)
	#create two list of lists [[int]], [[float]]
	cx = sps.coo_matrix(X)

	#print(time.time())
	iprev = -1
	X1, X2 = [], []
	for i,j,v in zip(cx.row, cx.col, cx.data):
		if i != iprev:
			#row changed
			X1.append([])
			X2.append([])
			iprev = i
		X1[i].append(j)
		X2[i].append(v)

	mypdsparse = Predictor.Predictor()
	ypred = mypdsparse.run(X1, X2, k)
	#n = X.shape[0]
	#print(time.time())
	#dump_data(X, "mydata")
	#print(time.time())
	return np.array(ypred)
	#subprocess.run(["./multiPred", "mydata.X", "model.txt", "-p", str(k), "rec_labels.y", str(k)])
	#recy = []
	'''
	recy = np.zeros( (n, k) )
	with open("rec_labels.y", 'r') as f:
		for line in f.readlines():
			recy.append([int(i) for i in line.split(' ')])
	'''
	#recy = np.loadtxt("rec_labels.y");
	#(_, recy) = load_data("rec_labels", -1, k)
	#print(recy)
	#return recy

#getReco(5)
