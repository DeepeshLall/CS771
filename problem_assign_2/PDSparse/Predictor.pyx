# distutils: language = c++

from Predictor cimport PDSparse
from libcpp.vector cimport vector

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.

cdef class Predictor:
	cdef PDSparse c_pdsparse  # Hold a C++ instance which we're wrapping

	def __cinit__(self):
		self.c_pdsparse = PDSparse()

	def run(self, vector[vector[int]] X1, vector[vector[float]] X2, int T):
		return self.c_pdsparse.run(X1, X2, T)