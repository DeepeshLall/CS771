# distutils: sources = PDSparse.cpp

from libcpp.vector cimport vector

cdef extern from "PDSparse.cpp":
	pass

cdef extern from "PDSparse.h":
	cdef cppclass PDSparse:
		PDSparse() except +
		vector[vector[int]] run(vector[vector[int]], vector[vector[float]], int)