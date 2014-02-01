from cyres cimport CostFunction, LossFunction
cimport ceres
cimport numpy as np
import numpy as np

cdef extern from "cost_functions.h":
    ceres.CostFunction* createCostFunction()

cdef class SimpleCostFunction(CostFunction):

    def __cinit__(self):
        self._cost_function = createCostFunction()
