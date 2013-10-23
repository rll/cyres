from cyres cimport CostFunction, LossFunction
cimport ceres
cimport numpy as np
import numpy as np

cdef extern from "cost_functions.h":
    ceres.CostFunction* createCostFunction()

cdef class SimpleCostFunction(CostFunction):

    def __cinit__(self):
        self._cost_function = createCostFunction()

    #def evaluate(self, x):
    #    cdef np.ndarray _x_tmp = np.ascontiguousarray(x, dtype=np.double)
    #    cdef np.ndarray _residual = np.zeros((1,1))
    #    cdef np.ndarray _jacobian = np.zeros((1,1))

    #    cdef double** _x_ptr = <double**> &(_x_tmp.data)
    #    cdef double* _res_ptr = <double*> _residual.data
    #    cdef double** _jac_ptr = <double**> &(_jacobian.data)
    #    self._cost_function.Evaluate(_x_ptr, _res_ptr, _jac_ptr)
    #    return _residual
