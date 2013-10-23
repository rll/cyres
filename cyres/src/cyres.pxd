cimport ceres

cdef class CostFunction:
    cdef ceres.CostFunction* _cost_function

    cpdef parameter_block_sizes(self)
    cpdef num_residuals(self)

cdef class LossFunction:
    cdef ceres.LossFunction* _loss_function
