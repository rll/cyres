#cython: boundscheck=False, wraparound=False

import cython
from libcpp.vector cimport vector
from libc.stdlib cimport malloc, free
from cython.operator cimport dereference as drf

cimport numpy as np

import numpy as np

cimport ceres
from cyres cimport *

def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    reverse = dict((value, key) for key, value in enums.iteritems())
    enums['reverse_mapping'] = reverse
    return type('Enum', (), enums)

Ownership = enum("DO_NOT_TAKE_OWNERSHIP", "TAKE_OWNERSHIP")
MinimizerType = enum("LINE_SEARCH", "TRUST_REGION")
LinearSolverType = enum("DENSE_NORMAL_CHOLESKY", "DENSE_QR",
                        "SPARSE_NORMAL_CHOLESKY", "DENSE_SCHUR", "SPARSE_SCHUR",
                        "ITERATIVE_SCHUR", "CGNR")
PreconditionerType = enum("IDENTITY", "JACOBI", "SCHUR_JACOBI",
                          "CLUSTER_JACOBI", "CLUSTER_TRIDIAGONAL")
SparseLinearAlgebraLibraryType = enum("SUITE_SPARSE", "CX_SPARSE")
LinearSolverTerminationType = enum("TOLERANCE", "MAX_ITERATIONS", "STAGNATION",
                                   "FAILURE")
LoggingType = enum("SILENT", "PER_MINIMIZER_ITERATION")
LineSearchDirectionType = enum("STEEPEST_DESCENT",
                               "NONLINEAR_CONJUGATE_GRADIENT",
                               "LBFGS")
NonlinearConjugateGradientType = enum("FLETCHER_REEVES", "POLAK_RIBIRERE",
                                      "HESTENES_STIEFEL")
LineSearchType = enum("ARMIJO")
TrustRegionStrategyType = enum("LEVENBERG_MARQUARDT", "DOGLEG")
DoglegType = enum("TRADITIONAL_DOGLEG", "SUBSPACE_DOGLEG")
SolverTerminationType = enum("DID_NOT_RUN", "NO_CONVERGENCE", "FUNCTION_TOLERANCE", "GRADIENT_TOLERANCE", "PARAMETER_TOLERANCE", "NUMERICAL_FAILURE", "USER_ABORT", "USER_SUCCESS")
CallbackReturnType = enum("SOLVER_CONTINUE", "SOLVER_ABORT", "SOLVER_TERMINATE_SUCCESSFULLY")
DumpFormatType = enum("CONSOLE", "PROTOBUF", "TEXTFILE")
DimensionType = enum(DYNAMIC=-1)
NumericDiffMethod = enum("CENTRAL", "FORWARD")

cdef class CostFunction:

    cpdef parameter_block_sizes(self):
        block_sizes = []
        cdef vector[ceres.int16] _parameter_block_sizes = self._cost_function.parameter_block_sizes()
        for i in range(_parameter_block_sizes.size()):
            block_sizes.append(_parameter_block_sizes[i])
        return block_sizes

    cpdef num_residuals(self):
        return self._cost_function.num_residuals()

    def evaluate(self, *param_blocks, **kwargs):

        include_jacobians = kwargs.get("include_jacobians", False)

        cdef double** _params_ptr = NULL
        cdef double* _residuals_ptr = NULL
        cdef double** _jacobians_ptr = NULL

        block_sizes = self.parameter_block_sizes()

        _params_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))

        cdef np.ndarray[np.double_t, ndim=1] _param_block

        for i, param_block in enumerate(param_blocks):
            if block_sizes[i] != len(param_block):
                raise Exception("Expected param block of size %d, got %d" % (block_sizes[i], len(param_block)))
            _param_block = param_block
            _params_ptr[i] = <double*> _param_block.data

        cdef np.ndarray[np.double_t, ndim=1] residuals

        residuals = np.empty((self.num_residuals()), dtype=np.double)
        _residuals_ptr = <double*> residuals.data

        cdef np.ndarray[np.double_t, ndim=2] _jacobian
        if include_jacobians:
            # jacobians is an array of size CostFunction::parameter_block_sizes_
            # containing pointers to storage for Jacobian matrices corresponding
            # to each parameter block. The Jacobian matrices are in the same
            # order as CostFunction::parameter_block_sizes_. jacobians[i] is an
            # array that contains CostFunction::num_residuals_ x
            # CostFunction::parameter_block_sizes_[i] elements. Each Jacobian
            # matrix is stored in row-major order, i.e., jacobians[i][r *
            # parameter_block_size_[i] + c]
            jacobians = []
            _jacobians_ptr = <double**> malloc(sizeof(double*)*len(block_sizes))
            for i, block_size in enumerate(block_sizes):
                jacobian = np.empty((self.num_residuals(), block_size), dtype=np.double)
                jacobians.append(jacobian)
                _jacobian = jacobian
                _jacobians_ptr[i] = <double*> _jacobian.data

        self._cost_function.Evaluate(_params_ptr, _residuals_ptr, _jacobians_ptr)

        free(_params_ptr)

        if include_jacobians:
            free(_jacobians_ptr)
            return residuals, jacobians
        else:
            return residuals

cdef class SquaredLossFunction(LossFunction):
    def __cinit__(self):
        _loss_function = NULL

cdef class Summary:
    cdef ceres.Summary _summary

    def briefReport(self):
        return self._summary.BriefReport()

cdef class SolverOptions:
    cdef ceres.SolverOptions* _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = new ceres.SolverOptions()

    property max_num_iterations:
        def __get__(self):
            return self._options.max_num_iterations

        def __set__(self, value):
            self._options.max_num_iterations = value

    property minimizer_progress_to_stdout:
        def __get__(self):
            return self._options.minimizer_progress_to_stdout

        def __set__(self, value):
            self._options.minimizer_progress_to_stdout = value

    property linear_solver_type:
        def __get__(self):
            return self._options.linear_solver_type

        def __set__(self, value):
            self._options.linear_solver_type = value

cdef class Problem:
    cdef ceres.Problem _problem

    def __cinit__(self):
        pass

    # loss_function=NULL yields squared loss
    cpdef add_residual_block(self,
                           CostFunction cost_function,
                           LossFunction loss_function,
                           parameter_blocks=[]):

        cdef np.ndarray _tmp_array
        cdef vector[double*] _parameter_blocks
        cdef double f

        for parameter_block in parameter_blocks:
            _tmp_array = np.ascontiguousarray(parameter_block, dtype=np.double)
            _parameter_blocks.push_back(<double*> _tmp_array.data)
        self._problem.AddResidualBlock(cost_function._cost_function, loss_function._loss_function, _parameter_blocks)

def solve(SolverOptions options, Problem problem, Summary summary):
    ceres.Solve(drf(options._options), &problem._problem, &summary._summary)
