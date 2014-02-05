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

cdef class SquaredLoss(LossFunction):
    def __cinit__(self):
        _loss_function = NULL

cdef class HuberLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.HuberLoss(_a)

cdef class SoftLOneLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.SoftLOneLoss(_a)

cdef class CauchyLoss(LossFunction):
    def __init__(self, double _a):
        self._loss_function = new ceres.CauchyLoss(_a)

cdef class ArctanLoss(LossFunction):
    def __init__(self, double _a):
        """
        Loss that is capped beyond a certain level using the arc-tangent
        function. The scaling parameter 'a' determines the level where falloff
        occurs. For costs much smaller than 'a', the loss function is linear
        and behaves like TrivialLoss, and for values much larger than 'a' the
        value asymptotically approaches the constant value of a * PI / 2.

          rho(s) = a atan(s / a).

        At s = 0: rho = [0, 1, 0].
        """
        self._loss_function = new ceres.ArctanLoss(_a)

cdef class TolerantLoss(LossFunction):
    """
    Loss function that maps to approximately zero cost in a range around the
    origin, and reverts to linear in error (quadratic in cost) beyond this
    range. The tolerance parameter 'a' sets the nominal point at which the
    transition occurs, and the transition size parameter 'b' sets the nominal
    distance over which most of the transition occurs. Both a and b must be
    greater than zero, and typically b will be set to a fraction of a. The
    slope rho'[s] varies smoothly from about 0 at s <= a - b to about 1 at s >=
    a + b.

    The term is computed as:

      rho(s) = b log(1 + exp((s - a) / b)) - c0.

    where c0 is chosen so that rho(0) == 0

      c0 = b log(1 + exp(-a / b)

    This has the following useful properties:

      rho(s) == 0               for s = 0
      rho'(s) ~= 0              for s << a - b
      rho'(s) ~= 1              for s >> a + b
      rho''(s) > 0              for all s

    In addition, all derivatives are continuous, and the curvature is
    concentrated in the range a - b to a + b.

    At s = 0: rho = [0, ~0, ~0].
    """
    def __init__(self, double _a, double _b):
        self._loss_function = new ceres.TolerantLoss(_a, _b)

cdef class ComposedLoss(LossFunction):

    def __init__(self, LossFunction f, LossFunction g):
        self._loss_function = new ceres.ComposedLoss(f._loss_function,
                                                     ceres.DO_NOT_TAKE_OWNERSHIP,
                                                     g._loss_function,
                                                     ceres.DO_NOT_TAKE_OWNERSHIP)

cdef class ScaledLoss(LossFunction):

    def __init__(self, LossFunction loss_function, double _a):
        self._loss_function = new ceres.ScaledLoss(loss_function._loss_function,
                                                   _a,
                                                   ceres.DO_NOT_TAKE_OWNERSHIP)

cdef class Summary:
    cdef ceres.Summary _summary

    def briefReport(self):
        return self._summary.BriefReport()

cdef class EvaluateOptions:
    cdef ceres.EvaluateOptions _options

    def __cinit__(self):
        pass

    def __init__(self):
        self._options = ceres.EvaluateOptions()

    property residual_blocks:
        def __get__(self):
            blocks = []
            cdef int i
            for i in range(self._options.residual_blocks.size()):
                block = ResidualBlockId()
                block._block_id = self._options.residual_blocks[i]
                blocks.append(block)
            return blocks
        def __set__(self, blocks):
            self._options.residual_blocks.clear()
            cdef ResidualBlockId block
            for block in blocks:
                self._options.residual_blocks.push_back(block._block_id)

    property apply_loss_function:
        def __get__(self):
            return self._options.apply_loss_function
        def __set__(self, value):
            self._options.apply_loss_function = value

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

    property num_threads:
        def __get__(self):
            return self._options.num_threads

        def __set__(self, value):
            self._options.num_threads = value

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

        cdef ceres.ResidualBlockId _block_id

        for parameter_block in parameter_blocks:
            _tmp_array = np.ascontiguousarray(parameter_block, dtype=np.double)
            _parameter_blocks.push_back(<double*> _tmp_array.data)
        _block_id = self._problem.AddResidualBlock(cost_function._cost_function,
                                                   loss_function._loss_function,
                                                   _parameter_blocks)
        block_id = ResidualBlockId()
        block_id._block_id = _block_id
        return block_id

    cpdef evaluate(self, residual_blocks, apply_loss_function=True):

        cdef double cost

        options = EvaluateOptions()
        options.apply_loss_function = apply_loss_function
        options.residual_blocks = residual_blocks

        self._problem.Evaluate(options._options, &cost, NULL, NULL, NULL)
        return cost

cdef class ResidualBlockId:
    cdef ceres.ResidualBlockId _block_id

def solve(SolverOptions options, Problem problem, Summary summary):
    ceres.Solve(drf(options._options), &problem._problem, &summary._summary)
