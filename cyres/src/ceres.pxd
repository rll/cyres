from libcpp cimport bool
from libcpp.vector cimport vector
from libcpp.string cimport string
from libc.stdint cimport uint8_t, uint16_t, uint32_t, uint64_t

cdef extern from "types.h" namespace "ceres":
    ctypedef short int16
    ctypedef int int32
    ctypedef enum Ownership:
        DO_NOT_TAKE_OWNERSHIP
        TAKE_OWNERSHIP

    ctypedef enum MinimizerType:
        LINE_SEARCH
        TRUST_REGION

    # TODO(keir): Considerably expand the explanations of each solver type.
    ctypedef enum LinearSolverType:
      # These solvers are for general rectangular systems formed from the
      # normal equations A'A x = A'b. They are direct solvers and do not
      # assume any special problem structure.

      # Solve the normal equations using a dense Cholesky solver; based
      # on Eigen.
      DENSE_NORMAL_CHOLESKY

      # Solve the normal equations using a dense QR solver; based on
      # Eigen.
      DENSE_QR

      # Solve the normal equations using a sparse cholesky solver; requires
      # SuiteSparse or CXSparse.
      SPARSE_NORMAL_CHOLESKY

      # Specialized solvers specific to problems with a generalized
      # bi-partitite structure.

      # Solves the reduced linear system using a dense Cholesky solver;
      # based on Eigen.
      DENSE_SCHUR

      # Solves the reduced linear system using a sparse Cholesky solver;
      # based on CHOLMOD.
      SPARSE_SCHUR

      # Solves the reduced linear system using Conjugate Gradients based
      # on a new Ceres implementation.  Suitable for large scale
      # problems.
      ITERATIVE_SCHUR

      # Conjugate gradients on the normal equations.
      CGNR

    ctypedef enum PreconditionerType:
      # Trivial preconditioner - the identity matrix.
      IDENTITY

      # Block diagonal of the Gauss-Newton Hessian.
      JACOBI

      # Block diagonal of the Schur complement. This preconditioner may
      # only be used with the ITERATIVE_SCHUR solver.
      SCHUR_JACOBI

      # Visibility clustering based preconditioners.
      #
      # These preconditioners are well suited for Structure from Motion
      # problems particularly problems arising from community photo
      # collections. These preconditioners use the visibility structure
      # of the scene to determine the sparsity structure of the
      # preconditioner. Requires SuiteSparse/CHOLMOD.
      CLUSTER_JACOBI
      CLUSTER_TRIDIAGONAL

    ctypedef enum SparseLinearAlgebraLibraryType:
      # High performance sparse Cholesky factorization and approximate
      # minimum degree ordering.
      SUITE_SPARSE

      # A lightweight replacment for SuiteSparse.
      CX_SPARSE

    ctypedef enum LinearSolverTerminationType:
      # Termination criterion was met. For factorization based solvers
      # the tolerance is assumed to be zero. Any user provided values are
      # ignored.
      TOLERANCE

      # Solver ran for max_num_iterations and terminated before the
      # termination tolerance could be satified.
      MAX_ITERATIONS

      # Solver is stuck and further iterations will not result in any
      # measurable progress.
      STAGNATION

      # Solver failed. Solver was terminated due to numerical errors. The
      # exact cause of failure depends on the particular solver being
      # used.
      FAILURE

    # Logging options
    # The options get progressively noisier.
    ctypedef enum LoggingType:
      SILENT
      PER_MINIMIZER_ITERATION

    ctypedef enum LineSearchDirectionType:
        STEEPEST_DESCENT

        # A generalization of the Conjugate Gradient method to non-linear
        # functions. The generalization can be performed in a number of
        # different ways, resulting in a variety of search directions. The
        # precise choice of the non-linear conjugate gradient algorithm
        # used is determined by NonlinerConjuateGradientType.
        NONLINEAR_CONJUGATE_GRADIENT

        # A limited memory approximation to the inverse Hessian is
        # maintained and used to compute a quasi-Newton step.
        #
        # For more details see
        #
        # Nocedal, J. (1980). "Updating Quasi-Newton Matrices with Limited
        # Storage". Mathematics of Computation 35 (151): 773–782.
        #
        # Byrd, R. H.; Nocedal, J.; Schnabel, R. B. (1994).
        # "Representations of Quasi-Newton Matrices and their use in
        # Limited Memory Methods". Mathematical Programming 63 (4):
        # 129–156.
        LBFGS

    # Nonliner conjugate gradient methods are a generalization of the
    # method of Conjugate Gradients for linear systems. The
    # generalization can be carried out in a number of different ways
    # leading to number of different rules for computing the search
    # direction. Ceres provides a number of different variants. For more
    # details see Numerical Optimization by Nocedal & Wright.
    ctypedef enum NonlinearConjugateGradientType:
        FLETCHER_REEVES
        POLAK_RIBIRERE
        HESTENES_STIEFEL

    ctypedef enum LineSearchType:
        # Backtracking line search with polynomial interpolation or
        # bisection.
        ARMIJO

    # Ceres supports different strategies for computing the trust region
    # step.
    ctypedef enum TrustRegionStrategyType:
        # The default trust region strategy is to use the step computation
        # used in the Levenberg-Marquardt algorithm. For more details see
        # levenberg_marquardt_strategy.h
        LEVENBERG_MARQUARDT

        # Powell's dogleg algorithm interpolates between the Cauchy point
        # and the Gauss-Newton step. It is particularly useful if the
        # LEVENBERG_MARQUARDT algorithm is making a large number of
        # unsuccessful steps. For more details see dogleg_strategy.h.
        #
        # NOTES:
        #
        # 1. This strategy has not been experimented with or tested as
        # extensively as LEVENBERG_MARQUARDT and therefore it should be
        # considered EXPERIMENTAL for now.
        #
        # 2. For now this strategy should only be used with exact
        # factorization based linear solvers i.e. SPARSE_SCHUR
        # DENSE_SCHUR DENSE_QR and SPARSE_NORMAL_CHOLESKY.
        DOGLEG

    # Ceres supports two different dogleg strategies.
    # The "traditional" dogleg method by Powell and the
    # "subspace" method described in
    # R. H. Byrd, R. B. Schnabel, and G. A. Shultz,
    # "Approximate solution of the trust region problem by minimization
    #  over two-dimensional subspaces", Mathematical Programming,
    # 40 (1988), pp. 247--263
    ctypedef enum DoglegType:
        # The traditional approach constructs a dogleg path
        # consisting of two line segments and finds the furthest
        # point on that path that is still inside the trust region.
        TRADITIONAL_DOGLEG

        # The subspace approach finds the exact minimum of the model
        # constrained to the subspace spanned by the dogleg path.
        SUBSPACE_DOGLEG

    ctypedef enum SolverTerminationType:
        # The minimizer did not run at all; usually due to errors in the user's
        # Problem or the solver options.
        DID_NOT_RUN

        # The solver ran for maximum number of iterations specified by the
        # user but none of the convergence criterion specified by the user
        # were met.
        NO_CONVERGENCE

        # Minimizer terminated because
        #  (new_cost - old_cost) < function_tolerance * old_cost;
        FUNCTION_TOLERANCE

        # Minimizer terminated because
        # max_i |gradient_i| < gradient_tolerance * max_i|initial_gradient_i|
        GRADIENT_TOLERANCE

        # Minimized terminated because
        #  |step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)
        PARAMETER_TOLERANCE

        # The minimizer terminated because it encountered a numerical error
        # that it could not recover from.
        NUMERICAL_FAILURE

        # Using an IterationCallback object user code can control the
        # minimizer. The following enums indicate that the user code was
        # responsible for termination.

        # User's IterationCallback returned SOLVER_ABORT.
        USER_ABORT

        # User's IterationCallback returned SOLVER_TERMINATE_SUCCESSFULLY
        USER_SUCCESS

    # Enums used by the IterationCallback instances to indicate to the
    # solver whether it should continue solving, the user detected an
    # error or the solution is good enough and the solver should
    # terminate.
    ctypedef enum CallbackReturnType:
        # Continue solving to next iteration.
        SOLVER_CONTINUE

        # Terminate solver and do not update the parameter blocks upon
        # return. Unless the user has set
        # Solver:Options:::update_state_every_iteration in which case the
        # state would have been updated every iteration
        # anyways. Solver::Summary::termination_type is set to USER_ABORT.
        SOLVER_ABORT

        # Terminate solver update state and
        # return. Solver::Summary::termination_type is set to USER_SUCCESS.
        SOLVER_TERMINATE_SUCCESSFULLY

    # The format in which linear least squares problems should be logged
    # when Solver::Options::lsqp_iterations_to_dump is non-empty.
    ctypedef enum DumpFormatType:
        # Print the linear least squares problem in a human readable format
        # to stderr. The Jacobian is printed as a dense matrix. The vectors
        # D x and f are printed as dense vectors. This should only be used
        # for small problems.
        CONSOLE

        # Write out the linear least squares problem to the directory
        # pointed to by Solver::Options::lsqp_dump_directory as a protocol
        # buffer. linear_least_squares_problems.h/cc contains routines for
        # loading these problems. For details on the on disk format used
        # see matrix.proto. The files are named lm_iteration_???.lsqp.
        PROTOBUF

        # Write out the linear least squares problem to the directory
        # pointed to by Solver::Options::lsqp_dump_directory as text files
        # which can be read into MATLAB/Octave. The Jacobian is dumped as a
        # text file containing (ijs) triplets the vectors D x and f are
        # dumped as text files containing a list of their values.
        #
        # A MATLAB/octave script called lm_iteration_???.m is also output
        # which can be used to parse and load the problem into memory.
        TEXTFILE

    # For SizedCostFunction and AutoDiffCostFunction, DYNAMIC can be specified for
    # the number of residuals. If specified, then the number of residuas for that
    # cost function can vary at runtime.
    ctypedef enum DimensionType:
        DYNAMIC = -1

    ctypedef enum NumericDiffMethod:
        CENTRAL
        FORWARD


cdef extern from "ordered_groups.h" namespace "ceres":
    cdef cppclass OrderedGroups[T]:
        pass
    ctypedef OrderedGroups[double*] ParameterBlockOrdering

cdef extern from "iteration_callback.h" namespace "ceres":
    cdef struct IterationSummary:
        IterationSummary()

        # Current iteration number.
        int32 iteration

        # Step was numerically valid, i.e., all values are finite and the
        # step reduces the value of the linearized model.
        #
        # Note: step_is_valid is false when iteration = 0.
        bool step_is_valid

        # Step did not reduce the value of the objective function
        # sufficiently, but it was accepted because of the relaxed
        # acceptance criterion used by the non-monotonic trust region
        # algorithm.
        #
        # Note: step_is_nonmonotonic is false when iteration = 0
        bool step_is_nonmonotonic

        # Whether or not the minimizer accepted this step or not. If the
        # ordinary trust region algorithm is used, this means that the
        # relative reduction in the objective function value was greater
        # than Solver::Options::min_relative_decrease. However, if the
        # non-monotonic trust region algorithm is used
        # (Solver::Options:use_nonmonotonic_steps = true), then even if the
        # relative decrease is not sufficient, the algorithm may accept the
        # step and the step is declared successful.
        #
        # Note: step_is_successful is false when iteration = 0.
        bool step_is_successful

        # Value of the objective function.
        double cost

        # Change in the value of the objective function in this
        # iteration. This can be positive or negative.
        double cost_change

        # Infinity norm of the gradient vector.
        double gradient_max_norm

        # 2-norm of the size of the step computed by the optimization
        # algorithm.
        double step_norm

        # For trust region algorithms, the ratio of the actual change in
        # cost and the change in the cost of the linearized approximation.
        double relative_decrease

        # Size of the trust region at the end of the current iteration. For
        # the Levenberg-Marquardt algorithm, the regularization parameter
        # mu = 1.0 / trust_region_radius.
        double trust_region_radius

        # For the inexact step Levenberg-Marquardt algorithm, this is the
        # relative accuracy with which the Newton(LM) step is solved. This
        # number affects only the iterative solvers capable of solving
        # linear systems inexactly. Factorization-based exact solvers
        # ignore it.
        double eta

        # Step sized computed by the line search algorithm.
        double step_size

        # Number of function evaluations used by the line search algorithm.
        int line_search_function_evaluations

        # Number of iterations taken by the linear solver to solve for the
        # Newton step.
        int linear_solver_iterations

        # Time (in seconds) spent inside the minimizer loop in the current
        # iteration.
        double iteration_time_in_seconds

        # Time (in seconds) spent inside the trust region step solver.
        double step_solver_time_in_seconds

        # Time (in seconds) since the user called Solve().
        double cumulative_time_in_seconds


    cdef cppclass IterationCallback:
        CallbackReturnType operator()(const IterationSummary& summary)

cdef extern from "crs_matrix.h" namespace "ceres":
    ctypedef struct CRSMatrix:
        CRSMatrix()

        int num_rows
        int num_cols

        vector[int] cols
        vector[int] rows
        vector[double] values

cdef extern from "local_parameterization.h" namespace "ceres":
    cdef cppclass LocalParameterization:
        bool Plus(const double* x,
                  const double* delta,
                  double* x_plus_delta) const

        bool ComputeJacobian(const double* x, double* jacobian) const
        int GlobalSize() const

        LocalSize() const

cdef extern from "problem.h" namespace "ceres::internal":

    cdef cppclass Preprocessor:
        pass
    cdef cppclass ProblemImpl:
        pass
    cdef cppclass ParameterBlock:
        pass
    cdef cppclass ResidualBlock:
        pass

cdef extern from "loss_function.h" namespace "ceres":

    cdef cppclass LossFunction:
        void Evaluate(double sq_norm, double out[3]) const

    cdef cppclass HuberLoss(LossFunction):
        HuberLoss(double _a)

    cdef cppclass SoftLOneLoss(LossFunction):
        SoftLOneLoss(double _a)

    cdef cppclass CauchyLoss(LossFunction):
        CauchyLoss(double _a)

    cdef cppclass ArctanLoss(LossFunction):
        ArctanLoss(double _a)

    cdef cppclass TolerantLoss(LossFunction):
         TolerantLoss(double _a, double _b)

    cdef cppclass ComposedLoss(LossFunction):
        ComposedLoss(const LossFunction* f, Ownership ownership_f,
                     const LossFunction* g, Ownership ownership_g)

    cdef cppclass ScaledLoss(LossFunction):
        ScaledLoss(const LossFunction* rho, double a, Ownership ownership)


cdef extern from "cost_function.h" namespace "ceres":
    cdef cppclass CostFunction:
        bool Evaluate(double** parameters,
                      double* residuals,
                      double** jacobians) const

        const vector[int16]& parameter_block_sizes() const

        int num_residuals() const

cdef extern from "solver.h" namespace "ceres::Solver":
    cdef cppclass SolverOptions "ceres::Solver::Options":
        MinimizerType minimizer_type

        LineSearchDirectionType line_search_direction_type
        LineSearchType line_search_type
        NonlinearConjugateGradientType nonlinear_conjugate_gradient_type

        # The LBFGS hessian approximation is a low rank approximation to
        # the inverse of the Hessian matrix. The rank of the
        # approximation determines (linearly) the space and time
        # complexity of using the approximation. Higher the rank, the
        # better is the quality of the approximation. The increase in
        # quality is however is bounded for a number of reasons.
        #
        # 1. The method only uses secant information and not actual
        # derivatives.
        #
        # 2. The Hessian approximation is constrained to be positive
        # definite.
        #
        # So increasing this rank to a large number will cost time and
        # space complexity without the corresponding increase in solution
        # quality. There are no hard and fast rules for choosing the
        # maximum rank. The best choice usually requires some problem
        # specific experimentation.
        #
        # For more theoretical and implementation details of the LBFGS
        # method, please see:
        #
        # Nocedal, J. (1980). "Updating Quasi-Newton Matrices with
        # Limited Storage". Mathematics of Computation 35 (151): 773–782.
        int max_lbfgs_rank

        TrustRegionStrategyType trust_region_strategy_type

        # Type of dogleg strategy to use.
        DoglegType dogleg_type

        # The classical trust region methods are descent methods, in that
        # they only accept a point if it strictly reduces the value of
        # the objective function.
        #
        # Relaxing this requirement allows the algorithm to be more
        # efficient in the long term at the cost of some local increase
        # in the value of the objective function.
        #
        # This is because allowing for non-decreasing objective function
        # values in a princpled manner allows the algorithm to "jump over
        # boulders" as the method is not restricted to move into narrow
        # valleys while preserving its convergence properties.
        #
        # Setting use_nonmonotonic_steps to true enables the
        # non-monotonic trust region algorithm as described by Conn,
        # Gould & Toint in "Trust Region Methods", Section 10.1.
        #
        # The parameter max_consecutive_nonmonotonic_steps controls the
        # window size used by the step selection algorithm to accept
        # non-monotonic steps.
        #
        # Even though the value of the objective function may be larger
        # than the minimum value encountered over the course of the
        # optimization, the final parameters returned to the user are the
        # ones corresponding to the minimum cost over all iterations.
        bool use_nonmonotonic_steps
        int max_consecutive_nonmonotonic_steps

        # Maximum number of iterations for the minimizer to run for.
        int max_num_iterations

        # Maximum time for which the minimizer should run for.
        double max_solver_time_in_seconds

        # Number of threads used by Ceres for evaluating the cost and
        # jacobians.
        int num_threads

        # Trust region minimizer settings.
        double initial_trust_region_radius
        double max_trust_region_radius

        # Minimizer terminates when the trust region radius becomes
        # smaller than this value.
        double min_trust_region_radius

        # Lower bound for the relative decrease before a step is
        # accepted.
        double min_relative_decrease

        # For the Levenberg-Marquadt algorithm, the scaled diagonal of
        # the normal equations J'J is used to control the size of the
        # trust region. Extremely small and large values along the
        # diagonal can make this regularization scheme
        # fail. lm_max_diagonal and lm_min_diagonal, clamp the values of
        # diag(J'J) from above and below. In the normal course of
        # operation, the user should not have to modify these parameters.
        double lm_min_diagonal
        double lm_max_diagonal

        # Sometimes due to numerical conditioning problems or linear
        # solver flakiness, the trust region strategy may return a
        # numerically invalid step that can be fixed by reducing the
        # trust region size. So the TrustRegionMinimizer allows for a few
        # successive invalid steps before it declares NUMERICAL_FAILURE.
        int max_num_consecutive_invalid_steps

        # Minimizer terminates when
        #
        #   (new_cost - old_cost) < function_tolerance * old_cost
        #
        double function_tolerance

        # Minimizer terminates when
        #
        #   max_i |gradient_i| < gradient_tolerance * max_i|initial_gradient_i|
        #
        # This value should typically be 1e-4 * function_tolerance.
        double gradient_tolerance

        # Minimizer terminates when
        #
        #   |step|_2 <= parameter_tolerance * ( |x|_2 +  parameter_tolerance)
        #
        double parameter_tolerance

        # Linear least squares solver options -------------------------------------

        LinearSolverType linear_solver_type

        # Type of preconditioner to use with the iterative linear solvers.
        PreconditionerType preconditioner_type

        # Ceres supports using multiple sparse linear algebra libraries
        # for sparse matrix ordering and factorizations. Currently,
        # SUITE_SPARSE and CX_SPARSE are the valid choices, depending on
        # whether they are linked into Ceres at build time.
        SparseLinearAlgebraLibraryType sparse_linear_algebra_library

        # Number of threads used by Ceres to solve the Newton
        # step. Currently only the SPARSE_SCHUR solver is capable of
        # using this setting.
        int num_linear_solver_threads

        # The order in which variables are eliminated in a linear solver
        # can have a significant of impact on the efficiency and accuracy
        # of the method. e.g., when doing sparse Cholesky factorization,
        # there are matrices for which a good ordering will give a
        # Cholesky factor with O(n) storage, where as a bad ordering will
        # result in an completely dense factor.
        #
        # Ceres allows the user to provide varying amounts of hints to
        # the solver about the variable elimination ordering to use. This
        # can range from no hints, where the solver is free to decide the
        # best possible ordering based on the user's choices like the
        # linear solver being used, to an exact order in which the
        # variables should be eliminated, and a variety of possibilities
        # in between.
        #
        # Instances of the ParameterBlockOrdering class are used to
        # communicate this information to Ceres.
        #
        # Formally an ordering is an ordered partitioning of the
        # parameter blocks, i.e, each parameter block belongs to exactly
        # one group, and each group has a unique non-negative integer
        # associated with it, that determines its order in the set of
        # groups.
        #
        # Given such an ordering, Ceres ensures that the parameter blocks in
        # the lowest numbered group are eliminated first, and then the
        # parmeter blocks in the next lowest numbered group and so on. Within
        # each group, Ceres is free to order the parameter blocks as it
        # chooses.
        #
        # If NULL, then all parameter blocks are assumed to be in the
        # same group and the solver is free to decide the best
        # ordering.
        #
        # e.g. Consider the linear system
        #
        #   x + y = 3
        #   2x + 3y = 7
        #
        # There are two ways in which it can be solved. First eliminating x
        # from the two equations, solving for y and then back substituting
        # for x, or first eliminating y, solving for x and back substituting
        # for y. The user can construct three orderings here.
        #
        #   {0: x}, {1: y} - eliminate x first.
        #   {0: y}, {1: x} - eliminate y first.
        #   {0: x, y}      - Solver gets to decide the elimination order.
        #
        # Thus, to have Ceres determine the ordering automatically using
        # heuristics, put all the variables in group 0 and to control the
        # ordering for every variable, create groups 0..N-1, one per
        # variable, in the desired order.
        #
        # Bundle Adjustment
        # -----------------
        #
        # A particular case of interest is bundle adjustment, where the user
        # has two options. The default is to not specify an ordering at all,
        # the solver will see that the user wants to use a Schur type solver
        # and figure out the right elimination ordering.
        #
        # But if the user already knows what parameter blocks are points and
        # what are cameras, they can save preprocessing time by partitioning
        # the parameter blocks into two groups, one for the points and one
        # for the cameras, where the group containing the points has an id
        # smaller than the group containing cameras.
        #
        # Once assigned, Solver::Options owns this pointer and will
        # deallocate the memory when destroyed.
        ParameterBlockOrdering* linear_solver_ordering

        # Sparse Cholesky factorization algorithms use a fill-reducing
        # ordering to permute the columns of the Jacobian matrix. There
        # are two ways of doing this.

        # 1. Compute the Jacobian matrix in some order and then have the
        #    factorization algorithm permute the columns of the Jacobian.

        # 2. Compute the Jacobian with its columns already permuted.

        # The first option incurs a significant memory penalty. The
        # factorization algorithm has to make a copy of the permuted
        # Jacobian matrix, thus Ceres pre-permutes the columns of the
        # Jacobian matrix and generally speaking, there is no performance
        # penalty for doing so.

        # In some rare cases, it is worth using a more complicated
        # reordering algorithm which has slightly better runtime
        # performance at the expense of an extra copy of the Jacobian
        # matrix. Setting use_postordering to true enables this tradeoff.
        bool use_postordering

        # Some non-linear least squares problems have additional
        # structure in the way the parameter blocks interact that it is
        # beneficial to modify the way the trust region step is computed.
        #
        # e.g., consider the following regression problem
        #
        #   y = a_1 exp(b_1 x) + a_2 exp(b_3 x^2 + c_1)
        #
        # Given a set of pairs{(x_i, y_i)}, the user wishes to estimate
        # a_1, a_2, b_1, b_2, and c_1.
        #
        # Notice here that the expression on the left is linear in a_1
        # and a_2, and given any value for b_1, b_2 and c_1, it is
        # possible to use linear regression to estimate the optimal
        # values of a_1 and a_2. Indeed, its possible to analytically
        # eliminate the variables a_1 and a_2 from the problem all
        # together. Problems like these are known as separable least
        # squares problem and the most famous algorithm for solving them
        # is the Variable Projection algorithm invented by Golub &
        # Pereyra.
        #
        # Similar structure can be found in the matrix factorization with
        # missing data problem. There the corresponding algorithm is
        # known as Wiberg's algorithm.
        #
        # Ruhe & Wedin (Algorithms for Separable Nonlinear Least Squares
        # Problems, SIAM Reviews, 22(3), 1980) present an analyis of
        # various algorithms for solving separable non-linear least
        # squares problems and refer to "Variable Projection" as
        # Algorithm I in their paper.
        #
        # Implementing Variable Projection is tedious and expensive, and
        # they present a simpler algorithm, which they refer to as
        # Algorithm II, where once the Newton/Trust Region step has been
        # computed for the whole problem (a_1, a_2, b_1, b_2, c_1) and
        # additional optimization step is performed to estimate a_1 and
        # a_2 exactly.
        #
        # This idea can be generalized to cases where the residual is not
        # linear in a_1 and a_2, i.e., Solve for the trust region step
        # for the full problem, and then use it as the starting point to
        # further optimize just a_1 and a_2. For the linear case, this
        # amounts to doing a single linear least squares solve. For
        # non-linear problems, any method for solving the a_1 and a_2
        # optimization problems will do. The only constraint on a_1 and
        # a_2 is that they do not co-occur in any residual block.
        #
        # This idea can be further generalized, by not just optimizing
        # (a_1, a_2), but decomposing the graph corresponding to the
        # Hessian matrix's sparsity structure in a collection of
        # non-overlapping independent sets and optimizing each of them.
        #
        # Setting "use_inner_iterations" to true enables the use of this
        # non-linear generalization of Ruhe & Wedin's Algorithm II.  This
        # version of Ceres has a higher iteration complexity, but also
        # displays better convergence behaviour per iteration. Setting
        # Solver::Options::num_threads to the maximum number possible is
        # highly recommended.
        bool use_inner_iterations

        # If inner_iterations is true, then the user has two choices.
        #
        # 1. Let the solver heuristically decide which parameter blocks
        #    to optimize in each inner iteration. To do this leave
        #    Solver::Options::inner_iteration_ordering untouched.
        #
        # 2. Specify a collection of of ordered independent sets. Where
        #    the lower numbered groups are optimized before the higher
        #    number groups. Each group must be an independent set.
        ParameterBlockOrdering* inner_iteration_ordering

        # Minimum number of iterations for which the linear solver should
        # run, even if the convergence criterion is satisfied.
        int linear_solver_min_num_iterations

        # Maximum number of iterations for which the linear solver should
        # run. If the solver does not converge in less than
        # linear_solver_max_num_iterations, then it returns
        # MAX_ITERATIONS, as its termination type.
        int linear_solver_max_num_iterations

        # Forcing sequence parameter. The truncated Newton solver uses
        # this number to control the relative accuracy with which the
        # Newton step is computed.
        #
        # This constant is passed to ConjugateGradientsSolver which uses
        # it to terminate the iterations when
        #
        #  (Q_i - Q_{i-1})/Q_i < eta/i
        double eta

        # Normalize the jacobian using Jacobi scaling before calling
        # the linear least squares solver.
        bool jacobi_scaling

        # Logging options ---------------------------------------------------------

        LoggingType logging_type

        # By default the Minimizer progress is logged to VLOG(1), which
        # is sent to STDERR depending on the vlog level. If this flag is
        # set to true, and logging_type is not SILENT, the logging output
        # is sent to STDOUT.
        bool minimizer_progress_to_stdout

        # List of iterations at which the optimizer should dump the
        # linear least squares problem to disk. Useful for testing and
        # benchmarking. If empty (default), no problems are dumped.
        #
        # This is ignored if protocol buffers are disabled.
        vector[int] lsqp_iterations_to_dump
        string lsqp_dump_directory
        DumpFormatType lsqp_dump_format_type

        # Finite differences options ----------------------------------------------

        # Check all jacobians computed by each residual block with finite
        # differences. This is expensive since it involves computing the
        # derivative by normal means (e.g. user specified, autodiff,
        # etc), then also computing it using finite differences. The
        # results are compared, and if they differ substantially, details
        # are printed to the log.
        bool check_gradients

        # Relative precision to check for in the gradient checker. If the
        # relative difference between an element in a jacobian exceeds
        # this number, then the jacobian for that cost term is dumped.
        double gradient_check_relative_precision

        # Relative shift used for taking numeric derivatives. For finite
        # differencing, each dimension is evaluated at slightly shifted
        # values for the case of central difference, this is what gets
        # evaluated:
        #
        #   delta = numeric_derivative_relative_step_size
        #   f_initial  = f(x)
        #   f_forward  = f((1 + delta) * x)
        #   f_backward = f((1 - delta) * x)
        #
        # The finite differencing is done along each dimension. The
        # reason to use a relative (rather than absolute) step size is
        # that this way, numeric differentation works for functions where
        # the arguments are typically large (e.g. 1e9) and when the
        # values are small (e.g. 1e-5). It is possible to construct
        # "torture cases" which break this finite difference heuristic,
        # but they do not come up often in practice.
        #
        # TODO(keir): Pick a smarter number than the default above! In
        # theory a good choice is sqrt(eps) * x, which for doubles means
        # about 1e-8 * x. However, I have found this number too
        # optimistic. This number should be exposed for users to change.
        double numeric_derivative_relative_step_size

        # If true, the user's parameter blocks are updated at the end of
        # every Minimizer iteration, otherwise they are updated when the
        # Minimizer terminates. This is useful if, for example, the user
        # wishes to visualize the state of the optimization every
        # iteration.
        bool update_state_every_iteration

        # Callbacks that are executed at the end of each iteration of the
        # Minimizer. An iteration may terminate midway, either due to
        # numerical failures or because one of the convergence tests has
        # been satisfied. In this case none of the callbacks are
        # executed.

        # Callbacks are executed in the order that they are specified in
        # this vector. By default, parameter blocks are updated only at
        # the end of the optimization, i.e when the Minimizer
        # terminates. This behaviour is controlled by
        # update_state_every_variable. If the user wishes to have access
        # to the update parameter blocks when his/her callbacks are
        # executed, then set update_state_every_iteration to true.
        #
        # The solver does NOT take ownership of these pointers.
        vector[IterationCallback*] callbacks

        # If non-empty, a summary of the execution of the solver is
        # recorded to this file.
        string solver_log

    cdef cppclass Summary:
        Summary()

        # A brief one line description of the state of the solver after
        # termination.
        string BriefReport() const

        # A full multiline description of the state of the solver after
        # termination.
        string FullReport() const

        # Minimizer summary -------------------------------------------------
        MinimizerType minimizer_type

        SolverTerminationType termination_type

        # If the solver did not run, or there was a failure, a
        # description of the error.
        string error

        # Cost of the problem before and after the optimization. See
        # problem.h for definition of the cost of a problem.
        double initial_cost
        double final_cost

        # The part of the total cost that comes from residual blocks that
        # were held fixed by the preprocessor because all the parameter
        # blocks that they depend on were fixed.
        double fixed_cost

        vector[IterationSummary] iterations

        int num_successful_steps
        int num_unsuccessful_steps

        # When the user calls Solve, before the actual optimization
        # occurs, Ceres performs a number of preprocessing steps. These
        # include error checks, memory allocations, and reorderings. This
        # time is accounted for as preprocessing time.
        double preprocessor_time_in_seconds

        # Time spent in the TrustRegionMinimizer.
        double minimizer_time_in_seconds

        # After the Minimizer is finished, some time is spent in
        # re-evaluating residuals etc. This time is accounted for in the
        # postprocessor time.
        double postprocessor_time_in_seconds

        # Some total of all time spent inside Ceres when Solve is called.
        double total_time_in_seconds

        double linear_solver_time_in_seconds
        double residual_evaluation_time_in_seconds
        double jacobian_evaluation_time_in_seconds

        # Preprocessor summary.
        int num_parameter_blocks
        int num_parameters
        int num_effective_parameters
        int num_residual_blocks
        int num_residuals

        int num_parameter_blocks_reduced
        int num_parameters_reduced
        int num_effective_parameters_reduced
        int num_residual_blocks_reduced
        int num_residuals_reduced

        int num_eliminate_blocks_given
        int num_eliminate_blocks_used

        int num_threads_given
        int num_threads_used

        int num_linear_solver_threads_given
        int num_linear_solver_threads_used

        LinearSolverType linear_solver_type_given
        LinearSolverType linear_solver_type_used

        vector[int] linear_solver_ordering_given
        vector[int] linear_solver_ordering_used

        PreconditionerType preconditioner_type

        TrustRegionStrategyType trust_region_strategy_type
        DoglegType dogleg_type
        bool inner_iterations

        SparseLinearAlgebraLibraryType sparse_linear_algebra_library

        LineSearchDirectionType line_search_direction_type
        LineSearchType line_search_type
        int max_lbfgs_rank

        vector[int] inner_iteration_ordering_given
        vector[int] inner_iteration_ordering_used

cdef extern from "solver.h" namespace "ceres":

    void Solve(const SolverOptions& options,
                Problem* problem,
                Summary* summary)

cdef extern from "problem.h" namespace "ceres::Problem":

    ctypedef ResidualBlock* ResidualBlockId

    ctypedef struct ProblemOptions "ceres::Problem::Options":
        ProblemOptions()
        Ownership cost_function_ownership
        Ownership loss_function_ownership
        Ownership local_parameterization_ownership

        bool enable_fast_parameter_block_removal
        bool disable_all_safety_checks

    ctypedef struct EvaluateOptions:
        EvaluateOptions()
        vector[double*] parameter_blocks
        vector[ResidualBlockId] residual_blocks
        bool apply_loss_function
        int num_threads

cdef extern from "problem.h" namespace "ceres":

    ctypedef ResidualBlock* ResidualBlockId

    cdef cppclass Problem:

        Problem()
        Problem(const ProblemOptions& options)

        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                         LossFunction* loss_function,
                                         const vector[double*]& parameter_blocks)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4, double* x5)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4, double* x5,
                                        double* x6)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4, double* x5,
                                        double* x6, double* x7)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4, double* x5,
                                        double* x6, double* x7, double* x8)
        ResidualBlockId AddResidualBlock(CostFunction* cost_function,
                                        LossFunction* loss_function,
                                        double* x0, double* x1, double* x2,
                                        double* x3, double* x4, double* x5,
                                        double* x6, double* x7, double* x8,
                                        double* x9)

        void AddParameterBlock(double* values, int size)

        void AddParameterBlock(double* values,
                               int size,
                               LocalParameterization* local_parameterization)

        void RemoveParameterBlock(double* values)

        void RemoveResidualBlock(ResidualBlockId residual_block)

        void SetParameterBlockConstant(double* values)

        void SetParameterBlockVariable(double* values)

        void SetParameterization(double* values,
                                LocalParameterization* local_parameterization)

        int NumParameterBlocks() const

        int NumParameters() const

        int NumResidualBlocks() const

        int NumResiduals() const

        int ParameterBlockSize(double* values) const

        int ParameterBlockLocalSize(double* values) const

        void GetParameterBlocks(vector[double*]* parameter_blocks) const

        bool Evaluate(const EvaluateOptions& options,
                        double* cost,
                        vector[double]* residuals,
                        vector[double]* gradient,
                        CRSMatrix* jacobian)
