from cyres import *
from cost_functions.wrappers import SimpleCostFunction

x = np.array([5.])
problem = Problem()
problem.add_residual_block(SimpleCostFunction(), SquaredLoss(), [x])

options = SolverOptions()
options.max_num_iterations = 50
options.linear_solver_type = LinearSolverType.DENSE_QR
options.trust_region_strategy_type = TrustRegionStrategyType.DOGLEG
options.dogleg_type = DoglegType.SUBSPACE_DOGLEG
options.minimizer_progress_to_stdout = False

summary = Summary()

solve(options, problem, summary)
print summary.briefReport()
print summary.fullReport()
print x
