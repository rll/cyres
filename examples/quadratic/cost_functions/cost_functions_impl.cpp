#include "ceres/ceres.h"
#include "cost_functions.h"

using ceres::CostFunction;
using ceres::SizedCostFunction;

class SimpleCostFunction
  : public SizedCostFunction<1 /* number of residuals */,
                             1 /* size of first parameter */> {
 public:
  virtual ~SimpleCostFunction() {}
  virtual bool Evaluate(double const* const* parameters,
                        double* residuals,
                        double** jacobians) const {
    double x = parameters[0][0];

    // f(x) = 10 - x.
    residuals[0] = 10 - x;

    // f'(x) = -1. Since there's only 1 parameter and that parameter
    // has 1 dimension, there is only 1 element to fill in the
    // jacobians.
    if (jacobians != NULL && jacobians[0] != NULL) {
      jacobians[0][0] = -1;
    }
    return true;
  }
};

CostFunction* createCostFunction(){
  return new SimpleCostFunction();
}
