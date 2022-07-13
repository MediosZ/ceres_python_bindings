#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "ceres/ceres.h"
#include "ceres/rotation.h"

namespace py = pybind11;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;
// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 110.0 - x[0];
    residual[1] = 20.0 - x[1];
    return true;
  }
};
int main_ceres() {
  // The variable to solve for with its initial value. It will be
  // mutated in place by the solver.
  // double x = -20.0;
  // double x2 = -10;
  auto vars = std::vector<double>{-20, -10};
  const double initial_x = vars[0];
  // Build the problem.
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  CostFunction* cost_function =
      new AutoDiffCostFunction<CostFunctor, 2, 2>(new CostFunctor);
  problem.AddResidualBlock(cost_function, nullptr, vars.data());
  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  std::cout << summary.BriefReport() << "\n";
  std::cout << "x : " << initial_x << " -> " << vars[0] << " " << vars[1]
            << "\n";
  std::cout << "get fork" << std::endl;
  return 0;
}
void add_pybinded_ceres_entry(py::module& m) {
  m.def("main_ceres", &main_ceres, R"pbdoc(
        Ceres entry.
    )pbdoc");
}
