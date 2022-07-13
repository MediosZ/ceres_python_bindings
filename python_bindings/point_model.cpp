#include <pybind11/numpy.h>
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

double* ParseNumpyData(py::array_t<double>& np_buf);

// def optimization(cur_pos, tar_pos, distance, gradient, eta=1, lamb=0.1,
// eta_d=0.1):
//     model = gurobipy.Model()
//     lb = -gurobipy.GRB.INFINITY
//     ub = gurobipy.GRB.INFINITY
//     v1 = model.addVar(lb, ub, 0.0, vtype=gurobipy.GRB.CONTINUOUS, name="v1")
//     v2 = model.addVar(lb, ub, 0.0, vtype=gurobipy.GRB.CONTINUOUS, name="v2")
//     v3 = model.addVar(lb, ub, 0.0, vtype=gurobipy.GRB.CONTINUOUS, name="v3")
//     model.update()
//     # print(cur_pos, tar_pos)
//     # print(distance, gradient)
//     delta_pos = cur_pos - tar_pos
//     def f():
//         (v1 + eta * delta_pos[0]) * (v1 + eta * delta_pos[0])
//         + (v2 + eta * delta_pos[1]) * (v2 + eta * delta_pos[1])
//         + (v3 + eta * delta_pos[2]) * (v3 + eta * delta_pos[2])
//         + lamb * v1**2 + lamb * v2**2 + lamb * v3**2
//     model.setObjective( f, gurobipy.GRB.MINIMIZE)
//     model.addConstr(v1 * gradient[0] + v2 * gradient[1] + v3 * gradient[2] >=
//     -eta_d * (distance - 0.005), "c0") model.optimize() return (v1.X, v2.X,
//     v3.X)

namespace {

template <typename T>
T pow2(T value) {
  return value * value;
}

struct PointCostFunctor {
 public:
  PointCostFunctor(double* target) : _target(target) {}
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = pow2(T(_target[0]) - x[0]);
    residual[1] = pow2(T(_target[1]) - x[1]);
    residual[2] = pow2(T(_target[2]) - x[2]);
    return true;
  }
  double* _target;
};

struct PointNormFunctor {
 public:
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = 0.1 * pow2(x[0]);
    residual[1] = 0.1 * pow2(x[1]);
    residual[2] = 0.1 * pow2(x[2]);
    return true;
  }
  double* _target;
};

struct PointConstraintFunctor {
 public:
  PointConstraintFunctor(double* gradient, double distance, double safe_d)
      : _gradient(gradient), _distance(distance), _safe_d(safe_d) {}
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    residual[0] = x[0];
    return x[0] * _gradient[0] + x[1] * _gradient[1] + x[2] * _gradient[2] >=
           -0.1 * (_distance - _safe_d);
  }
  double* _gradient;
  double _distance;
  double _safe_d;
};

class PointProblem {
 public:
  PointProblem() { vars = new double[3]{0.0, 0.0, 0.0}; }
  ~PointProblem() { delete[] vars; }
  double* vars;
};

}  // namespace

std::vector<double> point_optimization(py::array_t<double>& delta_pos,
                                       double distance,
                                       py::array_t<double>& gradient,
                                       double safe_d,
                                       double eta,
                                       double lambda,
                                       double eta_d) {
  //   auto point_problem = PointProblem();
  //   double vars[] = {0.0, 0.0, 0.0};
  auto vars = std::vector<double>{0.0, 0.0, 0.0};

  // Build the problem.
  Problem problem;
  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  double* delta_pos_pointer = ParseNumpyData(delta_pos);
  double* gradient_pointer = ParseNumpyData(delta_pos);

  CostFunction* cost_function =
      new AutoDiffCostFunction<::PointCostFunctor, 3, 3>(
          new ::PointCostFunctor(delta_pos_pointer));
  problem.AddResidualBlock(cost_function, nullptr, vars.data());

  //   CostFunction* norm_function =
  //       new AutoDiffCostFunction<::PointNormFunctor, 3, 3>(
  //           new ::PointNormFunctor);
  //   problem.AddResidualBlock(norm_function, nullptr, vars.data());

  CostFunction* constraint =
      new AutoDiffCostFunction<::PointConstraintFunctor, 1, 3>(
          new ::PointConstraintFunctor(gradient_pointer, distance, safe_d));
  problem.AddResidualBlock(constraint, nullptr, vars.data());

  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  Solve(options, &problem, &summary);

  return vars;
}

void add_pybinded_point_model(py::module& m) {
  m.def("point_optimization", &point_optimization, R"pbdoc(
        Point optimization.
    )pbdoc");
}
