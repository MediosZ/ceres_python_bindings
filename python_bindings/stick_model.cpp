#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "Eigen/Core"
#include "ceres/autodiff_manifold.h"
#include "ceres/ceres.h"
#include "ceres/internal/eigen.h"
#include "ceres/rotation.h"
namespace py = pybind11;
using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

double* ParseNumpyData(py::array_t<double>& np_buf);

namespace {

// class StickProblem {
//  public:
//   StickProblem() {}
//   ~StickProblem() {}

//  private:
//   double* _v;          // optimized velocity.
//   double* _w;          // optimized angle velocity.
//   double* _positions;  // positions on stick.
//   double* _distances;  // distances of each position.
//   double* _gradients;  // gradients of each position.
// };

template <typename T>
T pow2(T value) {
  return value * value;
}

struct StickCostFunctor {
 public:
  StickCostFunctor(double* target, double* target_q)
      : _target(target), _target_q(target_q) {}
  template <typename T>
  bool operator()(const T* const x, const T* const q, T* residuals_ptr) const {
    // residual[0] = pow2(T(_target[0]) - x[0]);
    // residual[1] = pow2(T(_target[1]) - x[1]);
    // residual[2] = pow2(T(_target[2]) - x[2]);
    // residual[3] = pow2(T(_target_q[0]) - q[0]);
    // residual[4] = pow2(T(_target_q[1]) - q[1]);
    // residual[5] = pow2(T(_target_q[2]) - q[2]);
    // residual[6] = pow2(T(_target_q[3]) - q[3]);
    auto targ_p =
        Eigen::Matrix<T, 3, 1>(T(_target[0]), T(_target[1]), T(_target[2]));
    auto esti_p = Eigen::Matrix<T, 3, 1>(x);
    Eigen::Matrix<T, 3, 1> delta_p = targ_p - esti_p;

    auto esti_quat = Eigen::Quaternion<T>(q[3], q[0], q[1], q[2]);
    auto targ_quat = Eigen::Quaternion<T>(
        T(_target_q[3]), T(_target_q[0]), T(_target_q[1]), T(_target_q[2]));
    Eigen::Quaternion<T> delta_q = targ_quat * esti_quat.conjugate();

    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template block<3, 1>(0, 0) = delta_p.array().square();
    residuals.template block<3, 1>(3, 0) = T(2.0) * delta_q.vec();
    return true;
  }
  double* _target;
  double* _target_q;
};

// struct PointNormFunctor {
//  public:
//   template <typename T>
//   bool operator()(const T* const x, T* residual) const {
//     residual[0] = 0.1 * pow2(x[0]);
//     residual[1] = 0.1 * pow2(x[1]);
//     residual[2] = 0.1 * pow2(x[2]);
//     return true;
//   }
//   double* _target;
// };

struct StickConstraintFunctor {
 public:
  StickConstraintFunctor(double* distance /* count */,
                         double* gradient /* count x 3 */,
                         double* positions /* count x 3 */,
                         double safe_d,
                         int count)
      : _gradient(gradient),
        _distance(distance),
        _positions(positions),
        _safe_d(safe_d),
        _count(count) {}

  template <typename T>
  bool operator()(const T* const x, const T* const q, T* residual) const {
    // w, x, y, z
    auto quat = Eigen::Quaternion<T>(q[3], q[0], q[1], q[2]);
    Eigen::AngleAxis<T> angleaxis = Eigen::AngleAxis<T>(quat);
    auto axis = angleaxis.axis();
    auto angle = angleaxis.angle();
    // quat
    // auto eigen_quat = ceres::EigenQuaternionManifold();
    residual[0] = x[0];

    if (x[0] * _gradient[0] + x[1] * _gradient[1] + x[2] * _gradient[2] <
        -0.1 * (_distance[0] - _safe_d)) {
      return false;
    }

    for (auto i = 1; i < _count + 1; i++) {
      auto r =
          Eigen::Matrix<double, 3, 1>(_positions[i * 3] - _positions[0],
                                      _positions[i * 3 + 1] - _positions[1],
                                      _positions[i * 3 + 2] - _positions[2]);
      auto w = axis.cross(r) * angle;
      // if the res is false, return false and skip rest tests
      if ((x[0] + w.coeff(0, 0)) * _gradient[i * 3 + 0] +
              (x[1] + w.coeff(1, 0)) * _gradient[i * 3 + 1] +
              (x[2] + w.coeff(2, 0)) * _gradient[i * 3 + 2] <
          -0.1 * (_distance[i] - _safe_d)) {
        return false;
      }
    }

    // all tests pass, return true.
    return true;
  }

  double* _gradient;
  double* _distance;
  double* _positions;
  double _safe_d;
  int _count;
};

// class PointProblem {
//  public:
//   PointProblem() { vars = new double[3]{0.0, 0.0, 0.0}; }
//   ~PointProblem() { delete[] vars; }
//   double* vars;
// };

}  // namespace

// /*
// variables:
// 1. end position
// 2. end rotation

// input:
// 1. delta_pos (velocity)
// 2. delta_angle (angle velocity)
// 3. positions(11x3)
// 4. distances(11)
// 5. gradients(11x3)
// 4. current direction(1x3)
// output:
// 1. velocity
// 2. angle velocity

// */

std::vector<double> stick_optimization(
    py::array_t<double>& target_pos,   // 1x3
    py::array_t<double>& target_quat,  // 1x4
    py::array_t<double>& pypositions,  // 11x3
    py::array_t<double>& pydistances,  // 11
    py::array_t<double>& pygradients,  // 11x3
    // py::array_t<double>& pydirection,   // 1x3
    double safe_d,
    int count,
    double eta,
    double lambda,
    double eta_d) {
  auto v_vars = std::vector<double>{0.0, 0.0, 0.0};
  auto q_vars = std::vector<double>{0.0, 0.0, 0.0, 1.0};

  // auto vec = Eigen::Vector3d(1, 2, 3);
  // auto vec2 = Eigen::Vector3d(2, 3, 4);
  // auto dir = vec2.cross(vec);
  // std::cout << dir << std::endl;

  // Set up the only cost function (also known as residual). This uses
  // auto-differentiation to obtain the derivative (jacobian).
  double* target_pos_pointer = ParseNumpyData(target_pos);
  double* target_quat_pointer = ParseNumpyData(target_quat);
  double* positions = ParseNumpyData(pypositions);
  double* distances = ParseNumpyData(pydistances);
  double* gradients = ParseNumpyData(pygradients);
  // double* direction = ParseNumpyData(pydirection);

  CostFunction* cost_function =
      new AutoDiffCostFunction<::StickCostFunctor, 6, 3, 4>(
          new ::StickCostFunctor(target_pos_pointer, target_quat_pointer));

  CostFunction* constraint =
      new AutoDiffCostFunction<::StickConstraintFunctor, 1, 3, 4>(
          new ::StickConstraintFunctor(
              distances, gradients, positions, safe_d, count));

  // Build the problem.
  Problem problem;
  problem.AddResidualBlock(
      cost_function, nullptr, v_vars.data(), q_vars.data());
  problem.AddResidualBlock(constraint, nullptr, v_vars.data(), q_vars.data());
  // Run the solver!
  Solver::Options options;
  options.minimizer_progress_to_stdout = false;
  Solver::Summary summary;
  Solve(options, &problem, &summary);
  v_vars.insert(v_vars.end(), q_vars.begin(), q_vars.end());
  return v_vars;
}

void add_pybinded_stick_model(py::module& m) {
  m.def("stick_optimization",
        &stick_optimization,
        R"pbdoc(
        Stick optimization.
    )pbdoc");
  m.def("test_eigen", [](py::array_t<double>& vec) {
    double* positions = ParseNumpyData(vec);
    auto m = Eigen::Matrix<double, 3, 11>(positions);
  });
}
