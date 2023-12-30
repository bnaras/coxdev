#ifdef DEBUG
#include <iostream>
#endif

#ifdef PY_INTERFACE

#include <cstddef>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>

// Map python buffers list element into an Eigen double vector
// SRC_LIST = python list, OFFSET = index offset (e.g. 0 or 1),
// DEST will be the ref downstream, TMP should be **unique** throwaway name with each invocation
#define MAP_BUFFER_LIST(SRC_LIST, OFFSET, DEST, TMP)				\
  py::array_t<double> TMP = SRC_LIST[OFFSET].cast<py::array_t<double>>();       \
  Eigen::Map<Eigen::VectorXd> DEST(TMP.mutable_data(), TMP.size());

namespace py = pybind11;
#define EIGEN_REF Eigen::Ref
#define ERROR_MSG(x) throw std::runtime_error(x)
#define BUFFER_LIST(x) py::list &x 
#define HESSIAN_MATVEC_TYPE void
#endif

#ifdef R_INTERFACE

#include <RcppEigen.h>

// Map buffer list element into an Eigen double vector
// SRC_LIST = R list, OFFSET = index offset (e.g. 0 or 1),
// DEST will be the ref downstream, TMP should be **unique** throwaway name with each invocation
#define MAP_BUFFER_LIST(SRC_LIST, OFFSET, DEST, TMP)				\
  Rcpp::NumericVector TMP = Rcpp::as<Rcpp::NumericVector>(SRC_LIST[OFFSET]); \
  Eigen::Map<Eigen::VectorXd> Rcpp::as<Eigen::Map<Eigen::VectorXd>>(TMP);

using namespace Rcpp;
#define EIGEN_REF Eigen::Map
#define ERROR_MSG(x) Rcpp::Rcerr << x
#define BUFFER_LIST(x) Rcpp::List x // List of vectors for scratch space.
#define HESSIAN_MATVEC_TYPE Eigen::VectorXd
#endif

