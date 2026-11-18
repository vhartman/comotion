#pragma once
#include <pybind11/numpy.h>
#include <stdexcept>

namespace comotion_python {
// Sequential native edge batch; returns one verdict per input edge. Kernels
// still vectorize interpolation samples within each edge.
template <typename Check>
pybind11::array_t<bool> check_edge_batch(
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> from,
    pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast> to,
    pybind11::array_t<int, pybind11::array::c_style> counts, Check check) {
  namespace py = pybind11;
  using Array = py::array_t<double, py::array::c_style | py::array::forcecast>;
  if (from.ndim() != 2 || to.ndim() != 2 || from.shape(0) != to.shape(0) ||
      from.shape(1) != to.shape(1) || counts.ndim() != 1 || counts.size() != from.shape(0))
    throw std::invalid_argument("Expected edge arrays (N,dof) and int32 counts (N,)");
  py::array_t<bool> out(from.shape(0));
  for (py::ssize_t i = 0; i < from.shape(0); ++i) {
    Array a({from.shape(1)}, from.data(i), from), b({to.shape(1)}, to.data(i), to);
    out.mutable_data()[i] = std::get<0>(check(a, b, counts.data()[i]));
  }
  return out;
}
}
