// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/vector_2d.h"

#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <string>

#include "nocturne.h"

namespace py = pybind11;

namespace nocturne {

namespace {

py::array_t<float> AsNumpyArray(const geometry::Vector2D& vec) {
  py::array_t<float> arr(2);
  float* arr_data = arr.mutable_data();
  arr_data[0] = vec.x();
  arr_data[1] = vec.y();
  return arr;
}

geometry::Vector2D FromNumpy(const py::array_t<float>& arr) {
  assert(arr.size() == 2);
  const float* arr_data = arr.data();
  return geometry::Vector2D(arr_data[0], arr_data[1]);
}

}  // namespace

void DefineVector2D(py::module& m) {
  py::class_<geometry::Vector2D>(m, "Vector2D")
      .def(py::init<float, float>(), py::arg("x") = 0.0, py::arg("y") = 0.0)
      .def("__repr__",
           [](const geometry::Vector2D& vec) {
             return "(" + std::to_string(vec.x()) + ", " +
                    std::to_string(vec.y()) + ")";
           })
      .def_property("x", &geometry::Vector2D::x, &geometry::Vector2D::set_x)
      .def_property("y", &geometry::Vector2D::y, &geometry::Vector2D::set_y)
      // Operators
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def(-py::self)
      .def(py::self + py::self)
      .def(py::self += py::self)
      .def(py::self + float())
      .def(float() + py::self)
      .def(py::self += float())
      .def(py::self - py::self)
      .def(py::self -= py::self)
      .def(py::self - float())
      .def(py::self -= float())
      .def(py::self * float())
      .def(float() * py::self)
      .def(py::self *= float())
      .def(py::self / float())
      .def(py::self /= float())
      // Methods
      .def("norm", &geometry::Vector2D::Norm, py::arg("p") = 2)
      .def("angle", &geometry::Vector2D::Angle)
      .def("rotate", &geometry::Vector2D::Rotate)
      .def("numpy", &AsNumpyArray)
      .def_static("from_numpy", &FromNumpy)
      .def(py::pickle(
          [](const geometry::Vector2D& vec) { return AsNumpyArray(vec); },
          [](const py::array_t<float>& arr) { return FromNumpy(arr); }));
}

}  // namespace nocturne
