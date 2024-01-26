// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "vehicle.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

void DefineVehicle(py::module& m) {
  py::class_<Vehicle, std::shared_ptr<Vehicle>, Object>(m, "Vehicle")
  .def_property_readonly("is_av", &Vehicle::is_av);
}

}  // namespace nocturne
