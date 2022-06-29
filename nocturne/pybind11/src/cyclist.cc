// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "cyclist.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "object.h"

namespace py = pybind11;

namespace nocturne {

void DefineCyclist(py::module& m) {
  py::class_<Cyclist, std::shared_ptr<Cyclist>, Object>(m, "Cyclist");
}

}  // namespace nocturne
