// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "nocturne.h"

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nocturne {
namespace {

PYBIND11_MODULE(nocturne_cpp, m) {
  m.doc() = "Nocturne library - 2D Driving Simulator";

  DefineAction(m);
  DefineObject(m);
  DefineRoadLine(m);
  DefineScenario(m);
  DefineSimulation(m);
  DefineVector2D(m);
  DefineVehicle(m);
}

}  // namespace
}  // namespace nocturne
