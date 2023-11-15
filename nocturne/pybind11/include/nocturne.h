// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace nocturne {

void DefineAction(py::module& m);
void DefineObject(py::module& m);
void DefineRoadPoint(py::module& m);
void DefineRoadLine(py::module& m);
void DefineScenario(py::module& m);
void DefineSimulation(py::module& m);
void DefineVector2D(py::module& m);
void DefineVehicle(py::module& m);
void DefinePedestrian(py::module& m);
void DefineCyclist(py::module& m);

}  // namespace nocturne
