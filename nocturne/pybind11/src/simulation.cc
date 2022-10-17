// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "simulation.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <variant>

namespace py = pybind11;

namespace nocturne {

void DefineSimulation(py::module& m) {
  py::class_<Simulation, std::shared_ptr<Simulation>>(m, "Simulation")
      .def(py::init<const std::string&,
                    const std::unordered_map<
                        std::string, std::variant<bool, int64_t, float>>&>(),
           "Constructor for Simulation", py::arg("scenario_path") = "",
           py::arg("config") =
               std::unordered_map<std::string,
                                  std::variant<bool, int64_t, float>>())
      .def("reset", &Simulation::Reset,
           py::call_guard<py::gil_scoped_release>())
      .def("step", &Simulation::Step, py::call_guard<py::gil_scoped_release>())
      .def("scenario", &Simulation::GetScenario,
           py::return_value_policy::reference)
      .def("render", &Simulation::Render)
      .def("save_screenshot", &Simulation::SaveScreenshot)

      // TODO: Deprecate the legacy methods below.
      .def("saveScreenshot", &Simulation::SaveScreenshot)
      .def("getScenario", &Simulation::GetScenario,
           py::return_value_policy::reference);
}

}  // namespace nocturne
