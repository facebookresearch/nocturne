// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "road.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>

namespace py = pybind11;

namespace nocturne {

void DefineRoadPoint(py::module& m) {
    py::class_<RoadPoint, std::shared_ptr<RoadPoint>>(m, "RoadPoint")
        .def(py::init<>())  // Default constructor
        .def(py::init<const geometry::Vector2D&, const geometry::Vector2D&, RoadType>(),
             py::arg("position"), py::arg("neighbor_position"), py::arg("road_type"))
        .def_property_readonly("road_type", &RoadPoint::road_type)
        .def_property_readonly("position", &RoadPoint::position)
        .def_property_readonly("neighbor_position", &RoadPoint::neighbor_position)
        .def("Coordinate", &RoadPoint::Coordinate)
        .def("X", &RoadPoint::X)
        .def("Y", &RoadPoint::Y);
}

void DefineRoadLine(py::module& m) {
  py::enum_<RoadType>(m, "RoadType")
      .value("NONE", RoadType::kNone)
      .value("LANE", RoadType::kLane)
      .value("ROAD_LINE", RoadType::kRoadLine)
      .value("ROAD_EDGE", RoadType::kRoadEdge)
      .value("STOP_SIGN", RoadType::kStopSign)
      .value("CROSSWALK", RoadType::kCrosswalk)
      .value("SPEED_BUMP", RoadType::kSpeedBump)
      .value("OTHER", RoadType::kOther)
      .export_values();

  py::class_<RoadLine, std::shared_ptr<RoadLine>>(m, "RoadLine")
      .def_property_readonly("road_type", &RoadLine::road_type)
      .def_property_readonly("check_collision", &RoadLine::check_collision)
      .def("geometry_points", &RoadLine::geometry_points)
      .def("road_points", &RoadLine::road_points, py::return_value_policy::reference_internal)

      // TODO: Deprecates the legacy methods below.
      .def("getGeometry", &RoadLine::geometry_points)
      .def("canCollide", &RoadLine::check_collision);
}

}  // namespace nocturne
