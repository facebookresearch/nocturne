// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "object.h"

#include <pybind11/pybind11.h>

#include <memory>

#include "geometry/vector_2d.h"
#include "object_base.h"

namespace py = pybind11;

namespace nocturne {

void DefineObject(py::module& m) {
  py::enum_<CollisionType>(m, "CollisionType")
      .value("UNCOLLIDED", CollisionType::kNotCollided)
      .value("VEHICLE_VEHICLE", CollisionType::kVehicleVehicleCollision)
      .value("VEHICLE_ROAD", CollisionType::kVehicleRoadEdgeCollision)
      .export_values();

  py::enum_<ObjectType>(m, "ObjectType")
      .value("UNSET", ObjectType::kUnset)
      .value("VEHICLE", ObjectType::kVehicle)
      .value("PEDESTRIAN", ObjectType::kPedestrian)
      .value("CYCLIST", ObjectType::kCyclist)
      .value("OTHER", ObjectType::kOther)
      .export_values();

  py::class_<Object, std::shared_ptr<Object>>(m, "Object")
      // Properties.
      .def_property_readonly("type", &Object::Type)
      .def_property_readonly("id", &Object::id)
      .def_property_readonly("length", &Object::length)
      .def_property_readonly("width", &Object::width)
      .def_property_readonly("max_speed", &Object::max_speed)
      .def_property(
          "position", &Object::position,
          py::overload_cast<const geometry::Vector2D&>(&Object::set_position))
      .def_property("heading", &Object::heading, &Object::set_heading)
      .def_property("speed", &Object::speed, &Object::set_speed)
      .def_property("target_position", &Object::target_position,
                    py::overload_cast<const geometry::Vector2D&>(
                        &Object::set_target_position))
      .def_property("target_heading", &Object::target_heading,
                    &Object::set_target_heading)
      .def_property("target_speed", &Object::target_speed,
                    &Object::set_target_speed)
      .def_property("acceleration", &Object::acceleration,
                    &Object::set_acceleration)
      .def_property("steering", &Object::steering, &Object::set_steering)
      .def_property("head_angle", &Object::head_angle, &Object::set_head_angle)
      .def_property("manual_control", &Object::manual_control,
                    &Object::set_manual_control)
      .def_property("expert_control", &Object::expert_control,
                    &Object::set_expert_control)
      .def_property("highlight", &Object::highlight, &Object::set_highlight)
      .def_property_readonly("collided", &Object::collided)
      .def_property_readonly("collision_type", &Object::collision_type)

      // Methods.
      .def("velocity", &Object::Velocity)
      .def("set_position",
           py::overload_cast<float, float>(&Object::set_position))
      .def("set_target_position",
           py::overload_cast<float, float>(&Object::set_target_position))
      .def("apply_action", &Object::ApplyAction)
      .def("_scale_shape", &Object::ScaleShape, py::arg("length_scale") = 1.0,
           py::arg("width_scale") = 1.0)

      // TODO: Deprecate the legacy interfaces below.
      .def("getWidth", &Object::width)
      .def("getLength", &Object::length)
      .def("getPosition", &Object::position)
      .def("getGoalPosition", &Object::target_position)
      .def("getSpeed", &Object::speed)
      .def("getHeading", &Object::heading)
      .def("getID", &Object::id)
      .def("getType", &Object::Type)
      .def("getCollided", &Object::collided)
      .def("setPosition",
           py::overload_cast<const geometry::Vector2D&>(&Object::set_position))
      .def("setPosition",
           py::overload_cast<float, float>(&Object::set_position))
      .def("setGoalPosition", py::overload_cast<const geometry::Vector2D&>(
                                  &Object::set_target_position))
      .def("setGoalPosition",
           py::overload_cast<float, float>(&Object::set_target_position))
      .def("setHeading", &Object::set_heading)
      .def("setSpeed", &Object::set_speed);
}

}  // namespace nocturne
