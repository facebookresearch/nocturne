// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {

enum class StaticObjectType {
  kUnset = 0,
  kTrafficLight = 1,
  kStopSign = 2,
  kOther = 3,
};

class StaticObject : public ObjectBase {
 public:
  StaticObject() = default;
  explicit StaticObject(const geometry::Vector2D& position)
      : ObjectBase(position) {}
  StaticObject(const geometry::Vector2D& position, bool can_block_sight,
               bool can_be_collided, bool check_collision)
      : ObjectBase(position, can_block_sight, can_be_collided,
                   check_collision) {}

  virtual StaticObjectType Type() const { return StaticObjectType::kUnset; }
};

inline StaticObjectType ParseStaticObjectType(const std::string& type) {
  if (type == "unset") {
    return StaticObjectType::kUnset;
  } else if (type == "traffic_light") {
    return StaticObjectType::kTrafficLight;
  } else if (type == "stop_sign") {
    return StaticObjectType::kStopSign;
  } else {
    return StaticObjectType::kOther;
  }
}

}  // namespace nocturne
