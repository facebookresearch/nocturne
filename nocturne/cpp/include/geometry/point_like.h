// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class PointLike {
 public:
  virtual Vector2D Coordinate() const = 0;

  virtual float X() const { return Coordinate().x(); }
  virtual float Y() const { return Coordinate().y(); }
};

}  // namespace geometry
}  // namespace nocturne
