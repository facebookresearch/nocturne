// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <string>

#include "geometry/aabb.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "static_object.h"

namespace nocturne {

constexpr float kStopSignRadius = 2.0f;
constexpr int kStopSignNumEdges = 6;

class StopSign : public StaticObject {
 public:
  StopSign() = default;
  explicit StopSign(const geometry::Vector2D& position)
      : StaticObject(position,
                     /*can_block_sight=*/false,
                     /*can_be_collided=*/false, /*check_collision=*/false) {}

  StaticObjectType Type() const override { return StaticObjectType::kStopSign; }

  float Radius() const { return kStopSignRadius; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  sf::Color Color() const { return sf::Color::Red; }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;
};

}  // namespace nocturne
