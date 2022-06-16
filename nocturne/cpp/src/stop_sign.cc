// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "stop_sign.h"

#include <vector>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon StopSign::BoundingPolygon() const {
  constexpr float kTheta = geometry::utils::kTwoPi / kStopSignNumEdges;
  float angle = geometry::utils::kHalfPi;
  std::vector<geometry::Vector2D> vertices;
  vertices.reserve(kStopSignNumEdges);
  for (int i = 0; i < kStopSignNumEdges; ++i) {
    vertices.push_back(position_ +
                       geometry::PolarToVector2D(kStopSignRadius, angle));
    angle = geometry::utils::AngleAdd(angle, kTheta);
  }
  return geometry::ConvexPolygon(std::move(vertices));
}

void StopSign::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  sf::CircleShape hexagon(kStopSignRadius, kStopSignNumEdges);
  hexagon.setFillColor(Color());
  hexagon.setPosition(utils::ToVector2f(position_));
  target.draw(hexagon, states);
}

}  // namespace nocturne
