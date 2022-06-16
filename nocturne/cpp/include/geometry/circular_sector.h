// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/circle.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/point_like.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class CircularSector : public CircleLike {
 public:
  CircularSector() = default;
  CircularSector(const Vector2D& center, float radius, float heading,
                 float theta)
      : CircleLike(center, radius),
        heading_(utils::NormalizeAngle<float>(heading)),
        theta_(utils::NormalizeAngle<float>(theta)) {}

  float heading() const { return heading_; }
  float theta() const { return theta_; }

  float Angle0() const {
    return theta_ < 0.0f
               ? utils::AngleSub<float>(heading_, theta_ * 0.5f + utils::kPi)
               : utils::AngleSub<float>(heading_, theta_ * 0.5f);
  }
  float Angle1() const {
    return theta_ < 0.0f
               ? utils::AngleAdd<float>(heading_, theta_ * 0.5f + utils::kPi)
               : utils::AngleAdd<float>(heading_, theta_ * 0.5f);
  }

  Vector2D Radius0() const { return PolarToVector2D(radius_, Angle0()); }
  Vector2D Radius1() const { return PolarToVector2D(radius_, Angle1()); }

  AABB GetAABB() const override;

  float Area() const override {
    const float theta = theta_ < 0.0f ? theta_ + utils::kTwoPi : theta_;
    return theta * radius_ * radius_ * 0.5f;
  }

  bool Contains(const Vector2D& p) const override;

  std::vector<utils::MaskType> BatchContains(
      const std::vector<const PointLike*>& points) const override;

  std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
      const LineSegment& segment) const override;

 protected:
  bool CenterAngleContains(const Vector2D& p) const;

  const float heading_;
  const float theta_;
};

}  // namespace geometry
}  // namespace nocturne
