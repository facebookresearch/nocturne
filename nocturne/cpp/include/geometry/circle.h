// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/point_like.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class CircleLike : public AABBInterface {
 public:
  CircleLike() = default;
  CircleLike(const Vector2D& center, float radius)
      : center_(center), radius_(radius) {}

  const Vector2D& center() const { return center_; }
  float radius() const { return radius_; }

  virtual float Area() const = 0;
  virtual bool Contains(const Vector2D& p) const = 0;
  virtual std::pair<std::optional<Vector2D>, std::optional<Vector2D>>
  Intersection(const LineSegment& segment) const;

  virtual std::vector<utils::MaskType> BatchContains(
      const std::vector<const PointLike*>& points) const = 0;

 protected:
  const Vector2D center_;
  const float radius_;
};

class Circle : public CircleLike {
 public:
  Circle() = default;
  Circle(const Vector2D& center, float radius) : CircleLike(center, radius) {}

  AABB GetAABB() const override {
    return AABB(center_ - radius_, center_ + radius_);
  }

  float Area() const override { return utils::kPi * radius_ * radius_; }

  bool Contains(const Vector2D& p) const override {
    const Vector2D d = p - center_;
    const float dx = d.x();
    const float dy = d.y();
    return dx * dx + dy * dy <= radius_ * radius_;
  }

  std::vector<utils::MaskType> BatchContains(
      const std::vector<const PointLike*>& points) const override;
};

}  // namespace geometry
}  // namespace nocturne
