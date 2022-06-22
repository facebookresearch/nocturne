// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <optional>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class LineSegment : public AABBInterface {
 public:
  LineSegment() = default;
  LineSegment(const Vector2D& p, const Vector2D& q) : endpoints_({p, q}) {}
  LineSegment(const LineSegment& seg) : endpoints_(seg.endpoints_) {}

  const Vector2D& Endpoint0() const { return endpoints_[0]; }
  const Vector2D& Endpoint1() const { return endpoints_[1]; }
  const Vector2D& Endpoint(int64_t index) const { return endpoints_.at(index); }

  float Length() const { return Distance(endpoints_[0], endpoints_[1]); }

  AABB GetAABB() const override {
    const float min_x = std::min(endpoints_[0].x(), endpoints_[1].x());
    const float max_x = std::max(endpoints_[0].x(), endpoints_[1].x());
    const float min_y = std::min(endpoints_[0].y(), endpoints_[1].y());
    const float max_y = std::max(endpoints_[0].y(), endpoints_[1].y());
    return AABB(min_x, min_y, max_x, max_y);
  }

  // Returns r(t) = (1 - t) * P0 + t * P1
  Vector2D Point(float t) const {
    assert(t >= 0.0f && t <= 1.0f);
    return Endpoint0() * (1.0f - t) + Endpoint1() * t;
  }

  Vector2D NormalVector() const {
    const Vector2D d = endpoints_[1] - endpoints_[0];
    return Vector2D(-d.y(), d.x()) / d.Norm();
  }

  bool Contains(const Vector2D& p) const {
    const Vector2D u = endpoints_[0] - p;
    const Vector2D v = endpoints_[1] - p;
    // return CrossProduct(u, v) == 0.0f && DotProduct(u, v) <= 0.0f;
    return utils::AlmostEquals(CrossProduct(u, v), 0.0f) &&
           DotProduct(u, v) <= 0.0f;
  }

  bool Intersects(const LineSegment& seg) const;

  // Returns the intersection point if *this intersects with seg.
  std::optional<Vector2D> Intersection(const LineSegment& seg) const;

  // Parametric equation for *this: r(t) = (1 - t) * P0 + t * P1
  // where t is in [0, 1].
  // Returns the intersection point parameter t if *this intersects with seg.
  std::optional<float> ParametricIntersection(const LineSegment& seg) const;

 protected:
  const std::array<Vector2D, 2> endpoints_;
};

inline bool CCW(const Vector2D& a, const Vector2D& b, const Vector2D& c) {
  return CrossProduct(b - a, c - a) > 0.0f;
}

constexpr bool CCW(float ax, float ay, float bx, float by, float cx, float cy) {
  return (bx - ax) * (cy - ay) - (cx - ax) * (by - ay) > 0.0f;
}

}  // namespace geometry
}  // namespace nocturne
