// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <iostream>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

class AABB {
 public:
  AABB(const Vector2D& p_min, const Vector2D& p_max)
      : min_(p_min), max_(p_max) {}
  AABB(float min_x, float min_y, float max_x, float max_y)
      : min_(min_x, min_y), max_(max_x, max_y) {}
  AABB(const AABB& aabb) : min_(aabb.min()), max_(aabb.max()) {}

  const Vector2D& min() const { return min_; }
  const Vector2D& max() const { return max_; }

  AABB operator||(const AABB& other) const { return Union(other); }

  float MinX() const { return min_.x(); }
  float MinY() const { return min_.y(); }

  float MaxX() const { return max_.x(); }
  float MaxY() const { return max_.y(); }

  Vector2D Center() const { return (min_ + max_) * 0.5f; }

  float Area() const { return (MaxX() - MinX()) * (MaxY() - MinY()); }

  bool Contains(const Vector2D& vec) const {
    return MinX() <= vec.x() && MaxX() >= vec.x() && MinY() <= vec.y() &&
           MaxY() >= vec.y();
  }

  bool Contains(const AABB& other) const {
    return MinX() <= other.MinX() && MaxX() >= other.MaxX() &&
           MinY() <= other.MinY() && MaxY() >= other.MaxY();
  }

  bool Intersects(const AABB& other) const {
    return MinX() < other.MaxX() && MaxX() > other.MinX() &&
           MinY() < other.MaxY() && MaxY() > other.MinY();
  }

  AABB Union(const AABB& other) const {
    const float min_x = std::min(MinX(), other.MinX());
    const float min_y = std::min(MinY(), other.MinY());
    const float max_x = std::max(MaxX(), other.MaxX());
    const float max_y = std::max(MaxY(), other.MaxY());
    return AABB(min_x, min_y, max_x, max_y);
  }

 protected:
  Vector2D min_;
  Vector2D max_;
};

inline float Distance(const AABB& a, const AABB& b) { return (a || b).Area(); }

}  // namespace geometry
}  // namespace nocturne
