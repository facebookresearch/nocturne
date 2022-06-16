// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/circular_sector.h"

#include <algorithm>
#include <array>
#include <cassert>

namespace nocturne {
namespace geometry {

namespace {

void CheckMinMaxCoordinates(const Vector2D& p, float& min_x, float& min_y,
                            float& max_x, float& max_y) {
  min_x = std::min(min_x, p.x());
  min_y = std::min(min_y, p.y());
  max_x = std::max(max_x, p.x());
  max_y = std::max(max_y, p.y());
}

}  // namespace

AABB CircularSector::GetAABB() const {
  const Vector2D p0 = center_ + Radius0();
  const Vector2D p1 = center_ + Radius1();
  float min_x = std::min({center_.x(), p0.x(), p1.x()});
  float min_y = std::min({center_.y(), p0.y(), p1.y()});
  float max_x = std::max({center_.x(), p0.x(), p1.x()});
  float max_y = std::max({center_.y(), p0.y(), p1.y()});

  // TODO: Optimize this.
  const Vector2D q0 = center_ + Vector2D(radius_, 0.0f);
  if (CenterAngleContains(q0)) {
    CheckMinMaxCoordinates(q0, min_x, min_y, max_x, max_y);
  }
  const Vector2D q1 = center_ + Vector2D(0.0f, radius_);
  if (CenterAngleContains(q1)) {
    CheckMinMaxCoordinates(q1, min_x, min_y, max_x, max_y);
  }
  const Vector2D q2 = center_ - Vector2D(radius_, 0.0f);
  if (CenterAngleContains(q2)) {
    CheckMinMaxCoordinates(q2, min_x, min_y, max_x, max_y);
  }
  const Vector2D q3 = center_ - Vector2D(0.0f, radius_);
  if (CenterAngleContains(q3)) {
    CheckMinMaxCoordinates(q3, min_x, min_y, max_x, max_y);
  }

  return AABB(min_x, min_y, max_x, max_y);
}

bool CircularSector::Contains(const Vector2D& p) const {
  const Vector2D d = p - center_;
  const float dx = d.x();
  const float dy = d.y();
  return dx * dx + dy * dy <= radius_ * radius_ && CenterAngleContains(p);
}

std::vector<utils::MaskType> CircularSector::BatchContains(
    const std::vector<const PointLike*>& points) const {
  const int64_t n = points.size();
  std::vector<utils::MaskType> mask(n);
  const auto [x, y] = utils::PackCoordinates(points);
  const Vector2D r0 = Radius0();
  const Vector2D r1 = Radius1();
  const float ox = center_.x();
  const float oy = center_.y();
  const float r0x = r0.x();
  const float r0y = r0.y();
  const float r1x = r1.x();
  const float r1y = r1.y();
  for (int64_t i = 0; i < n; ++i) {
    const float dx = x[i] - ox;
    const float dy = y[i] - oy;
    const float r2 = dx * dx + dy * dy;
    const float c0 = dx * r0y - r0x * dy;
    const float c1 = dx * r1y - r1x * dy;
    // Use bitwise operation to get better performance.
    const utils::MaskType m = theta_ < 0.0f ? (utils::MaskType(c0 <= 0.0f) | utils::MaskType(c1 >= 0.0f))
                                            : (utils::MaskType(c0 <= 0.0f) & utils::MaskType(c1 >= 0.0f));
    mask[i] = ((r2 <= radius_ * radius_) & m);
  }
  return mask;
}

std::pair<std::optional<Vector2D>, std::optional<Vector2D>>
CircularSector::Intersection(const LineSegment& segment) const {
  std::array<Vector2D, 2> ret;
  int64_t cnt = 0;

  const Vector2D& o = center();
  const LineSegment edge0(o, o + Radius0());
  const LineSegment edge1(o, o + Radius1());
  const auto u = edge0.Intersection(segment);
  if (u.has_value()) {
    ret[cnt++] = *u;
  }
  const auto v = edge1.Intersection(segment);
  if (v.has_value()) {
    ret[cnt++] = *v;
  }
  if (cnt == 2) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::make_optional(ret[1]));
  }

  auto [p, q] = CircleLike::Intersection(segment);
  if (p.has_value() && CenterAngleContains(*p)) {
    ret[cnt++] = *p;
  }
  if (q.has_value() && CenterAngleContains(*q)) {
    ret[cnt++] = *q;
  }

  if (cnt == 0) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::nullopt, std::nullopt);
  } else if (cnt == 1) {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::nullopt);
  } else {
    return std::make_pair<std::optional<Vector2D>, std::optional<Vector2D>>(
        std::make_optional(ret[0]), std::make_optional(ret[1]));
  }
}

bool CircularSector::CenterAngleContains(const Vector2D& p) const {
  const Vector2D d = p - center_;
  const Vector2D r0 = Radius0();
  const Vector2D r1 = Radius1();
  return theta_ < 0.0f
             ? (CrossProduct(d, r0) <= 0.0f || CrossProduct(d, r1) >= 0.0f)
             : (CrossProduct(d, r0) <= 0.0f && CrossProduct(d, r1) >= 0.0f);
}

}  // namespace geometry
}  // namespace nocturne
