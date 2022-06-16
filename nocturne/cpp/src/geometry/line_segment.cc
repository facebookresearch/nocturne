// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/line_segment.h"

namespace nocturne {
namespace geometry {

bool LineSegment::Intersects(const LineSegment& seg) const {
  const Vector2D& p1 = endpoints_[0];
  const Vector2D& q1 = endpoints_[1];
  const Vector2D& p2 = seg.endpoints_[0];
  const Vector2D& q2 = seg.endpoints_[1];
  return CCW(p1, q1, p2) != CCW(p1, q1, q2) &&
         CCW(p2, q2, p1) != CCW(p2, q2, q1);
}

std::optional<Vector2D> LineSegment::Intersection(
    const LineSegment& seg) const {
  if (!Intersects(seg)) {
    return std::nullopt;
  }
  const Vector2D& p0 = Endpoint0();
  const Vector2D& q0 = Endpoint1();
  const Vector2D& p1 = seg.Endpoint0();
  const Vector2D& q1 = seg.Endpoint1();
  const Vector2D d0 = q0 - p0;
  const Vector2D d1 = q1 - p1;
  const float c0 = CrossProduct(p0, d0);
  const float c1 = CrossProduct(p1, d1);
  const float x = d0.x() * c1 - d1.x() * c0;
  const float y = d0.y() * c1 - d1.y() * c0;
  return std::make_optional(Vector2D(x, y) / CrossProduct(d0, d1));
}

std::optional<float> LineSegment::ParametricIntersection(
    const LineSegment& seg) const {
  if (!Intersects(seg)) {
    return std::nullopt;
  }
  const Vector2D& p0 = Endpoint0();
  const Vector2D& q0 = Endpoint1();
  const Vector2D& p1 = seg.Endpoint0();
  const Vector2D& q1 = seg.Endpoint1();
  const Vector2D d0 = q0 - p0;
  const Vector2D d1 = q1 - p1;
  const float c0 = CrossProduct(p0, d1);
  const float c1 = CrossProduct(p1, d1);
  return std::make_optional((c1 - c0) / CrossProduct(d0, d1));
}

}  // namespace geometry
}  // namespace nocturne
