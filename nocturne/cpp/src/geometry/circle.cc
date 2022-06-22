// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/circle.h"

namespace nocturne {
namespace geometry {

std::pair<std::optional<Vector2D>, std::optional<Vector2D>>
CircleLike::Intersection(const LineSegment& segment) const {
  std::array<Vector2D, 2> ret;
  int64_t cnt = 0;

  const Vector2D& o = center();
  const Vector2D& p = segment.Endpoint0();
  const Vector2D& q = segment.Endpoint1();
  const Vector2D d1 = q - p;
  const Vector2D d2 = p - o;
  const float r = radius();
  const float a = DotProduct(d1, d1);
  const float b = DotProduct(d1, d2) * 2.0f;
  const float c = DotProduct(d2, d2) - r * r;
  const float delta = b * b - 4.0f * a * c;
  if (utils::AlmostEquals(delta, 0.0f)) {
    const float t = -b / (2.0f * a);
    if (t >= 0.0f && t <= 1.0f) {
      ret[cnt++] = segment.Point(t);
    }
  } else if (delta > 0.0f) {
    const float t0 = (-b - std::sqrt(delta)) / (2.0f * a);
    const float t1 = (-b + std::sqrt(delta)) / (2.0f * a);
    if (t0 >= 0.0f && t0 <= 1.0f) {
      ret[cnt++] = segment.Point(t0);
    }
    if (t1 >= 0.0f && t1 <= 1.0f) {
      ret[cnt++] = segment.Point(t1);
    }
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

std::vector<utils::MaskType> Circle::BatchContains(
    const std::vector<const PointLike*>& points) const {
  const int64_t n = points.size();
  std::vector<utils::MaskType> mask(n);
  const auto [x, y] = utils::PackCoordinates(points);
  const float ox = center_.x();
  const float oy = center_.y();
  for (int64_t i = 0; i < n; ++i) {
    const float dx = x[i] - ox;
    const float dy = y[i] - oy;
    mask[i] = (dx * dx + dy * dy <= radius_ * radius_);
  }
  return mask;
}

}  // namespace geometry
}  // namespace nocturne
