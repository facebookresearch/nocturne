// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/intersection.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <limits>

namespace nocturne {
namespace geometry {

namespace {

constexpr int kInside = 0;
constexpr int kLeft = 1;
constexpr int kRight = 2;
constexpr int kBottom = 4;
constexpr int kTop = 8;

int ComputeOutCode(const AABB& aabb, const Vector2D& p) {
  int code = kInside;
  if (p.x() < aabb.MinX()) {
    code |= kLeft;
  } else if (p.x() > aabb.MaxX()) {
    code |= kRight;
  }
  if (p.y() < aabb.MinY()) {
    code |= kBottom;
  } else if (p.y() > aabb.MaxY()) {
    code |= kTop;
  }
  return code;
}

template <int64_t N>
std::vector<utils::MaskType> SmallPolygonBatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<float>& x, const std::vector<float>& y) {
  assert(x.size() == y.size());
  const int64_t n = x.size();
  std::vector<utils::MaskType> mask(n, 1);

  const auto [pvx, pvy] = utils::PackSmallPolygon<N>(polygon);
  const float ox = o.x();
  const float oy = o.y();

  for (int64_t i = 0; i < n; ++i) {
    const float dx = x[i] - ox;
    const float dy = y[i] - oy;
    float min_v = std::numeric_limits<float>::max();
    float max_v = std::numeric_limits<float>::lowest();
    for (int64_t i = 0; i < N; ++i) {
      const float vx = pvx[i] - ox;
      const float vy = pvy[i] - oy;
      const float cur = vx * dy - dx * vy;
      // std::min and std::max are slow, use conditional operator here.
      min_v = min_v < cur ? min_v : cur;
      max_v = max_v > cur ? max_v : cur;
    }
    // Use bitwise operation to get better performance.
    // Use (^1) for not operation.
    mask[i] &=
        ((utils::MaskType(max_v < 0.0f) | utils::MaskType(min_v > 0.0f)) ^ 1);
  }

  for (int64_t i = 0; i < n; ++i) {
    float cur_v = std::numeric_limits<float>::lowest();
    for (int64_t j = 0; j < N; ++j) {
      const float p0x = pvx[j];
      const float p0y = pvy[j];
      const float p1x = pvx[(j + 1) % N];
      const float p1y = pvy[(j + 1) % N];
      const float dx = p1x - p0x;
      const float dy = p1y - p0y;
      const float v0x = ox - p0x;
      const float v0y = oy - p0y;
      const float v1x = x[i] - p0x;
      const float v1y = y[i] - p0y;
      const float v0 = v0x * dy - dx * v0y;
      const float v1 = v1x * dy - dx * v1y;
      const float v = v0 < v1 ? v0 : v1;
      cur_v = cur_v > v ? cur_v : v;
    }
    // Use bitwise operation to get better performance.
    // Use (^1) for not operation.
    mask[i] &= (utils::MaskType(cur_v > 0.0f) ^ 1);
  }

  return mask;
}

template <int64_t N>
std::vector<float> SmallPolygonBatchParametricIntersection(
    const Vector2D& o, const std::vector<float>& x, const std::vector<float>& y,
    const ConvexPolygon& polygon) {
  assert(x.size() == y.size());
  const int64_t n = x.size();
  std::vector<float> ret(n, std::numeric_limits<float>::infinity());

  const auto [pvx, pvy] = utils::PackSmallPolygon<N>(polygon);
  const float p0x = o.x();
  const float p0y = o.y();

  for (int64_t i = 0; i < n; ++i) {
    const float q0x = x[i];
    const float q0y = y[i];
    const float p0q0x = q0x - p0x;
    const float p0q0y = q0y - p0y;

    for (int64_t j = 0; j < N; ++j) {
      const float p1x = pvx[j];
      const float p1y = pvy[j];
      const float q1x = pvx[(j + 1) % N];
      const float q1y = pvy[(j + 1) % N];
      const float p1q1x = q1x - p1x;
      const float p1q1y = q1y - p1y;

      // Use bitwise operation to get better performance.
      const utils::MaskType intersects =
          (utils::MaskType(CCW(p0x, p0y, q0x, q0y, p1x, p1y) !=
                           CCW(p0x, p0y, q0x, q0y, q1x, q1y)) &
           utils::MaskType(CCW(p1x, p1y, q1x, q1y, p0x, p0y) !=
                           CCW(p1x, p1y, q1x, q1y, q0x, q0y)));

      const float c0 = p0x * p1q1y - p1q1x * p0y;
      const float c1 = p1x * p1q1y - p1q1x * p1y;
      const float cd = p0q0x * p1q1y - p1q1x * p0q0y;
      const float cur =
          intersects ? (c1 - c0) / cd : std::numeric_limits<float>::infinity();

      ret[i] = cur < ret[i] ? cur : ret[i];
    }
  }

  return ret;
}

}  // namespace

// Cohen–Sutherland algorithm
// https://en.wikipedia.org/wiki/Cohen%E2%80%93Sutherland_algorithm
bool Intersects(const AABB& aabb, const LineSegment& segment) {
  const float min_x = aabb.MinX();
  const float min_y = aabb.MinY();
  const float max_x = aabb.MaxX();
  const float max_y = aabb.MaxY();

  float x0 = segment.Endpoint0().x();
  float y0 = segment.Endpoint0().y();
  float x1 = segment.Endpoint1().x();
  float y1 = segment.Endpoint1().y();
  int code0 = ComputeOutCode(aabb, segment.Endpoint0());
  int code1 = ComputeOutCode(aabb, segment.Endpoint1());

  while (true) {
    if ((code0 | code1) == 0) {
      return true;
    }
    if ((code0 & code1) != 0) {
      return false;
    }
    const int code = std::max(code0, code1);
    float x = 0;
    float y = 0;
    if ((code & kTop) != 0) {
      x = x0 + (x1 - x0) * (max_y - y0) / (y1 - y0);
      y = max_y;
    } else if ((code & kBottom) != 0) {
      x = x0 + (x1 - x0) * (min_y - y0) / (y1 - y0);
      y = min_y;
    } else if ((code & kRight) != 0) {
      y = y0 + (y1 - y0) * (max_x - x0) / (x1 - x0);
      x = max_x;
    } else if ((code & kLeft) != 0) {
      y = y0 + (y1 - y0) * (min_x - x0) / (x1 - x0);
      x = min_x;
    }
    if (code == code0) {
      x0 = x;
      y0 = y;
      code0 = ComputeOutCode(aabb, Vector2D(x0, y0));
    } else {
      x1 = x;
      y1 = y;
      code1 = ComputeOutCode(aabb, Vector2D(x1, y1));
    }
  }
  return false;
}

bool Intersects(const LineSegment& segment, const AABB& aabb) {
  return Intersects(aabb, segment);
}

// Assume the vertices of polygon are in counterclockwise order.
bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment) {
  if (segment.Endpoint0() == segment.Endpoint1()) {
    return polygon.Contains(segment.Endpoint0());
  }

  // Check if polygon lies on the same side of segment.
  const Vector2D d = segment.Endpoint1() - segment.Endpoint0();
  float min_v = std::numeric_limits<float>::max();
  float max_v = std::numeric_limits<float>::lowest();
  for (const Vector2D& p : polygon.vertices()) {
    const float cur = CrossProduct(p - segment.Endpoint0(), d);
    min_v = std::min(min_v, cur);
    max_v = std::max(max_v, cur);
  }
  if (max_v < 0.0f || min_v > 0.0f) {
    return false;
  }

  // Check if segment lies on the right of one of the edges of polygon.
  const std::vector<LineSegment> edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    const Vector2D cur_d = edge.Endpoint1() - edge.Endpoint0();
    const float v0 =
        CrossProduct(segment.Endpoint0() - edge.Endpoint0(), cur_d);
    const float v1 =
        CrossProduct(segment.Endpoint1() - edge.Endpoint0(), cur_d);
    if (v0 > 0.0f && v1 > 0.0f) {
      return false;
    }
  }

  return true;
}

bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon) {
  return Intersects(polygon, segment);
}

std::vector<utils::MaskType> BatchIntersects(const ConvexPolygon& polygon,
                                             const Vector2D& o,
                                             const std::vector<float>& x,
                                             const std::vector<float>& y) {
  const int64_t m = polygon.NumEdges();
  if (m == 3) {
    return SmallPolygonBatchIntersects<3>(polygon, o, x, y);
  }
  if (m == 4) {
    return SmallPolygonBatchIntersects<4>(polygon, o, x, y);
  }
  if (m == 5) {
    return SmallPolygonBatchIntersects<5>(polygon, o, x, y);
  }
  if (m == 6) {
    return SmallPolygonBatchIntersects<6>(polygon, o, x, y);
  }

  assert(x.size() == y.size());
  const int64_t n = x.size();
  std::vector<utils::MaskType> mask(n, 1);
  std::vector<float> min_v(n, std::numeric_limits<float>::max());
  std::vector<float> max_v(n, std::numeric_limits<float>::lowest());

  const float ox = o.x();
  const float oy = o.y();

  for (const Vector2D& v : polygon.vertices()) {
    const float vx = v.x() - ox;
    const float vy = v.y() - oy;
    for (int64_t i = 0; i < n; ++i) {
      const float dx = x[i] - ox;
      const float dy = y[i] - oy;
      const float cur = vx * dy - dx * vy;
      // std::min and std::max are slow, use conditional operator here.
      min_v[i] = min_v[i] < cur ? min_v[i] : cur;
      max_v[i] = max_v[i] > cur ? max_v[i] : cur;
    }
  }
  for (int64_t i = 0; i < n; ++i) {
    // Use bitwise operation to get better performance.
    // Use (^1) for not operation.
    mask[i] &=
        ((utils::MaskType(max_v[i] < 0.0f) | utils::MaskType(min_v[i] > 0.0f)) ^
         1);
  }

  const std::vector<LineSegment> edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    const float p0x = edge.Endpoint0().x();
    const float p0y = edge.Endpoint0().y();
    const float p1x = edge.Endpoint1().x();
    const float p1y = edge.Endpoint1().y();
    const float dx = p1x - p0x;
    const float dy = p1y - p0y;
    const float v0x = ox - p0x;
    const float v0y = oy - p0y;
    for (int64_t i = 0; i < n; ++i) {
      const float v1x = x[i] - p0x;
      const float v1y = y[i] - p0y;
      const float v0 = v0x * dy - dx * v0y;
      const float v1 = v1x * dy - dx * v1y;
      // Use bitwise operation to get better performance.
      // Use (^1) for not operation.
      mask[i] &=
          ((utils::MaskType(v0 > 0.0f) & utils::MaskType(v1 > 0.0f)) ^ 1);
    }
  }

  return mask;
}

std::vector<utils::MaskType> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<Vector2D>& points) {
  const auto [x, y] = utils::PackCoordinates(points);
  return BatchIntersects(polygon, o, x, y);
}

std::vector<utils::MaskType> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<const PointLike*>& points) {
  const auto [x, y] = utils::PackCoordinates(points);
  return BatchIntersects(polygon, o, x, y);
}

std::vector<float> BatchParametricIntersection(const Vector2D& o,
                                               const std::vector<float>& x,
                                               const std::vector<float>& y,
                                               const ConvexPolygon& polygon) {
  const int64_t m = polygon.NumEdges();
  if (m == 3) {
    return SmallPolygonBatchParametricIntersection<3>(o, x, y, polygon);
  }
  if (m == 4) {
    return SmallPolygonBatchParametricIntersection<4>(o, x, y, polygon);
  }
  if (m == 5) {
    return SmallPolygonBatchParametricIntersection<5>(o, x, y, polygon);
  }
  if (m == 6) {
    return SmallPolygonBatchParametricIntersection<6>(o, x, y, polygon);
  }

  assert(x.size() == y.size());
  const int64_t n = x.size();
  std::vector<float> ret(n, std::numeric_limits<float>::infinity());

  const float p0x = o.x();
  const float p0y = o.y();

  const auto edges = polygon.Edges();
  for (const auto& e : edges) {
    const float p1x = e.Endpoint0().x();
    const float p1y = e.Endpoint0().y();
    const float q1x = e.Endpoint1().x();
    const float q1y = e.Endpoint1().y();

    const float p1q1x = q1x - p1x;
    const float p1q1y = q1y - p1y;

    for (int64_t i = 0; i < n; ++i) {
      const float q0x = x[i];
      const float q0y = y[i];

      const float p0q0x = q0x - p0x;
      const float p0q0y = q0y - p0y;

      // Use bitwise operation to get better performance.
      const utils::MaskType intersects =
          (utils::MaskType(CCW(p0x, p0y, q0x, q0y, p1x, p1y) !=
                           CCW(p0x, p0y, q0x, q0y, q1x, q1y)) &
           utils::MaskType(CCW(p1x, p1y, q1x, q1y, p0x, p0y) !=
                           CCW(p1x, p1y, q1x, q1y, q0x, q0y)));

      const float c0 = p0x * p1q1y - p1q1x * p0y;
      const float c1 = p1x * p1q1y - p1q1x * p1y;
      const float cd = p0q0x * p1q1y - p1q1x * p0q0y;
      const float cur =
          intersects ? (c1 - c0) / cd : std::numeric_limits<float>::infinity();

      ret[i] = cur < ret[i] ? cur : ret[i];
    }
  }

  return ret;
}

std::vector<float> BatchParametricIntersection(
    const Vector2D& o, const std::vector<Vector2D>& points,
    const ConvexPolygon& polygon) {
  const auto [x, y] = utils::PackCoordinates(points);
  return BatchParametricIntersection(o, x, y, polygon);
}

}  // namespace geometry
}  // namespace nocturne
