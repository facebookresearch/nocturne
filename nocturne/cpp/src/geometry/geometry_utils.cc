// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/geometry_utils.h"

#include <cassert>

#include "geometry/point_like.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace utils {

std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<Vector2D>& points) {
  const int64_t n = points.size();
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int64_t i = 0; i < n; ++i) {
    x[i] = points[i].x();
    y[i] = points[i].y();
  }
  return std::make_pair(x, y);
}

std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<const PointLike*>& points) {
  const int64_t n = points.size();
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (int64_t i = 0; i < n; ++i) {
    const Vector2D p = points[i]->Coordinate();
    x[i] = p.x();
    y[i] = p.y();
  }
  return std::make_pair(x, y);
}

#define NOCTURNE_DEFINE_PACK_SMALL_POLYGON(N)                                \
  template <>                                                                \
  std::pair<std::array<float, N>, std::array<float, N>> PackSmallPolygon<N>( \
      const Polygon& polygon) {                                              \
    assert(polygon.NumEdges() == N);                                         \
    std::array<float, N> x;                                                  \
    std::array<float, N> y;                                                  \
    for (int64_t i = 0; i < N; ++i) {                                        \
      x[i] = polygon.vertices().at(i).x();                                   \
      y[i] = polygon.vertices().at(i).y();                                   \
    }                                                                        \
    return std::make_pair(x, y);                                             \
  }
NOCTURNE_DEFINE_PACK_SMALL_POLYGON(3)
NOCTURNE_DEFINE_PACK_SMALL_POLYGON(4)
NOCTURNE_DEFINE_PACK_SMALL_POLYGON(5)
NOCTURNE_DEFINE_PACK_SMALL_POLYGON(6)
#undef NOCTURNE_DEFINE_PACK_SMALL_POLYGON

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
