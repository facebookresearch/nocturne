// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/circle.h"
#include "geometry/circular_sector.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/point_like.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

inline bool Intersects(const AABB& lhs, const AABB& rhs) {
  return lhs.Intersects(rhs);
}

inline bool Intersects(const LineSegment& lhs, const LineSegment& rhs) {
  return lhs.Intersects(rhs);
}

bool Intersects(const AABB& aabb, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const AABB& aabb);

bool Intersects(const ConvexPolygon& polygon, const LineSegment& segment);
bool Intersects(const LineSegment& segment, const ConvexPolygon& polygon);

std::vector<utils::MaskType> BatchIntersects(const ConvexPolygon& polygon,
                                             const Vector2D& o,
                                             const std::vector<float>& x,
                                             const std::vector<float>& y);

std::vector<utils::MaskType> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<Vector2D>& points);

std::vector<utils::MaskType> BatchIntersects(
    const ConvexPolygon& polygon, const Vector2D& o,
    const std::vector<const PointLike*>& points);

inline bool Intersects(const ConvexPolygon& lhs, const ConvexPolygon& rhs) {
  return lhs.Intersects(rhs);
}

inline std::optional<Vector2D> Intersection(const LineSegment& lhs,
                                            const LineSegment& rhs) {
  return lhs.Intersection(rhs);
}

inline std::optional<float> ParametricIntersection(const LineSegment& lhs,
                                                   const LineSegment& rhs) {
  return lhs.ParametricIntersection(rhs);
}

// Batch version of ParametricIntersection.
// Computes the parametric intersection of (ox_i, oy_i) and polygon.
// If there is no intersection, result will be inf.
std::vector<float> BatchParametricIntersection(const Vector2D& o,
                                               const std::vector<float>& x,
                                               const std::vector<float>& y,
                                               const ConvexPolygon& polygon);

std::vector<float> BatchParametricIntersection(
    const Vector2D& o, const std::vector<Vector2D>& points,
    const ConvexPolygon& polygon);

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const Circle& circle, const LineSegment& segment) {
  return circle.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const Circle& circle) {
  return circle.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const CircularSector& circular_sector, const LineSegment& segment) {
  return circular_sector.Intersection(segment);
}

inline std::pair<std::optional<Vector2D>, std::optional<Vector2D>> Intersection(
    const LineSegment& segment, const CircularSector& circular_sector) {
  return circular_sector.Intersection(segment);
}

}  // namespace geometry
}  // namespace nocturne
