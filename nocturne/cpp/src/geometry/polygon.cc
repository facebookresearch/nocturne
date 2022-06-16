// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/polygon.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <utility>

#include "geometry/geometry_utils.h"

namespace nocturne {
namespace geometry {

namespace {

bool Separates(const LineSegment& edge, const Polygon& polygon) {
  const Vector2D d = edge.Endpoint1() - edge.Endpoint0();
  for (const Vector2D& p : polygon.vertices()) {
    if (CrossProduct(p - edge.Endpoint0(), d) <= 0.0f) {
      return false;
    }
  }
  return true;
}

}  // namespace

AABB Polygon::GetAABB() const {
  float min_x = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float min_y = std::numeric_limits<float>::max();
  float max_y = std::numeric_limits<float>::lowest();
  for (const Vector2D& v : vertices_) {
    min_x = std::min(min_x, v.x());
    max_x = std::max(max_x, v.x());
    min_y = std::min(min_y, v.y());
    max_y = std::max(max_y, v.y());
  }
  return AABB(min_x, min_y, max_x, max_y);
}

std::vector<LineSegment> Polygon::Edges() const {
  std::vector<LineSegment> edges;
  const int64_t n = vertices_.size();
  edges.reserve(n);
  for (int64_t i = 1; i < n; ++i) {
    edges.emplace_back(vertices_[i - 1], vertices_[i]);
  }
  edges.emplace_back(vertices_.back(), vertices_.front());
  return edges;
}

float Polygon::Area() const {
  const int64_t n = vertices_.size();
  float s = CrossProduct(vertices_.back(), vertices_.front());
  for (int64_t i = 1; i < n; ++i) {
    s += CrossProduct(vertices_[i - 1], vertices_[i]);
  }
  return std::fabs(s) * 0.5f;
}

// Check if p lies on the left of all the edges given the vertices are in
// counterclockwise order.
// Time Complexy: O(N)
// TODO: Add O(logN) algorithm if some polygon contains many vertices.
bool ConvexPolygon::Contains(const Vector2D& p) const {
  const int64_t n = vertices_.size();
  for (int64_t i = 1; i < n; ++i) {
    if (CrossProduct(p - vertices_[i - 1], vertices_[i] - vertices_[i - 1]) >
        0.0f) {
      return false;
    }
  }
  // Check the last edge which is (V_{n - 1}, V_{0}).
  return CrossProduct(p - vertices_.back(),
                      vertices_.front() - vertices_.back()) <= 0.0f;
}

// Assume polygon vertices are in counterclockwise order.
// Check if the other polygon lies on the right of one of the edges.
bool ConvexPolygon::Intersects(const ConvexPolygon& polygon) const {
  std::vector<LineSegment> edges = Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(edge, polygon)) {
      return false;
    }
  }
  edges = polygon.Edges();
  for (const LineSegment& edge : edges) {
    if (Separates(edge, *this)) {
      return false;
    }
  }
  return true;
}

bool ConvexPolygon::VerifyVerticesOrder() const {
  const int64_t n = vertices_.size();
  assert(n > 2);
  for (int64_t i = 2; i < n; ++i) {
    if (CrossProduct(vertices_[i] - vertices_[i - 1],
                     vertices_[i - 1] - vertices_[i - 2]) > 0.0f) {
      return false;
    }
  }
  return CrossProduct(vertices_[0] - vertices_[n - 1],
                      vertices_[n - 1] - vertices_[n - 2]) <= 0.0f &&
         CrossProduct(vertices_[1] - vertices_[0],
                      vertices_[0] - vertices_[n - 1]) <= 0.0f;
}

}  // namespace geometry
}  // namespace nocturne
