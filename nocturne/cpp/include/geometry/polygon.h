// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <initializer_list>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

// Vertices must be in counterclockwise order.
class Polygon : public AABBInterface {
 public:
  Polygon() = default;
  explicit Polygon(const Polygon& polygon) : vertices_(polygon.vertices_) {}
  explicit Polygon(Polygon&& polygon)
      : vertices_(std::move(polygon.vertices_)) {}
  explicit Polygon(const std::initializer_list<Vector2D>& vertices)
      : vertices_(vertices) {}
  explicit Polygon(const std::vector<Vector2D>& vertices)
      : vertices_(vertices) {}
  explicit Polygon(std::vector<Vector2D>&& vertices)
      : vertices_(std::move(vertices)) {}

  Polygon& operator=(const Polygon& polyon) {
    vertices_ = polyon.vertices_;
    return *this;
  }
  Polygon& operator=(Polygon&& polyon) {
    vertices_ = std::move(polyon.vertices_);
    return *this;
  }

  AABB GetAABB() const override;

  int64_t NumEdges() const { return vertices_.size(); }

  const std::vector<Vector2D>& vertices() const { return vertices_; }
  const Vector2D& Vertex(int64_t index) const { return vertices_.at(index); }

  std::vector<LineSegment> Edges() const;

  float Area() const;

 protected:
  std::vector<Vector2D> vertices_;
};

// Vertices must be in counterclockwise order.
class ConvexPolygon : public Polygon {
 public:
  ConvexPolygon() = default;
  explicit ConvexPolygon(const ConvexPolygon& polygon) : Polygon(polygon) {}
  explicit ConvexPolygon(ConvexPolygon&& polygon)
      : Polygon(std::move(polygon)) {}

  explicit ConvexPolygon(const std::initializer_list<Vector2D>& vertices)
      : Polygon(vertices) {
    assert(VerifyVerticesOrder());
  }

  explicit ConvexPolygon(const std::vector<Vector2D>& vertices)
      : Polygon(vertices) {
    assert(VerifyVerticesOrder());
  }

  explicit ConvexPolygon(std::vector<Vector2D>&& vertices)
      : Polygon(std::move(vertices)) {
    assert(VerifyVerticesOrder());
  }

  ConvexPolygon& operator=(const ConvexPolygon& polyon) {
    Polygon::operator=(polyon);
    return *this;
  }
  ConvexPolygon& operator=(ConvexPolygon&& polyon) {
    Polygon::operator=(std::move(polyon));
    return *this;
  }

  bool Contains(const Vector2D& p) const;

  bool Intersects(const ConvexPolygon& polygon) const;

 protected:
  bool VerifyVerticesOrder() const;
};

}  // namespace geometry
}  // namespace nocturne
