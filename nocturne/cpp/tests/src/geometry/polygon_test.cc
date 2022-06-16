// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/polygon.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

constexpr float kTestEps = 1e-5;

TEST(PolygonTest, AABBTest) {
  const Vector2D a(1.0f, 2.0f);
  const Vector2D b(5.0f, 1.0f);
  const Vector2D c(2.0f, 4.0f);
  const Polygon polygon({a, b, c});
  const AABB aabb = polygon.GetAABB();
  EXPECT_FLOAT_EQ(aabb.min().x(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.min().y(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.max().x(), 5.0f);
  EXPECT_FLOAT_EQ(aabb.max().y(), 4.0f);
  EXPECT_FLOAT_EQ(aabb.Center().x(), 3.0f);
  EXPECT_FLOAT_EQ(aabb.Center().y(), 2.5f);
}

TEST(PolygonTest, AreaTest) {
  const Vector2D a(1.0f, 1.0f);
  const Vector2D b(2.0f, 2.0f);
  const Vector2D c(3.0f, 1.0f);
  const Vector2D d(2.0f, 5.0f);
  const Polygon polygon({a, b, c, d});
  EXPECT_FLOAT_EQ(polygon.Area(), 3.0f);
}

TEST(ConvexPolygonTest, ContainsTest) {
  const Vector2D a(1.0f, 2.0f);
  const Vector2D b(5.0f, 1.0f);
  const Vector2D c(2.0f, 4.0f);
  const ConvexPolygon polygon({a, b, c});
  EXPECT_TRUE(polygon.Contains(Vector2D(2.0f, 3.0f)));
  EXPECT_TRUE(polygon.Contains(a));
  EXPECT_TRUE(polygon.Contains(b));
  EXPECT_TRUE(polygon.Contains(c));
  EXPECT_FALSE(polygon.Contains(Vector2D(0.0f, 0.0f)));

  EXPECT_TRUE(polygon.Contains(a + kTestEps));
  EXPECT_FALSE(polygon.Contains(a - kTestEps));
  EXPECT_FALSE(polygon.Contains(b + kTestEps));
  EXPECT_FALSE(polygon.Contains(b - kTestEps));
  EXPECT_FALSE(polygon.Contains(c + kTestEps));
  EXPECT_FALSE(polygon.Contains(c - kTestEps));
}

TEST(ConvexPolygonTest, IntersectsTest) {
  Vector2D p1(0.0f, 0.0f);
  Vector2D p2(1.0f, 0.0f);
  Vector2D p3(1.0f, 1.0f);
  Vector2D p4(0.0f, 1.0f);
  ConvexPolygon polygon1({p1, p2, p3, p4});
  Vector2D q1(1.0f, 2.0f);
  Vector2D q2(2.0f, 1.0f);
  Vector2D q3(2.0f, 2.0f);
  ConvexPolygon polygon2({q1, q2, q3});
  EXPECT_FALSE(polygon1.Intersects(polygon2));
  EXPECT_FALSE(polygon2.Intersects(polygon1));

  q1 = Vector2D(1.0f - kTestEps, 1.0f - kTestEps);
  q2 = Vector2D(2.0f, 0.0f);
  q3 = Vector2D(2.0f, 2.0f);
  polygon2 = ConvexPolygon({q1, q2, q3});
  EXPECT_TRUE(polygon1.Intersects(polygon2));
  EXPECT_TRUE(polygon2.Intersects(polygon1));

  q1 = Vector2D(1.0f, 1.0f);
  q2 = Vector2D(2.0f, 0.0f);
  q3 = Vector2D(2.0f, 2.0f);
  polygon2 = ConvexPolygon({q1, q2, q3});
  EXPECT_TRUE(polygon1.Intersects(polygon2));
  EXPECT_TRUE(polygon2.Intersects(polygon1));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
