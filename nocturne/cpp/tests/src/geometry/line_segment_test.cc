// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/line_segment.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

constexpr float kTestEps = 1e-5;

TEST(LineSegmentTest, AABBTest) {
  const Vector2D p(0.0f, 2.0f);
  const Vector2D q(1.0f, 0.0f);
  const LineSegment seg(p, q);
  const AABB aabb = seg.GetAABB();
  EXPECT_FLOAT_EQ(aabb.min().x(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.min().y(), 0.0f);
  EXPECT_FLOAT_EQ(aabb.max().x(), 1.0f);
  EXPECT_FLOAT_EQ(aabb.max().y(), 2.0f);
  EXPECT_FLOAT_EQ(aabb.Center().x(), 0.5f);
  EXPECT_FLOAT_EQ(aabb.Center().y(), 1.0f);
}

TEST(LineSegmentTest, NormalVectorTest) {
  Vector2D p(1.0f, 1.0f);
  Vector2D q(2.0f, 1.0f);
  Vector2D normal_vector = LineSegment(p, q).NormalVector();
  EXPECT_FLOAT_EQ(normal_vector.x(), 0.0f);
  EXPECT_FLOAT_EQ(normal_vector.y(), 1.0f);

  p = Vector2D(-1.0f, -1.0f);
  q = Vector2D(-2.0f, -2.0f);
  normal_vector = LineSegment(p, q).NormalVector();
  EXPECT_FLOAT_EQ(normal_vector.x(), M_SQRT1_2);
  EXPECT_FLOAT_EQ(normal_vector.y(), -M_SQRT1_2);
}

TEST(LineSegmentTest, ContainsTest) {
  const Vector2D p(0.0f, 0.0f);
  const Vector2D q(1.0f, 1.0f);
  const LineSegment seg(p, q);
  EXPECT_TRUE(seg.Contains(p));
  EXPECT_TRUE(seg.Contains(q));
  EXPECT_TRUE(seg.Contains(p + kTestEps));
  EXPECT_TRUE(seg.Contains(q - kTestEps));
  EXPECT_FALSE(seg.Contains(Vector2D(0.5f + kTestEps, 0.5f - kTestEps)));
  EXPECT_FALSE(seg.Contains(Vector2D(0.0f - kTestEps, 0.0f)));
  EXPECT_FALSE(seg.Contains(Vector2D(1.0f, 1.0f + kTestEps)));
}

TEST(LineSegmentTest, IntersectsTest) {
  Vector2D p1(0.0f, 0.0f);
  Vector2D q1(1.0f, 1.0f);
  Vector2D p2(1.0f, 0.0f);
  Vector2D q2(0.0f, 1.0f);
  EXPECT_TRUE(LineSegment(p1, q1).Intersects(LineSegment(p2, q2)));

  p1 = Vector2D(0.0f, 0.0f);
  q1 = Vector2D(0.0f, 1.0f);
  p2 = Vector2D(1.0f, 0.0f);
  q2 = Vector2D(0.0f, 1.0f);
  EXPECT_FALSE(LineSegment(p1, q1).Intersects(LineSegment(p2, q2)));
}

TEST(LineSegmentTest, IntersectionTest) {
  Vector2D p1(-1.0f, 0.0f);
  Vector2D q1(10.0f, 0.0f);
  Vector2D p2(-1.0f, -1.0f);
  Vector2D q2(10.0f, 10.0f);
  auto ret = LineSegment(p1, q1).Intersection(LineSegment(p2, q2));
  ASSERT_TRUE(ret.has_value());
  EXPECT_FLOAT_EQ(ret->x(), 0.0f);
  EXPECT_FLOAT_EQ(ret->y(), 0.0f);

  p1 = Vector2D(1.0f, 1.0f);
  q1 = Vector2D(1.0f, 2.0f);
  p2 = Vector2D(2.0f, 1.0f);
  q2 = Vector2D(0.0f, 2.0f);
  ret = LineSegment(p1, q1).Intersection(LineSegment(p2, q2));
  ASSERT_TRUE(ret.has_value());
  EXPECT_FLOAT_EQ(ret->x(), 1.0f);
  EXPECT_FLOAT_EQ(ret->y(), 1.5f);

  p1 = Vector2D(-3.0f, -2.0f);
  q1 = Vector2D(-1.0f, -4.0f);
  p2 = Vector2D(-3.0f, -4.0f);
  q2 = Vector2D(-1.0f, -2.0f);
  ret = LineSegment(p1, q1).Intersection(LineSegment(p2, q2));
  ASSERT_TRUE(ret.has_value());
  EXPECT_FLOAT_EQ(ret->x(), -2.0f);
  EXPECT_FLOAT_EQ(ret->y(), -3.0f);

  p1 = Vector2D(0.0f, 0.0f);
  q1 = Vector2D(1.0f, -2.0f);
  p2 = Vector2D(0.0f, 1.0f);
  q2 = Vector2D(1.0f, -1.0f);
  ret = LineSegment(p1, q1).Intersection(LineSegment(p2, q2));
  ASSERT_FALSE(ret.has_value());
}

TEST(LineSegmentTest, ParametricIntersectionTest) {
  Vector2D p1(-1.0f, 0.0f);
  Vector2D q1(9.0f, 0.0f);
  Vector2D p2(-1.0f, -1.0f);
  Vector2D q2(9.0f, 9.0f);
  auto ret = LineSegment(p1, q1).ParametricIntersection(LineSegment(p2, q2));
  ASSERT_TRUE(ret.has_value());
  EXPECT_FLOAT_EQ(*ret, 0.1f);

  p1 = Vector2D(1.0f, 1.0f);
  q1 = Vector2D(1.0f, 2.0f);
  p2 = Vector2D(2.0f, 1.0f);
  q2 = Vector2D(0.0f, 2.0f);
  ret = LineSegment(p1, q1).ParametricIntersection(LineSegment(p2, q2));
  ASSERT_TRUE(ret.has_value());
  EXPECT_FLOAT_EQ(*ret, 0.5f);

  p1 = Vector2D(0.0f, 0.0f);
  q1 = Vector2D(1.0f, -2.0f);
  p2 = Vector2D(0.0f, 1.0f);
  q2 = Vector2D(1.0f, -1.0f);
  ret = LineSegment(p1, q1).ParametricIntersection(LineSegment(p2, q2));
  ASSERT_FALSE(ret.has_value());
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
