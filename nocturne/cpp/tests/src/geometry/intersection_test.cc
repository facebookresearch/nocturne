// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/intersection.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>

#include "geometry/circle.h"
#include "geometry/circular_sector.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

constexpr float kTestEps = 1e-5;
constexpr float kTestTol = 1e-6;

TEST(IntersectsTest, AABBLineSegmentTest) {
  const AABB aabb(0.0f, 0.0f, 4.0f, 2.0f);

  const LineSegment seg1(Vector2D(1.0f, 1.0f), Vector2D(1.5f, 1.5f));
  EXPECT_TRUE(Intersects(aabb, seg1));
  EXPECT_TRUE(Intersects(seg1, aabb));
  const LineSegment seg2(Vector2D(-1.0f, -1.0f), Vector2D(0.0f, 0.0f));
  EXPECT_TRUE(Intersects(aabb, seg2));
  EXPECT_TRUE(Intersects(seg2, aabb));
  const LineSegment seg3(Vector2D(-1.0f, 1.0f), Vector2D(2.0f, -1.0f));
  EXPECT_TRUE(Intersects(aabb, seg3));
  EXPECT_TRUE(Intersects(seg3, aabb));
  const LineSegment seg4(Vector2D(1.0f, 3.0f), Vector2D(1.0f, 1.0f));
  EXPECT_TRUE(Intersects(aabb, seg4));
  EXPECT_TRUE(Intersects(seg4, aabb));

  const LineSegment seg5(Vector2D(4.0f + kTestEps, 0.5f),
                         Vector2D(4.0f + kTestEps, 1.0f));
  EXPECT_FALSE(Intersects(aabb, seg5));
  EXPECT_FALSE(Intersects(seg5, aabb));
  const LineSegment seg6(Vector2D(3.0f + kTestEps, -1.0f),
                         Vector2D(5.0f + kTestEps, 1.0f));
  EXPECT_FALSE(Intersects(aabb, seg6));
  EXPECT_FALSE(Intersects(seg6, aabb));
}

TEST(IntersectsTest, ConvexPolygonLineSegmentTest) {
  const ConvexPolygon polygon({Vector2D(1.0f, 0.0f), Vector2D(0.0f, 1.0f),
                               Vector2D(-1.0f, 0.0f), Vector2D(0.0f, -1.0f)});

  const LineSegment seg1(Vector2D(0.0f, 0.5f), Vector2D(0.0f, -0.5f));
  EXPECT_TRUE(Intersects(polygon, seg1));
  EXPECT_TRUE(Intersects(seg1, polygon));
  const LineSegment seg2(Vector2D(-0.5f, -0.5f), Vector2D(-0.5f, -1.0f));
  EXPECT_TRUE(Intersects(polygon, seg2));
  EXPECT_TRUE(Intersects(seg2, polygon));
  const LineSegment seg3(Vector2D(-1.0f, 0.5f), Vector2D(1.0f, 1.0f));
  EXPECT_TRUE(Intersects(polygon, seg3));
  EXPECT_TRUE(Intersects(seg3, polygon));
  const LineSegment seg4(Vector2D(1.0f, 1.0f), Vector2D(-1.0f, -1.0f));
  EXPECT_TRUE(Intersects(polygon, seg4));
  EXPECT_TRUE(Intersects(seg4, polygon));

  const LineSegment seg5(Vector2D(-1.0f, -1.0f - kTestEps),
                         Vector2D(1.0f, -1.0f - kTestEps));
  EXPECT_FALSE(Intersects(polygon, seg5));
  EXPECT_FALSE(Intersects(seg5, polygon));
  const LineSegment seg6(Vector2D(-3.0f, 0.5f), Vector2D(-2.0f, 1.0f));
  EXPECT_FALSE(Intersects(polygon, seg6));
  EXPECT_FALSE(Intersects(seg6, polygon));
}

TEST(IntersectionTest, CircleLineSegmentTest) {
  const Circle circle(Vector2D(1.0, 1.0f), 2.0f);
  const LineSegment line1(Vector2D(-1.0f - kTestEps, -1.0f),
                          Vector2D(-1.0f - kTestEps, 1.0f));
  const auto [p1, q1] = Intersection(circle, line1);
  ASSERT_FALSE(p1.has_value());
  ASSERT_FALSE(q1.has_value());

  const LineSegment line2(Vector2D(0.0f, 3.0f), Vector2D(4.0f, 3.0f));
  const auto [p2, q2] = Intersection(circle, line2);
  ASSERT_TRUE(p2.has_value());
  ASSERT_FALSE(q2.has_value());
  EXPECT_FLOAT_EQ(p2->x(), 1.0f);
  EXPECT_FLOAT_EQ(p2->y(), 3.0f);

  const LineSegment line3(Vector2D(0.0f, 0.0f), Vector2D(4.0f, 4.0f));
  const auto [p3, q3] = Intersection(circle, line3);
  ASSERT_TRUE(p3.has_value());
  ASSERT_FALSE(q3.has_value());
  EXPECT_FLOAT_EQ(p3->x(), 1.0f + M_SQRT2);
  EXPECT_FLOAT_EQ(p3->y(), 1.0f + M_SQRT2);

  const LineSegment line4(Vector2D(0.0f, 4.0f), Vector2D(0.0f, -4.0f));
  const auto [p4, q4] = Intersection(circle, line4);
  ASSERT_TRUE(p4.has_value());
  ASSERT_TRUE(q4.has_value());
  EXPECT_FLOAT_EQ(p4->x(), 0.0f);
  EXPECT_FLOAT_EQ(p4->y(), 1.0f + std::sqrt(3.0f));
  EXPECT_FLOAT_EQ(q4->x(), 0.0f);
  EXPECT_FLOAT_EQ(q4->y(), 1.0f - std::sqrt(3.0f));
}

TEST(IntersectionTest, CircularSectorLineSegmentTest) {
  const CircularSector sector(Vector2D(0.0f, 0.0f), 2.0f, utils::kQuarterPi,
                              utils::kHalfPi);

  const LineSegment line1(Vector2D(2.0f + kTestEps, -1.0f),
                          Vector2D(2.0f + kTestEps, 1.0f));
  const auto [p1, q1] = Intersection(sector, line1);
  ASSERT_FALSE(p1.has_value());
  ASSERT_FALSE(q1.has_value());

  const LineSegment line2(Vector2D(2.0f - kTestEps, -1.0f),
                          Vector2D(2.0f - kTestEps, kTestEps));
  const auto [p2, q2] = Intersection(sector, line2);
  ASSERT_TRUE(p2.has_value());
  ASSERT_FALSE(q2.has_value());
  EXPECT_FLOAT_EQ(p2->x(), 2.0f - kTestEps);
  EXPECT_FLOAT_EQ(p2->y(), 0.0f);

  const LineSegment line3(Vector2D(2.0f, -1.0f), Vector2D(-1.0f, 2.0f));
  const auto [p3, q3] = Intersection(sector, line3);
  ASSERT_TRUE(p3.has_value());
  ASSERT_TRUE(q3.has_value());
  EXPECT_NEAR(p3->x(), 1.0f, kTestTol);
  EXPECT_NEAR(p3->y(), 0.0f, kTestTol);
  EXPECT_NEAR(q3->x(), 0.0f, kTestTol);
  EXPECT_NEAR(q3->y(), 1.0f, kTestTol);

  const LineSegment line4(Vector2D(1.0f, 1.0f), Vector2D(5.0f, 5.0f));
  const auto [p4, q4] = Intersection(sector, line4);
  ASSERT_TRUE(p4.has_value());
  ASSERT_FALSE(q4.has_value());
  EXPECT_FLOAT_EQ(p4->x(), M_SQRT2);
  EXPECT_FLOAT_EQ(p4->y(), M_SQRT2);

  const float d = std::sqrt(3.75f);
  const LineSegment line5(Vector2D(d + 0.5f, 0.0f), Vector2D(0.0f, d + 0.5f));
  const auto [p5, q5] = Intersection(sector, line5);
  ASSERT_TRUE(p5.has_value());
  ASSERT_TRUE(q5.has_value());
  EXPECT_NEAR(p5->x(), d, kTestTol);
  EXPECT_NEAR(p5->y(), 0.5f, kTestTol);
  EXPECT_NEAR(q5->x(), 0.5f, kTestTol);
  EXPECT_NEAR(q5->y(), d, kTestTol);
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
