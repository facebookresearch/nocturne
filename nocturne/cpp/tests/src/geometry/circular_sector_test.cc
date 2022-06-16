// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/circular_sector.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>

#include "geometry/aabb.h"
#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace {

using testing::AnyOf;
using testing::FloatEq;

constexpr float kTestEps = 1e-5;

TEST(CircularSectorTest, CtorTest) {
  const Vector2D center(1.0f, 1.0f);
  const float radius = 2.0f;
  const float heading = utils::kQuarterPi;

  const float theta1 = utils::kHalfPi;
  const CircularSector circular_sector1(center, radius, heading, theta1);
  EXPECT_FLOAT_EQ(circular_sector1.theta(), utils::kHalfPi);
  EXPECT_FLOAT_EQ(circular_sector1.Angle0(), 0.0f);
  EXPECT_FLOAT_EQ(circular_sector1.Angle1(), utils::kHalfPi);
  EXPECT_FLOAT_EQ(circular_sector1.Area(), utils::kPi);

  const float theta2 = utils::kHalfPi * 3.0f;
  const CircularSector circular_sector2(center, radius, heading, theta2);
  EXPECT_FLOAT_EQ(circular_sector2.theta(), -utils::kHalfPi);
  EXPECT_FLOAT_EQ(circular_sector2.Angle0(), -utils::kHalfPi);
  EXPECT_THAT(circular_sector2.Angle1(),
              AnyOf(FloatEq(utils::kPi), FloatEq(-utils::kPi)));
  EXPECT_FLOAT_EQ(circular_sector2.Area(), utils::kPi * 3.0f);
}

TEST(CircularSectorTest, AABBTest) {
  const Vector2D center(1.0f, 1.0f);
  const float radius = 2.0f;

  const float heading1 = 0.0f;
  const float theta1 = utils::kHalfPi;
  const CircularSector circular_sector1(center, radius, heading1, theta1);
  const AABB aabb1 = circular_sector1.GetAABB();
  EXPECT_FLOAT_EQ(aabb1.MinX(), 1.0f);
  EXPECT_FLOAT_EQ(aabb1.MinY(), 1.0f - M_SQRT2);
  EXPECT_FLOAT_EQ(aabb1.MaxX(), 3.0f);
  EXPECT_FLOAT_EQ(aabb1.MaxY(), 1.0f + M_SQRT2);

  const float heading2 = utils::kHalfPi;
  const float theta2 = utils::kHalfPi * 3.0f;
  const CircularSector circular_sector2(center, radius, heading2, theta2);
  const AABB aabb2 = circular_sector2.GetAABB();
  EXPECT_FLOAT_EQ(aabb2.MinX(), -1.0f);
  EXPECT_NEAR(aabb2.MinY(), 1.0f - M_SQRT2, kTestEps);
  EXPECT_FLOAT_EQ(aabb2.MaxX(), 3.0f);
  EXPECT_FLOAT_EQ(aabb2.MaxY(), 3.0f);
}

TEST(CircularSectorTest, ContainsTest) {
  const Vector2D center(1.0f, 1.0f);
  const float radius = 2.0f;
  const float heading = utils::kQuarterPi * 3.0f;
  const float theta = utils::Radians(120.0f);
  const CircularSector circular_sector(center, radius, heading, theta);

  EXPECT_TRUE(circular_sector.Contains(center));
  EXPECT_TRUE(
      circular_sector.Contains(center + circular_sector.Radius0() - kTestEps));
  EXPECT_TRUE(
      circular_sector.Contains(center + circular_sector.Radius1() + kTestEps));
  EXPECT_TRUE(circular_sector.Contains(center + Vector2D(-1.0f, 1.0f)));

  EXPECT_FALSE(
      circular_sector.Contains(center + Vector2D(kTestEps, -kTestEps)));
  EXPECT_FALSE(
      circular_sector.Contains(center + circular_sector.Radius0() + kTestEps));
  EXPECT_FALSE(
      circular_sector.Contains(center + circular_sector.Radius1() - kTestEps));
  EXPECT_FALSE(circular_sector.Contains(Vector2D(10.0f, 10.0f)));
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
