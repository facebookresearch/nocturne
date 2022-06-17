// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "object.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <utility>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace {

using geometry::utils::kHalfPi;
using geometry::utils::kQuarterPi;

constexpr float kTol = 1e-4;

std::pair<geometry::Vector2D, float> KinematicBicycleModel(
    const geometry::Vector2D& position, float length, float heading,
    float speed, float delta, float dt) {
  const float beta = std::atan(std::tan(delta) * 0.5f);
  const float dx = speed * std::cos(heading + beta);
  const float dy = speed * std::sin(heading + beta);
  const float dtheta = speed * std::tan(delta) * std::cos(beta) / length;
  return std::make_pair(position + geometry::Vector2D(dx * dt, dy * dt),
                        geometry::utils::NormalizeAngle(heading + dtheta * dt));
}

TEST(ObjectTest, UniformLinearMotionTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  const float speed = 10.0f;
  const geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  const geometry::Vector2D target_position = position + velocity * t;
  const float target_heading = heading;
  const float target_speed = speed;

  Object obj(/*id=*/0, length, width, position, heading, speed, target_position,
             target_heading, target_speed);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_NEAR(obj.position().x(), target_position.x(), kTol);
  EXPECT_NEAR(obj.position().y(), target_position.y(), kTol);
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_FLOAT_EQ(obj.speed(), target_speed);
}

TEST(ObjectTest, ConstantAccelerationMotionTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  float speed = 0.0f;
  float acceleration = 2.0f;
  geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  geometry::Vector2D target_position =
      position + velocity * t +
      geometry::PolarToVector2D(acceleration, heading) * (t * t * 0.5f);
  const float target_heading = heading;
  float target_speed = speed + acceleration * t;

  // Forward test.
  Object obj(/*id=*/0, length, width, position, heading, speed, target_position,
             target_heading, target_speed);
  obj.set_acceleration(acceleration);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_NEAR(obj.position().x(), target_position.x(), kTol);
  EXPECT_NEAR(obj.position().y(), target_position.y(), kTol);
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_NEAR(obj.speed(), target_speed, kTol);

  // Backward test.
  speed = 10.0f;
  acceleration = -2.0f;
  velocity = geometry::PolarToVector2D(speed, heading);
  target_position =
      position + velocity * t +
      geometry::PolarToVector2D(acceleration, heading) * (t * t * 0.5f);
  target_speed = speed + acceleration * t;
  obj.set_position(position);
  obj.set_speed(speed);
  obj.set_target_position(target_position);
  obj.set_target_speed(target_speed);
  obj.set_acceleration(acceleration);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_NEAR(obj.position().x(), target_position.x(), kTol);
  EXPECT_NEAR(obj.position().y(), target_position.y(), kTol);
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_NEAR(obj.speed(), target_speed, kTol);
}

TEST(ObjectTest, SpeedCliptTest) {
  const float t = 10.0f;
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kHalfPi;
  const float max_speed = 10.0f;
  const float speed = 0.0f;
  float acceleration = 2.0f;
  const geometry::Vector2D velocity = geometry::PolarToVector2D(speed, heading);
  geometry::Vector2D final_velocity =
      geometry::PolarToVector2D(max_speed, heading);
  const geometry::Vector2D position(1.0f, 1.0f);
  const float t1 = max_speed / acceleration;
  const float t2 = t - t1;
  geometry::Vector2D target_position =
      position + velocity * t1 +
      geometry::PolarToVector2D(acceleration, heading) * (t1 * t1 * 0.5f) +
      final_velocity * t2;
  const float target_heading = heading;
  float target_speed = max_speed;

  // Forward test.
  Object obj(/*id=*/0, length, width, max_speed, position, heading, speed,
             target_position, target_heading, target_speed);
  obj.set_acceleration(acceleration);
  const int num_steps = 100;
  const float dt = t / static_cast<float>(num_steps);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_NEAR(obj.position().x(), target_position.x(), kTol);
  EXPECT_NEAR(obj.position().y(), target_position.y(), kTol);
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_FLOAT_EQ(obj.speed(), target_speed);

  // Backward test.
  acceleration = -2.0f;
  final_velocity = geometry::PolarToVector2D(-max_speed, heading);
  target_position =
      position + velocity * t1 +
      geometry::PolarToVector2D(acceleration, heading) * (t1 * t1 * 0.5f) +
      final_velocity * t2;
  target_speed = -max_speed;
  obj.set_position(position);
  obj.set_speed(speed);
  obj.set_target_position(target_position);
  obj.set_target_speed(target_speed);
  obj.set_acceleration(acceleration);
  for (int i = 0; i < num_steps; ++i) {
    obj.Step(dt);
  }
  EXPECT_NEAR(obj.position().x(), target_position.x(), kTol);
  EXPECT_NEAR(obj.position().y(), target_position.y(), kTol);
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_FLOAT_EQ(obj.speed(), target_speed);
}

TEST(ObjectTest, SteeringMotionTest) {
  const float length = 2.0f;
  const float width = 1.0f;
  const float heading = kQuarterPi;
  const float speed = 2.0f;
  const float steering = geometry::utils::Radians(10.0f);
  const float dt = 0.1f;
  const geometry::Vector2D position(1.0f, 1.0f);
  const auto [target_position, target_heading] =
      KinematicBicycleModel(position, length, heading, speed, steering, dt);
  const float target_speed = speed;

  Object obj(/*id=*/0, length, width, position, heading, speed, target_position,
             target_heading, target_speed);
  obj.set_steering(steering);
  obj.Step(dt);

  EXPECT_FLOAT_EQ(obj.position().x(), target_position.x());
  EXPECT_FLOAT_EQ(obj.position().y(), target_position.y());
  EXPECT_FLOAT_EQ(obj.heading(), target_heading);
  EXPECT_FLOAT_EQ(obj.speed(), target_speed);
  EXPECT_FLOAT_EQ(obj.Velocity().x(), speed * std::cos(target_heading));
  EXPECT_FLOAT_EQ(obj.Velocity().y(), speed * std::sin(target_heading));
}

}  // namespace
}  // namespace nocturne
