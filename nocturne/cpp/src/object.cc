// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "object.h"

#include <algorithm>

#include "geometry/geometry_utils.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon Object::BoundingPolygon() const {
  const geometry::Vector2D p0 =
      geometry::Vector2D(length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p1 =
      geometry::Vector2D(-length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p2 =
      geometry::Vector2D(-length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  const geometry::Vector2D p3 =
      geometry::Vector2D(length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
      position_;
  return geometry::ConvexPolygon({p0, p1, p2, p3});
}

void Object::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  sf::RectangleShape rect(sf::Vector2f(length_, width_));
  rect.setOrigin(length_ / 2.0f, width_ / 2.0f);
  rect.setPosition(utils::ToVector2f(position_));
  rect.setRotation(geometry::utils::Degrees(heading_));

  sf::Color col;
  if (can_block_sight_ && can_be_collided_) {
    col = color_;
  } else if (can_block_sight_ && !can_be_collided_) {
    col = sf::Color::Blue;
  } else if (!can_block_sight_ && can_be_collided_) {
    col = sf::Color::White;
  } else {
    col = sf::Color::Black;
  }

  rect.setFillColor(col);
  target.draw(rect, states);

  if (highlight_) {
    float radius = std::max(length_, width_);
    sf::CircleShape circ(radius);
    circ.setOrigin(length_ / 2.0f, width_ / 2.0f);
    circ.setPosition(utils::ToVector2f(position_));
    circ.setFillColor(sf::Color(255, 0, 0, 100));
    target.draw(circ, states);
  }

  sf::ConvexShape arrow;
  arrow.setPointCount(3);
  arrow.setPoint(0, sf::Vector2f(0.0f, -width_ / 2.0f));
  arrow.setPoint(1, sf::Vector2f(0.0f, width_ / 2.0f));
  arrow.setPoint(2, sf::Vector2f(length_ / 2.0f, 0.0f));
  arrow.setOrigin(0.0f, 0.0f);
  arrow.setPosition(utils::ToVector2f(position_));
  arrow.setRotation(geometry::utils::Degrees(heading_));
  arrow.setFillColor(sf::Color::White);
  target.draw(arrow, states);
}

void Object::InitRandomColor() {
  std::uniform_int_distribution<int32_t> dis(0, 255);
  int32_t r = dis(random_gen_);
  int32_t g = dis(random_gen_);
  int32_t b = dis(random_gen_);
  // Rescale colors to avoid dark objects.
  const int32_t max_rgb = std::max({r, g, b});
  r = r * 255 / max_rgb;
  g = g * 255 / max_rgb;
  b = b * 255 / max_rgb;
  color_ = sf::Color(r, g, b);
}

void Object::SetActionFromKeyboard() {
  // up: accelerate ; down: brake
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Up)) {
    acceleration_ = 1.0f;
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Down)) {
    // larger acceleration for braking than for moving backwards
    acceleration_ = speed_ > 0 ? -2.0f : -1.0f;
  } else if (std::abs(speed_) < 0.05) {
    // clip to 0
    speed_ = 0.0f;
  } else {
    // friction
    acceleration_ = 0.5f * (speed_ > 0 ? -1.0f : 1.0f);
  }

  // right: turn right; left: turn left
  if (sf::Keyboard::isKeyPressed(sf::Keyboard::Right)) {
    steering_ = geometry::utils::Radians(-10.0f);
  } else if (sf::Keyboard::isKeyPressed(sf::Keyboard::Left)) {
    steering_ = geometry::utils::Radians(10.0f);
  } else {
    steering_ = 0.0f;
  }
}

// Kinematic Bicycle Model from Waymax
// https://github.com/waymo-research/waymax/blob/main/waymax/dynamics/bicycle_model.py
void Object::KinematicBicycleStep(float dt) {
  // Forward dynamics:
  //     new_x = x + vel_x * t + 1/2 * accel * cos(yaw) * t ** 2
  //     new_y = y + vel_y * t + 1/2 * accel * sin(yaw) * t ** 2
  //     new_yaw = yaw + steering * (speed * t + 1/2 * accel * t ** 2)
  //     new_vel = vel + accel * t
  geometry::Vector2D vel = Velocity();
  geometry::Vector2D rot = geometry::Vector2D(std::cos(heading_), std::sin(heading_));
  position_ = position_ + vel * dt + 0.5 * acceleration_ * rot * (float)pow(dt, 2);
  heading_ = geometry::utils::AngleAdd(heading_, (float)(steering_ * (speed_ * dt + 0.5 * acceleration_ * pow(dt, 2))));
  speed_ = ClipSpeed(speed_ + acceleration_ * dt);
}

}  // namespace nocturne
