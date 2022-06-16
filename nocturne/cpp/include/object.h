// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <SFML/Graphics.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <random>
#include <string>

#include "action.h"
#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {

constexpr float kViewRadius = 120.0f;

enum class ObjectType {
  kUnset = 0,
  kVehicle = 1,
  kPedestrian = 2,
  kCyclist = 3,
  kOther = 4,
};

class Object : public ObjectBase {
 public:
  Object() = default;

  Object(int64_t id, float length, float width,
         const geometry::Vector2D& position, float heading, float speed,
         const geometry::Vector2D& target_position, float target_heading,
         float target_speed, bool can_block_sight = true,
         bool can_be_collided = true, bool check_collision = true)
      : ObjectBase(position, can_block_sight, can_be_collided, check_collision),
        id_(id),
        length_(length),
        width_(width),
        heading_(heading),
        speed_(ClipSpeed(speed)),
        target_position_(target_position),
        target_heading_(target_heading),
        target_speed_(target_speed),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  Object(int64_t id, float length, float width, float max_speed,
         const geometry::Vector2D& position, float heading, float speed,
         const geometry::Vector2D& target_position, float target_heading,
         float target_speed, bool can_block_sight = true,
         bool can_be_collided = true, bool check_collision = true)
      : ObjectBase(position, can_block_sight, can_be_collided, check_collision),
        id_(id),
        length_(length),
        width_(width),
        max_speed_(max_speed),
        heading_(heading),
        speed_(ClipSpeed(speed)),
        target_position_(target_position),
        target_heading_(target_heading),
        target_speed_(target_speed),
        random_gen_(std::random_device()()) {
    InitRandomColor();
  }

  virtual ObjectType Type() const { return ObjectType::kUnset; }

  int64_t id() const { return id_; }

  float length() const { return length_; }
  float width() const { return width_; }
  float max_speed() const { return max_speed_; }

  float heading() const { return heading_; }
  void set_heading(float heading) { heading_ = heading; }

  float speed() const { return speed_; }
  void set_speed(float speed) { speed_ = ClipSpeed(speed); }

  const geometry::Vector2D& target_position() const { return target_position_; }
  void set_target_position(const geometry::Vector2D& target_position) {
    target_position_ = target_position;
  }
  void set_target_position(float x, float y) {
    target_position_ = geometry::Vector2D(x, y);
  }

  float target_heading() const { return target_heading_; }
  void set_target_heading(float target_heading) {
    target_heading_ = target_heading;
  }

  float target_speed() const { return target_speed_; }
  void set_target_speed(float target_speed) { target_speed_ = target_speed; }

  float acceleration() const { return acceleration_; }
  void set_acceleration(float acceleration) { acceleration_ = acceleration; }

  float steering() const { return steering_; }
  void set_steering(float steering) { steering_ = steering; }

  float head_angle() const { return head_angle_; }
  void set_head_angle(float head_angle) { head_angle_ = head_angle; }

  bool manual_control() const { return manual_control_; }
  void set_manual_control(bool manual_control) {
    manual_control_ = manual_control;
  }

  bool expert_control() const { return expert_control_; }
  void set_expert_control(bool expert_control) {
    expert_control_ = expert_control;
  }

  bool highlight() const { return highlight_; }
  void set_highlight(bool highlight) { highlight_ = highlight; }

  const sf::Color& color() const { return color_; }

  sf::RenderTexture* ConeTexture() const { return cone_texture_.get(); }

  bool InitConeTexture(int64_t h, int64_t w,
                       const sf::ContextSettings& settings) {
    if (cone_texture_ == nullptr) {
      cone_texture_ = std::make_unique<sf::RenderTexture>();
      cone_texture_->create(w, h, settings);
      return true;
    }
    return false;
  }

  float Radius() const override {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  geometry::Vector2D Velocity() const {
    return geometry::PolarToVector2D(speed_, heading_);
  }

  geometry::ConvexPolygon BoundingPolygon() const override;

  void ScaleShape(float length_scale, float width_scale) {
    length_ *= length_scale;
    width_ *= width_scale;
  }

  void ApplyAction(const Action& action) {
    if (action.acceleration().has_value()) {
      acceleration_ = action.acceleration().value();
    }
    if (action.steering().has_value()) {
      steering_ = action.steering().value();
    }
    if (action.head_angle().has_value()) {
      head_angle_ = action.head_angle().value();
    }
  }

  void SetActionFromKeyboard();

  virtual void Step(float dt) {
    if (manual_control_) {
      SetActionFromKeyboard();
    }
    KinematicBicycleStep(dt);
  }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRandomColor();

  void KinematicBicycleStep(float dt);

  float ClipSpeed(float speed) const {
    return std::max(std::min(speed, max_speed_), -max_speed_);
  }

  const int64_t id_;

  float length_ = 0.0f;
  float width_ = 0.0f;
  const float max_speed_ = std::numeric_limits<float>::max();

  float heading_ = 0.0f;
  // Postive for moving forward, negative for moving backward.
  float speed_ = 0.0f;

  geometry::Vector2D target_position_;
  float target_heading_ = 0.0f;
  float target_speed_ = 0.0f;

  float acceleration_ = 0.0f;
  float steering_ = 0.0f;
  float head_angle_ = 0.0f;

  // used to color the object in videos if set to True
  bool highlight_ = false;

  // If true the object is controlled by keyboard input.
  bool manual_control_ = false;
  // If true the object is placed along positions in its recorded trajectory.
  bool expert_control_ = false;

  sf::Color color_;
  std::unique_ptr<sf::RenderTexture> cone_texture_ = nullptr;

  std::mt19937 random_gen_;
};

inline ObjectType ParseObjectType(const std::string& type) {
  if (type == "unset") {
    return ObjectType::kUnset;
  } else if (type == "vehicle") {
    return ObjectType::kVehicle;
  } else if (type == "pedestrian") {
    return ObjectType::kPedestrian;
  } else if (type == "cyclist") {
    return ObjectType::kCyclist;
  } else {
    return ObjectType::kOther;
  }
}

}  // namespace nocturne
