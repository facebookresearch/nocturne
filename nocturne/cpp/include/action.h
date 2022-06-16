// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <optional>

namespace nocturne {

class Action {
 public:
  Action() = default;
  Action(std::optional<float> acceleration, std::optional<float> steering,
         std::optional<float> head_angle)
      : acceleration_(acceleration),
        steering_(steering),
        head_angle_(head_angle) {}

  std::optional<float> acceleration() const { return acceleration_; }
  void set_acceleration(std::optional<float> acceleration) {
    acceleration_ = acceleration;
  }

  std::optional<float> steering() const { return steering_; }
  void set_steering(std::optional<float> steering) { steering_ = steering; }

  std::optional<float> head_angle() const { return head_angle_; }
  void set_head_angle(std::optional<float> head_angle) {
    head_angle_ = head_angle;
  }

 protected:
  std::optional<float> acceleration_ = std::nullopt;
  std::optional<float> steering_ = std::nullopt;
  std::optional<float> head_angle_ = std::nullopt;
};

}  // namespace nocturne
