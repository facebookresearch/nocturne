// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <SFML/Graphics.hpp>
#include <cassert>
#include <string>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/polygon.h"
#include "static_object.h"

namespace nocturne {

constexpr float kTrafficLightRadius = 2.0f;
constexpr int kTrafficLightNumEdges = 5;

enum class TrafficLightState {
  kUnknown = 0,
  kStop = 1,
  kCaution = 2,
  kGo = 3,
  kArrowStop = 4,
  kArrowCaution = 5,
  kArrowGo = 6,
  kFlashingStop = 7,
  kFlashingCaution = 8,
};

class TrafficLight : public StaticObject {
 public:
  TrafficLight() = default;
  TrafficLight(const geometry::Vector2D& position,
               const std::vector<int64_t> timestamps,
               const std::vector<TrafficLightState>& light_states,
               int64_t current_time)
      : StaticObject(position,
                     /*can_block_sight=*/false,
                     /*can_be_collided=*/false, /*check_collision=*/false),
        timestamps_(timestamps),
        light_states_(light_states),
        current_time_(current_time) {
    assert(timestamps_.size() == light_states_.size());
  }

  StaticObjectType Type() const override {
    return StaticObjectType::kTrafficLight;
  }

  float Radius() const override { return kTrafficLightRadius; }

  geometry::ConvexPolygon BoundingPolygon() const override;

  const std::vector<int64_t>& timestamps() const { return timestamps_; }
  const std::vector<TrafficLightState>& light_states() const {
    return light_states_;
  }

  int64_t current_time() const { return current_time_; }
  void set_current_time(int64_t current_time) { current_time_ = current_time; }

  TrafficLightState LightState() const;

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  const std::vector<int64_t> timestamps_;
  const std::vector<TrafficLightState> light_states_;
  int64_t current_time_;
};

inline TrafficLightState ParseTrafficLightState(const std::string& s) {
  if (s == "stop") {
    return TrafficLightState::kStop;
  } else if (s == "caution") {
    return TrafficLightState::kCaution;
  } else if (s == "go") {
    return TrafficLightState::kGo;
  } else if (s == "arrow_stop") {
    return TrafficLightState::kArrowStop;
  } else if (s == "arrow_caution") {
    return TrafficLightState::kArrowCaution;
  } else if (s == "arrow_go") {
    return TrafficLightState::kArrowGo;
  } else if (s == "flashing_stop") {
    return TrafficLightState::kFlashingStop;
  } else if (s == "flashing_caution") {
    return TrafficLightState::kFlashingCaution;
  } else {
    return TrafficLightState::kUnknown;
  }
}

}  // namespace nocturne
