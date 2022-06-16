// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "traffic_light.h"

#include <SFML/Graphics.hpp>
#include <algorithm>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

geometry::ConvexPolygon TrafficLight::BoundingPolygon() const {
  constexpr float kTheta = geometry::utils::kTwoPi / kTrafficLightNumEdges;
  float angle = geometry::utils::kHalfPi;
  std::vector<geometry::Vector2D> vertices;
  vertices.reserve(kTrafficLightNumEdges);
  for (int i = 0; i < kTrafficLightNumEdges; ++i) {
    vertices.push_back(position_ +
                       geometry::PolarToVector2D(kTrafficLightRadius, angle));
    angle = geometry::utils::AngleAdd(angle, kTheta);
  }
  return geometry::ConvexPolygon(std::move(vertices));
}

TrafficLightState TrafficLight::LightState() const {
  const auto it =
      std::lower_bound(timestamps_.cbegin(), timestamps_.cend(), current_time_);
  return it == timestamps_.cend()
             ? TrafficLightState::kUnknown
             : light_states_.at(std::distance(timestamps_.cbegin(), it));
}

void TrafficLight::draw(sf::RenderTarget& target,
                        sf::RenderStates states) const {
  const TrafficLightState state = LightState();
  sf::Color color;
  switch (state) {
    case TrafficLightState::kStop: {
      color = sf::Color::Red;
      break;
    }
    case TrafficLightState::kCaution: {
      color = sf::Color::Yellow;
      break;
    }
    case TrafficLightState::kGo: {
      color = sf::Color::Green;
      break;
    }
    case TrafficLightState::kArrowStop: {
      color = sf::Color::Blue;
      break;
    }
    case TrafficLightState::kArrowCaution: {
      color = sf::Color::Magenta;
      break;
    }
    case TrafficLightState::kArrowGo: {
      color = sf::Color::Cyan;
      break;
    }
    case TrafficLightState::kFlashingStop: {
      color = sf::Color{255, 51, 255};
      break;
    }
    case TrafficLightState::kFlashingCaution: {
      color = sf::Color{255, 153, 51};
      break;
    }
    default: {
      // kUnknown
      color = sf::Color{102, 102, 255};
      break;
    }
  }

  sf::CircleShape pentagon(kTrafficLightRadius, kTrafficLightNumEdges);
  pentagon.setFillColor(color);
  pentagon.setPosition(utils::ToVector2f(position_));
  target.draw(pentagon, states);
}

}  // namespace nocturne
