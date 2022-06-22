// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "road.h"

#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"

namespace nocturne {

sf::Color RoadTypeColor(const RoadType& road_type) {
  switch (road_type) {
    case RoadType::kLane: {
      return sf::Color::Yellow;
    }
    case RoadType::kRoadLine: {
      return sf::Color::Blue;
    }
    case RoadType::kRoadEdge: {
      return sf::Color::Green;
    }
    case RoadType::kStopSign: {
      return sf::Color::Red;
    }
    case RoadType::kCrosswalk: {
      return sf::Color::Magenta;
    }
    case RoadType::kSpeedBump: {
      return sf::Color::Cyan;
    }
    default: {
      return sf::Color::Transparent;
    }
  };
}

sf::Color RoadLine::Color() const { return RoadTypeColor(road_type_); }

void RoadLine::draw(sf::RenderTarget& target, sf::RenderStates states) const {
  target.draw(graphic_points_.data(), graphic_points_.size(), sf::LineStrip,
              states);
}

void RoadLine::InitRoadPoints() {
  const int64_t num_segments = geometry_points_.size() - 1;
  const int64_t num_sampled_points =
      (num_segments + sample_every_n_ - 1) / sample_every_n_ + 1;
  road_points_.reserve(num_sampled_points);
  for (int64_t i = 0; i < num_sampled_points - 2; ++i) {
    road_points_.emplace_back(geometry_points_[i * sample_every_n_],
                              geometry_points_[(i + 1) * sample_every_n_],
                              road_type_);
  }
  const int64_t p = (num_sampled_points - 2) * sample_every_n_;
  road_points_.emplace_back(geometry_points_[p], geometry_points_.back(),
                            road_type_);
  // Use itself as neighbor for the last point.
  road_points_.emplace_back(geometry_points_.back(), geometry_points_.back(),
                            road_type_);
}

void RoadLine::InitRoadLineGraphics() {
  const int64_t n = geometry_points_.size();
  graphic_points_.reserve(n);
  for (const geometry::Vector2D& p : geometry_points_) {
    graphic_points_.emplace_back(sf::Vertex(utils::ToVector2f(p), Color()));
  }
}

}  // namespace nocturne
