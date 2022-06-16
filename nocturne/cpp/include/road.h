// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <initializer_list>
#include <string>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/point_like.h"
#include "geometry/vector_2d.h"
#include "static_object.h"
#include "utils/sf_utils.h"

namespace nocturne {

enum class RoadType {
  kNone = 0,
  kLane = 1,
  kRoadLine = 2,
  kRoadEdge = 3,
  kStopSign = 4,
  kCrosswalk = 5,
  kSpeedBump = 6,
  kOther = 7,
};

sf::Color RoadTypeColor(const RoadType& road_type);

class RoadPoint : public sf::Drawable, public geometry::PointLike {
 public:
  RoadPoint() = default;
  RoadPoint(const geometry::Vector2D& position,
            const geometry::Vector2D& neighbor_position, RoadType road_type)
      : position_(position),
        neighbor_position_(neighbor_position),
        road_type_(road_type),
        drawable_(utils::MakeCircleShape(position, 0.5,
                                         RoadTypeColor(road_type), true)) {}

  RoadType road_type() const { return road_type_; }

  const geometry::Vector2D& position() const { return position_; }
  const geometry::Vector2D& neighbor_position() const {
    return neighbor_position_;
  }

  geometry::Vector2D Coordinate() const override { return position_; }
  float X() const override { return position_.x(); }
  float Y() const override { return position_.y(); }

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override {
    target.draw(*drawable_, states);
  }

  const geometry::Vector2D position_;
  // coordinates of the next point in the roadline
  const geometry::Vector2D neighbor_position_;

  const RoadType road_type_ = RoadType::kNone;
  std::unique_ptr<sf::CircleShape> drawable_ = nullptr;
};

// RoadLine is not an Object now.
class RoadLine : public sf::Drawable {
 public:
  RoadLine() = default;

  RoadLine(RoadType road_type,
           const std::initializer_list<geometry::Vector2D>& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(geometry_points),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadLine(RoadType road_type,
           const std::vector<geometry::Vector2D>& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(geometry_points),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadLine(RoadType road_type,
           std::vector<geometry::Vector2D>&& geometry_points,
           int64_t sample_every_n = 1, bool check_collision = false)
      : road_type_(road_type),
        geometry_points_(std::move(geometry_points)),
        sample_every_n_(sample_every_n),
        check_collision_(check_collision) {
    InitRoadPoints();
    InitRoadLineGraphics();
  }

  RoadType road_type() const { return road_type_; }

  int64_t sample_every_n() const { return sample_every_n_; }

  const std::vector<RoadPoint>& road_points() const { return road_points_; }

  const std::vector<geometry::Vector2D>& geometry_points() const {
    return geometry_points_;
  }

  bool check_collision() const { return check_collision_; }

  sf::Color Color() const;

 protected:
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  void InitRoadPoints();
  void InitRoadLineGraphics();

  const RoadType road_type_ = RoadType::kNone;
  std::vector<geometry::Vector2D> geometry_points_;

  // Sample rate from geometry points.
  const int64_t sample_every_n_ = 1;
  std::vector<RoadPoint> road_points_;

  const bool check_collision_ = false;

  std::vector<sf::Vertex> graphic_points_;
};

inline RoadType ParseRoadType(const std::string& s) {
  if (s == "none") {
    return RoadType::kNone;
  } else if (s == "lane") {
    return RoadType::kLane;
  } else if (s == "road_line") {
    return RoadType::kRoadLine;
  } else if (s == "road_edge") {
    return RoadType::kRoadEdge;
  } else if (s == "stop_sign") {
    return RoadType::kStopSign;
  } else if (s == "crosswalk") {
    return RoadType::kCrosswalk;
  } else if (s == "speed_bump") {
    return RoadType::kSpeedBump;
  } else {
    return RoadType::kOther;
  }
}

}  // namespace nocturne
