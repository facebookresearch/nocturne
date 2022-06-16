// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <variant>
#include <vector>

#include "action.h"
#include "canvas.h"
#include "cyclist.h"
#include "geometry/bvh.h"
#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/point_like.h"
#include "geometry/range_tree_2d.h"
#include "geometry/vector_2d.h"
#include "ndarray.h"
#include "object.h"
#include "object_base.h"
#include "pedestrian.h"
#include "road.h"
#include "static_object.h"
#include "stop_sign.h"
#include "traffic_light.h"
#include "utils/data_utils.h"
#include "vehicle.h"

namespace nocturne {

using json = nlohmann::json;

// Default values for visible features.
constexpr int64_t kMaxVisibleObjects = 20;
constexpr int64_t kMaxVisibleRoadPoints = 300;
constexpr int64_t kMaxVisibleTrafficLights = 20;
constexpr int64_t kMaxVisibleStopSigns = 4;

// Object features are:
// [ valid, distance, azimuth, length, witdh, relative_object_heading,
//   relative_velocity_heading, relative_velocity_speed,
//   object_type (one_hot of 5) ]
constexpr int64_t kObjectFeatureSize = 13;

// RoadPoint features are:
// [ valid, distance, azimuth, distance to the next point, relative azimuth to
//   the next point, road_type (one_hot of 8) ]
constexpr int64_t kRoadPointFeatureSize = 13;

// TrafficLight features are:
// [ valid, distance, azimuth, current_state (one_hot of 9) ]
constexpr int64_t kTrafficLightFeatureSize = 12;

// StopSign features are:
// [ valid, distance, azimuth ]
constexpr int64_t kStopSignsFeatureSize = 3;

// Ego features are:
// [ length, width, speed, target distance, relative_target_azimuth,
//   relative_target_heading, relative_target_speed ]
constexpr int64_t kEgoFeatureSize = 10;

class Scenario : public sf::Drawable {
 public:
  Scenario(const std::string& scenario_path,
           const std::unordered_map<std::string,
                                    std::variant<bool, int64_t, float>>& config)
      : current_time_(std::get<int64_t>(config.at("start_time"))),
        allow_non_vehicles_(std::get<bool>(
            utils::FindWithDefault(config, "allow_non_vehicles", true))),
        spawn_invalid_objects_(std::get<bool>(
            utils::FindWithDefault(config, "spawn_invalid_objects", false))),
        max_visible_objects_(std::get<int64_t>(utils::FindWithDefault(
            config, "max_visible_objects", kMaxVisibleObjects))),
        max_visible_road_points_(std::get<int64_t>(utils::FindWithDefault(
            config, "max_visible_road_points", kMaxVisibleRoadPoints))),
        max_visible_traffic_lights_(std::get<int64_t>(utils::FindWithDefault(
            config, "max_visible_traffic_lights", kMaxVisibleTrafficLights))),
        max_visible_stop_signs_(std::get<int64_t>(utils::FindWithDefault(
            config, "max_visible_stop_signs", kMaxVisibleStopSigns))),
        sample_every_n_(std::get<int64_t>(
            utils::FindWithDefault(config, "sample_every_n", int64_t(1)))),
        road_edge_first_(std::get<bool>(
            utils::FindWithDefault(config, "road_edge_first", false))),
        moving_threshold_(std::get<float>(
            utils::FindWithDefault(config, "moving_threshold", 0.2f))),
        speed_threshold_(std::get<float>(
            utils::FindWithDefault(config, "speed_threshold", 0.05f))) {
    if (!scenario_path.empty()) {
      LoadScenario(scenario_path);
    } else {
      throw std::invalid_argument("No scenario file inputted.");
      // TODO(nl) right now an empty scenario crashes, expectedly
      std::cerr << "No scenario path inputted. Defaulting to an empty scenario."
                << std::endl;
    }
  }

  void LoadScenario(const std::string& scenario_path);

  const std::string& name() const { return name_; }

  void Step(float dt);

  // void removeVehicle(Vehicle* object);
  bool RemoveObject(const Object& object);

  // Returns expert position for obj at timestamp.
  geometry::Vector2D ExpertPosition(const Object& obj,
                                    int64_t timestamp) const {
    return expert_trajectories_.at(obj.id()).at(timestamp);
  }

  // Returns expert heading for obj at timestamp.
  float ExpertHeading(const Object& obj, int64_t timestamp) const {
    return expert_headings_.at(obj.id()).at(timestamp);
  }

  // Returns expert speed for obj at timestamp.
  float ExpertSpeed(const Object& obj, int64_t timestamp) const {
    return expert_speeds_.at(obj.id()).at(timestamp);
  }

  // Returns expert velocity for obj at timestamp.
  geometry::Vector2D ExpertVelocity(const Object& obj,
                                    int64_t timestamp) const {
    const float heading = expert_headings_.at(obj.id()).at(timestamp);
    const float speed = expert_speeds_.at(obj.id()).at(timestamp);
    return geometry::PolarToVector2D(speed, heading);
  }

  std::optional<Action> ExpertAction(const Object& obj,
                                     int64_t timestamp) const;

  std::optional<geometry::Vector2D> ExpertPosShift(const Object& obj,
                                                   int64_t timestamp) const;

  std::optional<float> ExpertHeadingShift(const Object& obj,
                                          int64_t timestamp) const;

  /*********************** Drawing Functions *****************/

  // Computes and returns an `sf::View` of size (`view_height`,
  // `view_width`) (in scenario coordinates), centered around `view_center`
  // (in scenario coordinates) and rotated by `rotation` radians. The view
  // is mapped to a viewport of size (`target_height`, `target_width`)
  // pixels, with a minimum padding of `padding` pixels between the scenario
  // boundaries and the viewport border. A scale-to-fit transform is applied
  // so that the scenario view is scaled to fit the viewport (minus padding)
  // without changing the width:height ratio of the captured view.
  sf::View View(geometry::Vector2D view_center, float rotation,
                float view_height, float view_width, float target_height,
                float target_width, float padding) const;

  // Computes and returns an `sf::View``, mapping the whole scenario into a
  // viewport of size (`target_height`, `target_width`) pixels with a minimum
  // padding of `padding` pixels around the scenario. See the other definition
  // of `sf::View View` for more information.
  sf::View View(float target_height, float target_width, float padding) const;

  // Computes and returns an image of the scenario. The returned image has
  // dimension `img_height` * `img_width` * 4 where 4 is the number of channels
  // (RGBA). If `draw_destinations` is true, the vehicles' goals will be drawn.
  // `padding` (in pixels) can be used to add some padding around the image
  // (included in its width/height). If a `source` object is provided, computes
  // an image of a rectangle of size (`view_height`, `view_width`) centered
  // around the object, rather than of the whole scenario. Besides, if
  // `rotate_with_source` is set to true, the source object will be pointing
  // upwards (+pi/2) in the returned image. Note that the size of the view will
  // be scaled to fit the image size without changing the width:height ratio, so
  // that the resulting image is not distorted.
  NdArray<unsigned char> Image(uint64_t img_height = 1000,
                               uint64_t img_width = 1000,
                               bool draw_destinations = true,
                               float padding = 0.0f, Object* source = nullptr,
                               uint64_t view_height = 200,
                               uint64_t view_width = 200,
                               bool rotate_with_source = true) const;

  // Computes and returns an image of the visible state of the `source` object,
  // ie. the features returned by the `VisibleState` method. See the
  // documentation of `VisibleState` for an explanation of the `view_dist`,
  // `view_angle` and `head_angle` parameters. See the documentation of `Image`
  // for an explanation of the remaining parameters of this function.
  NdArray<unsigned char> EgoVehicleFeaturesImage(
      const Object& source, float view_dist = 120.0f,
      float view_angle = geometry::utils::kPi * 0.8f, float head_angle = 0.0f,
      uint64_t img_height = 1000, uint64_t img_width = 1000,
      float padding = 0.0f, bool draw_destination = true) const;

  // Computes and returns an image of a cone of vision of the `source` object.
  // The image is centered around the `source` object, with a cone of vision of
  // radius `view_dist` and of angle `view_angle` (in radians). The cone points
  // upwards (+pi/2) with an optional tilt `head_angle` (in radians). See the
  // documentation of `Image` for an explanation of the remaining parameters of
  // this function.
  NdArray<unsigned char> EgoVehicleConeImage(
      const Object& source, float view_dist = 120.0f,
      float view_angle = geometry::utils::kPi * 0.8f, float head_angle = 0.0f,
      uint64_t img_height = 1000, uint64_t img_width = 1000,
      float padding = 0.0f, bool draw_destinations = true) const;

  /*********************** State Accessors *******************/

  const std::vector<std::shared_ptr<Vehicle>>& vehicles() const {
    return vehicles_;
  }

  const std::vector<std::shared_ptr<Pedestrian>>& pedestrians() const {
    return pedestrians_;
  }

  const std::vector<std::shared_ptr<Cyclist>>& cyclists() const {
    return cyclists_;
  }

  const std::vector<std::shared_ptr<Object>>& objects() const {
    return objects_;
  }

  const std::vector<std::shared_ptr<Object>>& moving_objects() const {
    return moving_objects_;
  }

  const std::vector<std::shared_ptr<RoadLine>>& road_lines() const {
    return road_lines_;
  }

  NdArray<float> EgoState(const Object& src) const;

  std::unordered_map<std::string, NdArray<float>> VisibleState(
      const Object& src, float view_dist, float view_angle,
      float head_angle = 0.0f, bool padding = false) const;

  NdArray<float> FlattenedVisibleState(const Object& src, float view_dist,
                                       float view_angle,
                                       float head_angle = 0.0f) const;

  int64_t getMaxNumVisibleObjects() const { return max_visible_objects_; }
  int64_t getMaxNumVisibleRoadPoints() const {
    return max_visible_road_points_;
  }
  int64_t getMaxNumVisibleTrafficLights() const {
    return max_visible_traffic_lights_;
  }
  int64_t getMaxNumVisibleStopSigns() const { return max_visible_stop_signs_; }
  int64_t getObjectFeatureSize() const { return kObjectFeatureSize; }
  int64_t getRoadPointFeatureSize() const { return kRoadPointFeatureSize; }
  int64_t getTrafficLightFeatureSize() const {
    return kTrafficLightFeatureSize;
  }
  int64_t getStopSignsFeatureSize() const { return kStopSignsFeatureSize; }
  int64_t getEgoFeatureSize() const { return kEgoFeatureSize; }

 protected:
  void LoadObjects(const json& objects_json);
  void LoadRoads(const json& roads_json);

  // Update the collision status of all objects
  void UpdateCollision();

  std::tuple<std::vector<const ObjectBase*>,
             std::vector<const geometry::PointLike*>,
             std::vector<const ObjectBase*>, std::vector<const ObjectBase*>>
  VisibleObjects(const Object& src, float view_dist, float view_angle,
                 float head_angle = 0.0f) const;

  std::vector<const TrafficLight*> VisibleTrafficLights(
      const Object& src, float view_dist, float view_angle,
      float head_angle = 0.0f) const;

  // Draws the objects contained in `drawables` on the render target `target`.
  // The view `view` is applied to the target before drawing the objects, and
  // the transform `transform` is applied when drawing each object. `drawables`
  // should contain pointers to objects inheriting from sf::Drawable.
  template <typename P>
  void DrawOnTarget(sf::RenderTarget& target, const std::vector<P>& drawables,
                    const sf::View& view, const sf::Transform& transform) const;

  // Computes and returns a list of `sf::Drawable` objects representing the
  // goals/destinations of the `source` vehicle, or of all vehicles in the
  // scenario if `source == nullptr`. Each goal is represented as a circle of
  // radius `radius`.
  std::vector<std::unique_ptr<sf::CircleShape>> VehiclesDestinationsDrawables(
      const Object* source = nullptr, float radius = 2.0f) const;

  // Draws the scenario to a render target. This is used by SFML to know how
  // to draw classes inheriting sf::Drawable.
  void draw(sf::RenderTarget& target, sf::RenderStates states) const override;

  std::string name_;

  int64_t current_time_;

  // Config

  // Whether to use non vehicle objects.
  const bool allow_non_vehicles_ = true;
  // Whether to spawn vehicles that are invalid in the first time step
  const bool spawn_invalid_objects_ = false;

  // TODO(ev) hardcoding, this is the maximum number of vehicles that can be
  // returned in the state
  const int64_t max_visible_objects_ = 20;
  const int64_t max_visible_road_points_ = 300;
  const int64_t max_visible_traffic_lights_ = 20;
  const int64_t max_visible_stop_signs_ = 4;

  // from the set of road points that comprise each polyline, we take
  // every n-th one
  const int64_t sample_every_n_ = 1;

  const bool road_edge_first_ = false;

  // The distance to goal must be greater than this
  // for a vehicle to be included in ObjectsThatMoved
  const float moving_threshold_ = 0.2;
  // The vehicle speed at some point must be greater than this
  // for a vehicle to be included in ObjectsThatMoved
  const float speed_threshold_ = 0.05;

  std::vector<std::shared_ptr<Vehicle>> vehicles_;
  std::vector<std::shared_ptr<Pedestrian>> pedestrians_;
  std::vector<std::shared_ptr<Cyclist>> cyclists_;
  std::vector<std::shared_ptr<Object>> objects_;
  // Rrack the object that moved, useful for figuring out which agents should
  // actually be controlled
  std::vector<std::shared_ptr<Object>> moving_objects_;

  std::vector<std::shared_ptr<geometry::LineSegment>> line_segments_;
  std::vector<std::shared_ptr<RoadLine>> road_lines_;
  std::vector<std::shared_ptr<StopSign>> stop_signs_;
  std::vector<std::shared_ptr<TrafficLight>> traffic_lights_;

  geometry::BVH object_bvh_;        // track objects for collisions
  geometry::BVH line_segment_bvh_;  // track line segments for collisions
  geometry::BVH static_bvh_;        // static objects other than road points
  geometry::RangeTree2d road_point_tree_;  // track road points

  // expert data
  const float expert_dt_ = 0.1f;
  std::vector<std::vector<geometry::Vector2D>> expert_trajectories_;
  std::vector<std::vector<float>> expert_headings_;
  std::vector<std::vector<float>> expert_speeds_;
  std::vector<std::vector<bool>> expert_valid_masks_;

  std::unique_ptr<sf::RenderTexture> image_texture_ = nullptr;
  sf::FloatRect road_network_bounds_;
};

}  // namespace nocturne
