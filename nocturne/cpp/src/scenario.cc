// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "scenario.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <type_traits>

#include "geometry/aabb_interface.h"
#include "geometry/geometry_utils.h"
#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/polygon.h"
#include "geometry/vector_2d.h"
#include "utils/sf_utils.h"
#include "view_field.h"

namespace nocturne {

namespace {

template <class T>
bool RemoveObjectImpl(const Object& object, std::vector<T>& objects) {
  const int64_t id = object.id();
  const auto it = std::find_if(
      objects.begin(), objects.end(),
      [id](const std::shared_ptr<Object>& obj) { return obj->id() == id; });
  if (it == objects.end()) {
    return false;
  }
  objects.erase(it);
  return true;
}

std::vector<const ObjectBase*> VisibleCandidates(const geometry::BVH& bvh,
                                                 const Object& src,
                                                 const ViewField& vf) {
  std::vector<const ObjectBase*> objects =
      bvh.IntersectionCandidates<ObjectBase>(vf);
  auto it = std::find(objects.begin(), objects.end(),
                      dynamic_cast<const ObjectBase*>(&src));
  if (it != objects.end()) {
    std::swap(*it, objects.back());
    objects.pop_back();
  }
  return objects;
}

void VisibleRoadPoints(const Object& src,
                       const std::vector<const ObjectBase*>& objects,
                       std::vector<const geometry::PointLike*>& road_points) {
  const int64_t n = road_points.size();
  const auto [x, y] = geometry::utils::PackCoordinates(road_points);
  std::vector<geometry::utils::MaskType> mask(n, 1);
  for (const ObjectBase* obj : objects) {
    if (!obj->can_block_sight()) {
      continue;
    }
    const std::vector<geometry::utils::MaskType> block_mask =
        geometry::BatchIntersects(obj->BoundingPolygon(), src.position(), x, y);
    for (int64_t i = 0; i < n; ++i) {
      // Use bitwise operation to get better performance.
      // Use (^1) for not operation.
      mask[i] &= (block_mask[i] ^ 1);
    }
  }
  const int64_t pivot = utils::MaskedPartition(mask, road_points);
  road_points.resize(pivot);
}

template <class ObjType>
std::vector<std::pair<const ObjType*, float>> NearestK(
    const Object& src, const std::vector<const ObjType*>& objects, int64_t k) {
  const geometry::Vector2D& src_pos = src.position();
  const int64_t n = objects.size();
  std::vector<std::pair<const ObjType*, float>> ret;
  ret.reserve(n);
  for (const ObjType* obj : objects) {
    if constexpr (std::is_same<ObjType, geometry::PointLike>::value) {
      ret.emplace_back(obj, geometry::Distance(src_pos, obj->Coordinate()));
    } else {
      ret.emplace_back(obj, geometry::Distance(src_pos, obj->position()));
    }
  }
  const auto cmp = [](const std::pair<const ObjType*, float>& lhs,
                      const std::pair<const ObjType*, float>& rhs) {
    return lhs.second < rhs.second;
  };
  if (n <= k) {
    std::sort(ret.begin(), ret.end(), cmp);
  } else {
    utils::PartialSort(ret.begin(), ret.begin() + k, ret.end(), cmp);
    ret.resize(k);
  }
  return ret;
}

template <class PointType>
std::vector<std::pair<const PointType*, float>> NearestKRoadPoints(
    const Object& src, const std::vector<const PointType*>& points, int64_t k,
    bool road_edge_first) {
  if (!road_edge_first) {
    return NearestK(src, points, k);
  }
  const geometry::Vector2D& src_pos = src.position();
  const int64_t n = points.size();
  std::vector<std::pair<const PointType*, float>> ret;
  std::vector<geometry::utils::MaskType> mask(n);
  ret.reserve(n);
  for (int64_t i = 0; i < n; ++i) {
    const PointType* p = points[i];
    mask[i] =
        (dynamic_cast<const RoadPoint*>(p))->road_type() == RoadType::kRoadEdge;
    ret.emplace_back(p, geometry::Distance(src_pos, p->Coordinate()));
  }
  const auto cmp = [](const std::pair<const PointType*, float>& lhs,
                      const std::pair<const PointType*, float>& rhs) {
    return lhs.second < rhs.second;
  };
  const int64_t pivot = utils::MaskedPartition(mask, ret);
  if (pivot >= k) {
    utils::PartialSort(ret.begin(), ret.begin() + k, ret.begin() + pivot, cmp);
  } else {
    std::sort(ret.begin(), ret.begin() + pivot, cmp);
    if (n <= k) {
      std::sort(ret.begin() + pivot, ret.end(), cmp);
    } else {
      utils::PartialSort(ret.begin() + pivot, ret.begin() + k, ret.end(), cmp);
    }
  }
  if (n > k) {
    ret.resize(k);
  }
  return ret;
}

void ExtractObjectFeature(const Object& src, const Object& obj, float dis,
                          float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const float relative_heading =
      geometry::utils::AngleSub(obj.heading(), src.heading());
  const geometry::Vector2D relative_velocity = obj.Velocity() - src.Velocity();
  const int64_t obj_type = static_cast<int64_t>(obj.Type());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  feature[3] = obj.length();
  feature[4] = obj.width();
  feature[5] = relative_heading;
  feature[6] =
      geometry::utils::AngleSub(relative_velocity.Angle(), src.heading());
  feature[7] = relative_velocity.Norm();
  // One-hot vector for object_type, assume feature is initially 0.
  feature[8 + obj_type] = 1.0f;
}

void ExtractRoadPointFeature(const Object& src, const RoadPoint& obj, float dis,
                             float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t road_type = static_cast<int64_t>(obj.road_type());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  const geometry::Vector2D neighbor_vec =
      obj.neighbor_position() - obj.position();
  const float neighbor_dis = neighbor_vec.Norm();
  const float neighbor_azimuth =
      geometry::utils::AngleSub(neighbor_vec.Angle(), src.heading());
  feature[3] = neighbor_dis;
  feature[4] = neighbor_azimuth;
  // One-hot vector for road_type, assume feature is initially 0.
  feature[5 + road_type] = 1.0f;
}

void ExtractTrafficLightFeature(const Object& src, const TrafficLight& obj,
                                float dis, float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  const int64_t light_state = static_cast<int64_t>(obj.LightState());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
  // One-hot vector for light_state, assume feature is initially 0.
  feature[3 + light_state] = 1.0f;
}

void ExtractStopSignFeature(const Object& src, const StopSign& obj, float dis,
                            float* feature) {
  const float azimuth = geometry::utils::AngleSub(
      (obj.position() - src.position()).Angle(), src.heading());
  feature[0] = 1.0f;  // Valid
  feature[1] = dis;
  feature[2] = azimuth;
}

}  // namespace

void Scenario::LoadScenario(const std::string& scenario_path) {
  std::ifstream data(scenario_path);
  if (!data.is_open()) {
    throw std::invalid_argument("Scenario file couldn't be opened: " +
                                scenario_path);
  }

  json j;
  data >> j;
  name_ = j["name"];

  LoadObjects(j["objects"]);
  LoadRoads(j["roads"]);

  // Now handle the traffic light states
  for (const auto& tl : j["tl_states"]) {
    // Lane positions don't move so we can just use the first
    // element
    float x_pos = float(tl["x"][0]);
    float y_pos = float(tl["y"][0]);
    std::vector<int64_t> validTimes;
    std::vector<TrafficLightState> lightStates;

    for (size_t i = 0; i < tl["state"].size(); i++) {
      // TODO(ev) do this more compactly
      const TrafficLightState light_state =
          ParseTrafficLightState(tl["state"][i]);
      lightStates.push_back(light_state);
      validTimes.push_back(int(tl["time_index"][i]));
    }
    std::shared_ptr<TrafficLight> traffic_light =
        std::make_shared<TrafficLight>(geometry::Vector2D(x_pos, y_pos),
                                       validTimes, lightStates, current_time_);
    traffic_lights_.push_back(traffic_light);
  }

  std::vector<const RoadPoint*> road_points;
  for (const auto& road_line : road_lines_) {
    for (const auto& road_point : road_line->road_points()) {
      road_points.push_back(&road_point);
    }
  }
  road_point_tree_.Reset(road_points);

  std::vector<const geometry::AABBInterface*> static_objects;
  for (const auto& obj : stop_signs_) {
    static_objects.push_back(
        dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  for (const auto& obj : traffic_lights_) {
    static_objects.push_back(
        dynamic_cast<const geometry::AABBInterface*>(obj.get()));
  }
  static_bvh_.Reset(static_objects);

  // Update collision to check for collisions of any vehicles at initialization
  UpdateCollision();
}

void Scenario::Step(float dt) {
  current_time_ += static_cast<int>(dt / 0.1);  // TODO(ev) hardcoding
  for (auto& object : objects_) {
    // reset the collision flags for the objects before stepping
    // we do not want to label a vehicle as persistently having collided
    object->ResetCollision();
    if (!object->expert_control()) {
      object->Step(dt);
    } else {
      const int64_t obj_id = object->id();
      object->set_position(expert_trajectories_.at(obj_id).at(current_time_));
      object->set_heading(expert_headings_.at(obj_id).at(current_time_));
      object->set_speed(expert_speeds_.at(obj_id).at(current_time_));
    }
  }
  for (auto& object : traffic_lights_) {
    object->set_current_time(current_time_);
  }

  // update the vehicle bvh
  object_bvh_.Reset(objects_);
  UpdateCollision();
}

void Scenario::UpdateCollision() {
  // check vehicle-vehicle collisions
  for (auto& obj1 : objects_) {
    std::vector<const Object*> candidates =
        object_bvh_.IntersectionCandidates<Object>(*obj1);
    for (const auto* obj2 : candidates) {
      if (obj1->id() == obj2->id()) {
        continue;
      }
      if (!obj1->can_be_collided() || !obj2->can_be_collided()) {
        continue;
      }
      if (!obj1->check_collision() && !obj2->check_collision()) {
        continue;
      }
      if (geometry::Intersects(obj1->BoundingPolygon(),
                               obj2->BoundingPolygon())) {
        obj1->set_collided(true);
        obj1->set_collision_type(CollisionType::kVehicleVehicleCollision);
        const_cast<Object*>(obj2)->set_collided(true);
      }
    }
  }
  // check vehicle-lane segment collisions
  for (auto& obj : objects_) {
    std::vector<const geometry::LineSegment*> candidates =
        line_segment_bvh_.IntersectionCandidates<geometry::LineSegment>(*obj);
    for (const auto* seg : candidates) {
      if (geometry::Intersects(obj->BoundingPolygon(), *seg)) {
        obj->set_collision_type(CollisionType::kVehicleRoadEdgeCollision);
        obj->set_collided(true);
      }
    }
  }
}

std::tuple<std::vector<const ObjectBase*>,
           std::vector<const geometry::PointLike*>,
           std::vector<const ObjectBase*>, std::vector<const ObjectBase*>>
Scenario::VisibleObjects(const Object& src, float view_dist, float view_angle,
                         float head_angle) const {
  const float heading = geometry::utils::AngleAdd(src.heading(), head_angle);
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  std::vector<const ObjectBase*> objects =
      VisibleCandidates(object_bvh_, src, vf);
  std::vector<const geometry::PointLike*> road_points =
      road_point_tree_.RangeSearch<geometry::PointLike>(vf);
  const std::vector<const ObjectBase*> static_candidates =
      VisibleCandidates(static_bvh_, src, vf);

  std::vector<const ObjectBase*> traffic_lights;
  std::vector<const ObjectBase*> stop_signs;
  for (const ObjectBase* obj : static_candidates) {
    const StaticObject* obj_ptr = dynamic_cast<const StaticObject*>(obj);
    if (obj_ptr->Type() == StaticObjectType::kTrafficLight) {
      traffic_lights.push_back(dynamic_cast<const ObjectBase*>(obj));
    } else if (obj_ptr->Type() == StaticObjectType::kStopSign) {
      stop_signs.push_back(dynamic_cast<const ObjectBase*>(obj));
    }
  }

  vf.FilterVisibleObjects(objects);
  vf.FilterVisiblePoints(road_points);
  VisibleRoadPoints(src, objects, road_points);
  vf.FilterVisibleNonblockingObjects(traffic_lights);
  vf.FilterVisibleNonblockingObjects(stop_signs);

  return std::make_tuple(objects, road_points, traffic_lights, stop_signs);
}

std::vector<const TrafficLight*> Scenario::VisibleTrafficLights(
    const Object& src, float view_dist, float view_angle,
    float head_angle) const {
  std::vector<const TrafficLight*> ret;

  const float heading = geometry::utils::AngleAdd(src.heading(), head_angle);
  const geometry::Vector2D& position = src.position();
  const ViewField vf(position, view_dist, heading, view_angle);

  // Assume limited number of TrafficLights, check all of them.
  std::vector<const ObjectBase*> objects;
  objects.reserve(traffic_lights_.size());
  for (const auto& obj : traffic_lights_) {
    objects.push_back(dynamic_cast<const ObjectBase*>(obj.get()));
  }
  objects = vf.VisibleNonblockingObjects(objects);

  ret.reserve(objects.size());
  for (const ObjectBase* obj : objects) {
    ret.push_back(dynamic_cast<const TrafficLight*>(obj));
  }

  return ret;
}

NdArray<float> Scenario::EgoState(const Object& src) const {
  NdArray<float> state({kEgoFeatureSize}, 0.0f);

  const float src_heading = src.heading();
  const geometry::Vector2D d = src.target_position() - src.position();
  const float target_dist = d.Norm();
  const float target_azimuth =
      geometry::utils::AngleSub(d.Angle(), src_heading);
  const float target_heading =
      geometry::utils::AngleSub(src.target_heading(), src_heading);
  const float target_speed = src.target_speed() - src.speed();

  float* state_data = state.DataPtr();
  state_data[0] = src.length();
  state_data[1] = src.width();
  state_data[2] = src.speed();
  state_data[3] = target_dist;
  state_data[4] = target_azimuth;
  state_data[5] = target_heading;
  state_data[6] = target_speed;
  state_data[7] = src.acceleration();
  state_data[8] = src.steering();
  state_data[9] = src.head_angle();

  return state;
}

std::unordered_map<std::string, NdArray<float>> Scenario::VisibleState(
    const Object& src, float view_dist, float view_angle, float head_angle,
    bool padding) const {
  const auto [objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(src, view_dist, view_angle, head_angle);
  const auto o_targets = NearestK(src, objects, max_visible_objects_);
  const auto r_targets = NearestKRoadPoints(
      src, road_points, max_visible_road_points_, road_edge_first_);
  const auto t_targets =
      NearestK(src, traffic_lights, max_visible_traffic_lights_);
  const auto s_targets = NearestK(src, stop_signs, max_visible_stop_signs_);

  const int64_t num_objects =
      padding ? max_visible_objects_ : static_cast<int64_t>(o_targets.size());
  const int64_t num_road_points = padding
                                      ? max_visible_road_points_
                                      : static_cast<int64_t>(r_targets.size());
  const int64_t num_traffic_lights =
      padding ? max_visible_traffic_lights_
              : static_cast<int64_t>(t_targets.size());
  const int64_t num_stop_signs = padding
                                     ? max_visible_stop_signs_
                                     : static_cast<int64_t>(s_targets.size());

  NdArray<float> o_feature({num_objects, kObjectFeatureSize}, 0.0f);
  NdArray<float> r_feature({num_road_points, kRoadPointFeatureSize}, 0.0f);
  NdArray<float> t_feature({num_traffic_lights, kTrafficLightFeatureSize},
                           0.0f);
  NdArray<float> s_feature({num_stop_signs, kStopSignsFeatureSize}, 0.0f);

  // Object feature.
  float* o_feature_ptr = o_feature.DataPtr();
  for (const auto [obj, dis] : o_targets) {
    ExtractObjectFeature(src, *(dynamic_cast<const Object*>(obj)), dis,
                         o_feature_ptr);
    o_feature_ptr += kObjectFeatureSize;
  }

  // RoadPoint feature.
  float* r_feature_ptr = r_feature.DataPtr();
  for (const auto [obj, dis] : r_targets) {
    ExtractRoadPointFeature(src, *(dynamic_cast<const RoadPoint*>(obj)), dis,
                            r_feature_ptr);
    r_feature_ptr += kRoadPointFeatureSize;
  }

  // TrafficLight feature.
  float* t_feature_ptr = t_feature.DataPtr();
  for (const auto [obj, dis] : t_targets) {
    ExtractTrafficLightFeature(src, *(dynamic_cast<const TrafficLight*>(obj)),
                               dis, t_feature_ptr);
    t_feature_ptr += kTrafficLightFeatureSize;
  }

  // StopSign feature.
  float* s_feature_ptr = s_feature.DataPtr();
  for (const auto [obj, dis] : s_targets) {
    ExtractStopSignFeature(src, *(dynamic_cast<const StopSign*>(obj)), dis,
                           s_feature_ptr);
    s_feature_ptr += kStopSignsFeatureSize;
  }

  return {{"objects", o_feature},
          {"road_points", r_feature},
          {"traffic_lights", t_feature},
          {"stop_signs", s_feature}};
}

NdArray<float> Scenario::FlattenedVisibleState(const Object& src,
                                               float view_dist,
                                               float view_angle,
                                               float head_angle) const {
  const int64_t kObjectFeatureStride = 0;
  const int64_t kRoadPointFeatureStride =
      kObjectFeatureStride + max_visible_objects_ * kObjectFeatureSize;
  const int64_t kTrafficLightFeatureStride =
      kRoadPointFeatureStride +
      max_visible_road_points_ * kRoadPointFeatureSize;
  const int64_t kStopSignFeatureStride =
      kTrafficLightFeatureStride +
      max_visible_traffic_lights_ * kTrafficLightFeatureSize;
  const int64_t kFeatureSize =
      kStopSignFeatureStride + max_visible_stop_signs_ * kStopSignsFeatureSize;

  const auto [objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(src, view_dist, view_angle, head_angle);

  const auto o_targets = NearestK(src, objects, max_visible_objects_);
  const auto r_targets = NearestKRoadPoints(
      src, road_points, max_visible_road_points_, road_edge_first_);
  const auto t_targets =
      NearestK(src, traffic_lights, max_visible_traffic_lights_);
  const auto s_targets = NearestK(src, stop_signs, max_visible_stop_signs_);

  NdArray<float> state({kFeatureSize}, 0.0f);

  // Object feature.
  float* o_feature_ptr = state.DataPtr() + kObjectFeatureStride;
  for (const auto [obj, dis] : o_targets) {
    ExtractObjectFeature(src, *(dynamic_cast<const Object*>(obj)), dis,
                         o_feature_ptr);
    o_feature_ptr += kObjectFeatureSize;
  }

  // RoadPoint feature.
  float* r_feature_ptr = state.DataPtr() + kRoadPointFeatureStride;
  for (const auto [obj, dis] : r_targets) {
    ExtractRoadPointFeature(src, *(dynamic_cast<const RoadPoint*>(obj)), dis,
                            r_feature_ptr);
    r_feature_ptr += kRoadPointFeatureSize;
  }

  // TrafficLight feature.
  float* t_feature_ptr = state.DataPtr() + kTrafficLightFeatureStride;
  for (const auto [obj, dis] : t_targets) {
    ExtractTrafficLightFeature(src, *(dynamic_cast<const TrafficLight*>(obj)),
                               dis, t_feature_ptr);
    t_feature_ptr += kTrafficLightFeatureSize;
  }

  // StopSign feature.
  float* s_feature_ptr = state.DataPtr() + kStopSignFeatureStride;
  for (const auto [obj, dis] : s_targets) {
    ExtractStopSignFeature(src, *(dynamic_cast<const StopSign*>(obj)), dis,
                           s_feature_ptr);
    s_feature_ptr += kStopSignsFeatureSize;
  }

  return state;
}

std::optional<Action> Scenario::ExpertAction(const Object& obj,
                                             int64_t timestamp) const {
  const std::vector<float>& cur_headings = expert_headings_.at(obj.id());
  const std::vector<float>& cur_speeds = expert_speeds_.at(obj.id());
  const std::vector<bool>& valid_mask = expert_valid_masks_.at(obj.id());
  const int64_t trajectory_length = valid_mask.size();

  if (timestamp < 0 || timestamp > trajectory_length - 1) {
    return std::nullopt;
  }
  if (!valid_mask[timestamp] || !valid_mask[timestamp + 1]) {
    return std::nullopt;
  }

  // compute acceleration
  // a_t = (v_{t+1} - v_t) / dt
  const float acceleration =
      (cur_speeds[timestamp + 1] - cur_speeds[timestamp]) / expert_dt_;

  // compute steering
  // cf Object::KinematicBicycleStep
  // w = (h_{t+1} - h_t) / dt = v * tan(steering) * cos(beta) / length
  // -> solve for steering s_t, we get s_t = atan(2C / sqrt(4 - C^2)) + k * pi
  // with C = 2 * length * (h_{t+1} - h_t) / (dt * (v_t + v_{t+1}))
  const float w = geometry::utils::AngleSub(cur_headings[timestamp + 1],
                                            cur_headings[timestamp]) /
                  expert_dt_;
  const float C = 2.0f * obj.length() * w /
                  (cur_speeds[timestamp + 1] + cur_speeds[timestamp]);
  const float steering = std::atan(2.0f * C / std::sqrt(4 - C * C));

  // return action
  return std::make_optional<Action>(acceleration, steering, 0.0);
}

std::optional<geometry::Vector2D> Scenario::ExpertPosShift(
    const Object& obj, int64_t timestamp) const {
  const std::vector<geometry::Vector2D>& cur_positions =
      expert_trajectories_.at(obj.id());
  const std::vector<bool>& valid_mask = expert_valid_masks_.at(obj.id());
  const int64_t trajectory_length = valid_mask.size();

  if (timestamp < 0 || timestamp > trajectory_length - 1) {
    return std::nullopt;
  }
  if (!valid_mask[timestamp] || !valid_mask[timestamp + 1]) {
    return std::nullopt;
  }

  // compute acceleration
  // a_t = (v_{t+1} - v_t) / dt
  geometry::Vector2D pos_shift =
      (cur_positions[timestamp + 1] - cur_positions[timestamp]);

  // return action
  return pos_shift;
}

std::optional<float> Scenario::ExpertHeadingShift(const Object& obj,
                                                  int64_t timestamp) const {
  const std::vector<float>& cur_heading = expert_headings_.at(obj.id());
  const std::vector<bool>& valid_mask = expert_valid_masks_.at(obj.id());
  const int64_t trajectory_length = valid_mask.size();

  if (timestamp < 0 || timestamp > trajectory_length - 1) {
    return std::nullopt;
  }
  if (!valid_mask[timestamp] || !valid_mask[timestamp + 1]) {
    return std::nullopt;
  }

  // compute acceleration
  // a_t = (v_{t+1} - v_t) / dt
  float heading_shift = geometry::utils::AngleSub(cur_heading[timestamp + 1],
                                                  cur_heading[timestamp]);

  // return action
  return heading_shift;
}

// O(N) time remove.
bool Scenario::RemoveObject(const Object& object) {
  if (!RemoveObjectImpl(object, objects_)) {
    return false;
  }
  switch (object.Type()) {
    case ObjectType::kVehicle: {
      RemoveObjectImpl(object, vehicles_);
      break;
    }
    case ObjectType::kPedestrian: {
      RemoveObjectImpl(object, pedestrians_);
      break;
    }
    case ObjectType::kCyclist: {
      RemoveObjectImpl(object, cyclists_);
      break;
    }
    default: {
      break;
    }
  }
  RemoveObjectImpl(object, moving_objects_);
  object_bvh_.Reset(objects_);
  return true;
}

/*********************** Drawing Functions *****************/

sf::View Scenario::View(geometry::Vector2D view_center, float rotation,
                        float view_height, float view_width,
                        float target_height, float target_width,
                        float padding) const {
  // create view (note that the y coordinates and the view rotation are flipped
  // because the scenario is always drawn with a horizontally flip transform)
  const sf::Vector2f center = utils::ToVector2f(view_center, /*flip_y=*/true);
  const sf::Vector2f size(view_width, view_height);
  sf::View view(center, size);
  view.setRotation(-rotation);

  // compute the placement (viewport) of the view within its render target of
  // size (target_width, target_height), so that it is centered with adequate
  // padding and that proportions are keeped (ie. scale-to-fit)
  const float min_ratio = std::min((target_width - 2 * padding) / view_width,
                                   (target_height - 2 * padding) / view_height);
  const float real_view_width_ratio = min_ratio * view_width / target_width;
  const float real_view_height_ratio = min_ratio * view_height / target_height;
  const sf::FloatRect viewport(
      /*left=*/(1.0f - real_view_width_ratio) / 2.0f,
      /*top=*/(1.0f - real_view_height_ratio) / 2.0f,
      /*width=*/real_view_width_ratio,
      /*height=*/real_view_height_ratio);
  view.setViewport(viewport);

  return view;
}

sf::View Scenario::View(float target_height, float target_width,
                        float padding) const {
  // compute center and size of view based on known scenario bounds
  const geometry::Vector2D view_center(
      road_network_bounds_.left + road_network_bounds_.width / 2.0f,
      road_network_bounds_.top + road_network_bounds_.height / 2.0f);
  const float view_width = road_network_bounds_.width;
  const float view_height = road_network_bounds_.height;

  // build the view from overloaded function
  return View(view_center, 0.0f, view_height, view_width, target_height,
              target_width, padding);
}

std::vector<std::unique_ptr<sf::CircleShape>>
Scenario::VehiclesDestinationsDrawables(const Object* source,
                                        float radius) const {
  std::vector<std::unique_ptr<sf::CircleShape>> target_position_drawables;
  if (source == nullptr) {
    for (const auto& obj : objects_) {
      auto circle_shape = utils::MakeCircleShape(obj->target_position(), radius,
                                                 obj->color(), false);
      target_position_drawables.push_back(std::move(circle_shape));
    }
  } else {
    auto circle_shape = utils::MakeCircleShape(source->target_position(),
                                               radius, source->color(), false);
    target_position_drawables.push_back(std::move(circle_shape));
  }
  return target_position_drawables;
}

template <typename P>
void Scenario::DrawOnTarget(sf::RenderTarget& target,
                            const std::vector<P>& drawables,
                            const sf::View& view,
                            const sf::Transform& transform) const {
  target.setView(view);
  for (const P& drawable : drawables) {
    target.draw(*drawable, transform);
  }
}

void Scenario::draw(sf::RenderTarget& target,
                    sf::RenderStates /*states*/) const {
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);
  sf::View view =
      View(target.getSize().y, target.getSize().x, /*padding=*/30.0f);
  DrawOnTarget(target, road_lines_, view, horizontal_flip);
  DrawOnTarget(target, objects_, view, horizontal_flip);
  DrawOnTarget(target, traffic_lights_, view, horizontal_flip);
  DrawOnTarget(target, stop_signs_, view, horizontal_flip);
  DrawOnTarget(target, VehiclesDestinationsDrawables(), view, horizontal_flip);
}

NdArray<unsigned char> Scenario::Image(uint64_t img_height, uint64_t img_width,
                                       bool draw_target_positions,
                                       float padding, Object* source,
                                       uint64_t view_height,
                                       uint64_t view_width,
                                       bool rotate_with_source) const {
  // construct transform (flip the y-axis)
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);

  // construct view
  sf::View view;
  if (source == nullptr) {
    // if no source object is provided, get the entire scenario
    view = View(img_height, img_width, padding);
  } else {
    // otherwise get a region around the source object, possibly rotated
    const float rotation =
        rotate_with_source ? geometry::utils::Degrees(source->heading()) - 90.0f
                           : 0.0f;
    view = View(source->position(), rotation, view_height, view_width,
                img_height, img_width, padding);
  }

  // create canvas and draw objects
  Canvas canvas(img_height, img_width);

  DrawOnTarget(canvas, road_lines_, view, horizontal_flip);
  DrawOnTarget(canvas, objects_, view, horizontal_flip);
  DrawOnTarget(canvas, traffic_lights_, view, horizontal_flip);
  DrawOnTarget(canvas, stop_signs_, view, horizontal_flip);

  if (draw_target_positions) {
    DrawOnTarget(canvas, VehiclesDestinationsDrawables(source), view,
                 horizontal_flip);
  }

  return canvas.AsNdArray();
}

NdArray<unsigned char> Scenario::EgoVehicleConeImage(
    const Object& source, float view_dist, float view_angle, float head_angle,
    uint64_t img_height, uint64_t img_width, float padding,
    bool draw_target_positions) const {
  // define transforms
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);
  sf::Transform obstruction_transform = horizontal_flip;
  obstruction_transform.rotate(-geometry::utils::Degrees(source.heading()) +
                               90.0f);

  // define views
  const float rotation = geometry::utils::Degrees(source.heading()) - 90.0f;
  const sf::View scenario_view =
      View(source.position(), rotation, 2.0f * view_dist, 2.0f * view_dist,
           img_height, img_width, padding);
  const sf::View cone_view =
      View(geometry::Vector2D(0.0f, 0.0f), 0.0f, 2.0f * view_dist,
           2.0f * view_dist, img_height, img_width, padding);

  // create canvas
  Canvas canvas(img_height, img_width, sf::Color::Black);

  // draw background
  auto background = std::make_unique<sf::RectangleShape>(
      sf::Vector2f(2.0f * view_dist, 2.0f * view_dist));
  background->setOrigin(view_dist, view_dist);
  background->setPosition(0.0f, 0.0f);
  background->setFillColor(sf::Color(50, 50, 50));
  std::vector<std::unique_ptr<sf::RectangleShape>> background_drawable;
  background_drawable.push_back(std::move(background));
  DrawOnTarget(canvas, background_drawable, cone_view, horizontal_flip);

  // draw roads and objects
  DrawOnTarget(canvas, road_lines_, scenario_view, horizontal_flip);
  DrawOnTarget(canvas, objects_, scenario_view, horizontal_flip);

  // draw target_positions
  if (draw_target_positions) {
    DrawOnTarget(canvas, VehiclesDestinationsDrawables(&source), scenario_view,
                 horizontal_flip);
  }

  // draw obstructions
  for (const auto& obj : objects_) {
    if (obj->id() == source.id() || !obj->can_block_sight()) continue;
    const float dist_to_source = (obj->position() - source.position()).Norm();
    if (dist_to_source > view_dist + obj->Radius()) continue;

    const auto obj_lines = obj->BoundingPolygon().Edges();
    auto obscurity_drawables =
        utils::MakeObstructionShape(source.position(), obj_lines, view_dist);
    DrawOnTarget(canvas, obscurity_drawables, cone_view, obstruction_transform);
  }

  // draw stop signs and traffic lights (not subject to obstructions)
  DrawOnTarget(canvas, traffic_lights_, scenario_view, horizontal_flip);
  DrawOnTarget(canvas, stop_signs_, scenario_view, horizontal_flip);

  // draw cone
  auto cone_drawables =
      utils::MakeInvertedConeShape(view_dist, view_angle, head_angle);
  DrawOnTarget(canvas, cone_drawables, cone_view, horizontal_flip);

  return canvas.AsNdArray();
}

NdArray<unsigned char> Scenario::EgoVehicleFeaturesImage(
    const Object& source, float view_dist, float view_angle, float head_angle,
    uint64_t img_height, uint64_t img_width, float padding,
    bool draw_target_position) const {
  sf::Transform horizontal_flip;
  horizontal_flip.scale(1, -1);

  const float rotation = geometry::utils::Degrees(source.heading()) - 90.0f;
  sf::View view = View(source.position(), rotation, 2.0f * view_dist,
                       2.0f * view_dist, img_height, img_width, padding);

  Canvas canvas(img_height, img_width);

  // TODO(nl) remove code duplication and linear overhead
  const auto [kinetic_objects, road_points, traffic_lights, stop_signs] =
      VisibleObjects(source, view_dist, view_angle, head_angle);
  std::vector<const sf::Drawable*> drawables;

  for (const auto [obj, dist] : NearestKRoadPoints(
           source, road_points, max_visible_road_points_, road_edge_first_)) {
    drawables.emplace_back(dynamic_cast<const RoadPoint*>(obj));
  }
  for (const auto& [objects, limit] :
       std::vector<std::pair<std::vector<const ObjectBase*>, int64_t>>{
           // {road_points, max_visible_road_points_},
           {kinetic_objects, max_visible_objects_},
           {traffic_lights, max_visible_traffic_lights_},
           {stop_signs, max_visible_stop_signs_},
       }) {
    for (const auto [obj, dist] : NearestK(source, objects, limit)) {
      drawables.emplace_back(obj);
    }
  }
  // draw source
  drawables.emplace_back(&source);
  DrawOnTarget(canvas, drawables, view, horizontal_flip);
  if (draw_target_position) {
    DrawOnTarget(canvas, VehiclesDestinationsDrawables(&source), view,
                 horizontal_flip);
  }

  return canvas.AsNdArray();
}

void Scenario::LoadObjects(const json& objects_json) {
  int64_t cur_id = 0;
  for (const auto& obj : objects_json) {
    const ObjectType object_type = ParseObjectType(obj["type"]);

    // TODO(ev) current_time_ should be passed in rather than defined here.
    const geometry::Vector2D position(obj["position"][current_time_]["x"],
                                      obj["position"][current_time_]["y"]);
    const float width = static_cast<float>(obj["width"]);
    const float length = static_cast<float>(obj["length"]);
    geometry::Vector2D target_position;
    if (obj.contains("goalPosition")) {
      target_position = geometry::Vector2D(obj["goalPosition"]["x"],
                                           obj["goalPosition"]["y"]);
    }

    const auto& obj_position = obj["position"];
    const auto& obj_heading = obj["heading"];
    const auto& obj_velocity = obj["velocity"];
    const auto& obj_valid = obj["valid"];
    const int64_t trajectory_length = obj_position.size();
    const bool is_av = static_cast<bool>(obj["is_av"]);

    std::vector<geometry::Vector2D> cur_trajectory;
    std::vector<float> cur_headings;
    std::vector<float> cur_speeds;
    std::vector<bool> valid_mask;
    cur_trajectory.reserve(trajectory_length);
    cur_headings.reserve(trajectory_length);
    cur_speeds.reserve(trajectory_length);
    valid_mask.reserve(trajectory_length);

    float target_heading = 0.0f;
    float target_speed = 0.0f;
    bool is_moving = false;
    for (int64_t i = 0; i < trajectory_length; ++i) {
      const geometry::Vector2D cur_pos(obj_position[i]["x"],
                                       obj_position[i]["y"]);
      const float cur_heading = geometry::utils::NormalizeAngle(
          geometry::utils::Radians(static_cast<float>(obj_heading[i])));
      const float cur_speed =
          geometry::Vector2D(obj_velocity[i]["x"], obj_velocity[i]["y"]).Norm();
      const bool valid = static_cast<bool>(obj_valid[i]);

      cur_trajectory.push_back(cur_pos);
      cur_headings.push_back(cur_heading);
      cur_speeds.push_back(cur_speed);
      valid_mask.push_back(valid);

      if (valid) {
        // Use the last valid heading and speed as target heading and speed.
        // TODO: Improve this later.
        target_heading = cur_heading;
        target_speed = cur_speed;
        if (cur_speed > speed_threshold_ ||
            geometry::Distance(cur_pos, target_position) > moving_threshold_) {
          is_moving = true;
        }
      }
    }

    // we only want to store and load vehicles that are valid at this
    // initialization time, unless spawn_invalid_objects_ is set
    if (!valid_mask[current_time_] && !spawn_invalid_objects_) {
      continue;
    }

    if (object_type == ObjectType::kVehicle) {
      std::shared_ptr<Vehicle> vehicle = std::make_shared<Vehicle>(
          cur_id, length, width, position, cur_headings[current_time_],
          cur_speeds[current_time_], target_position, target_heading,
          target_speed, is_av);
      vehicles_.push_back(vehicle);
      objects_.push_back(vehicle);
      if (is_moving) {
        moving_objects_.push_back(vehicle);
      }
    } else if (allow_non_vehicles_) {
      if (object_type == ObjectType::kPedestrian) {
        std::shared_ptr<Pedestrian> pedestrian = std::make_shared<Pedestrian>(
            cur_id, length, width, position, cur_headings[current_time_],
            cur_speeds[current_time_], target_position, target_heading,
            target_speed);
        pedestrians_.push_back(pedestrian);
        objects_.push_back(pedestrian);
        if (is_moving) {
          moving_objects_.push_back(pedestrian);
        }
      } else if (object_type == ObjectType::kCyclist) {
        std::shared_ptr<Cyclist> cyclist = std::make_shared<Cyclist>(
            cur_id, length, width, position, cur_headings[current_time_],
            cur_speeds[current_time_], target_position, target_heading,
            target_speed);
        cyclists_.push_back(cyclist);
        objects_.push_back(cyclist);
        if (is_moving) {
          moving_objects_.push_back(cyclist);
        }
      } else {
        std::cerr << "Unknown object type: " << obj["type"] << std::endl;
      }
    }

    expert_trajectories_.push_back(std::move(cur_trajectory));
    expert_headings_.push_back(std::move(cur_headings));
    expert_speeds_.push_back(std::move(cur_speeds));
    expert_valid_masks_.push_back(std::move(valid_mask));
    ++cur_id;
  }

  // Reset the road objects bvh
  object_bvh_.Reset(objects_);
}

void Scenario::LoadRoads(const json& roads_json) {
  float min_x = std::numeric_limits<float>::max();
  float min_y = std::numeric_limits<float>::max();
  float max_x = std::numeric_limits<float>::lowest();
  float max_y = std::numeric_limits<float>::lowest();

  for (const auto& road : roads_json) {
    const auto& geometry_json = road["geometry"];
    const RoadType road_type = ParseRoadType(road["type"]);
    const bool check_collision = (road_type == RoadType::kRoadEdge);

    // We have to handle stop signs differently from other lane types
    if (road_type == RoadType::kStopSign) {
      const geometry::Vector2D position(geometry_json.front()["x"],
                                        geometry_json.front()["y"]);
      stop_signs_.push_back(std::make_shared<StopSign>(position));
    } else {
      const int64_t geometry_size = geometry_json.size();
      std::vector<geometry::Vector2D> geometry;
      geometry.reserve(geometry_size);

      // Iterate over every line segment
      for (int64_t i = 0; i < geometry_size; ++i) {
        const geometry::Vector2D cur_pos(geometry_json[i]["x"],
                                         geometry_json[i]["y"]);
        min_x = std::min(min_x, cur_pos.x());
        min_y = std::min(min_y, cur_pos.y());
        max_x = std::max(max_x, cur_pos.x());
        max_y = std::max(max_y, cur_pos.y());
        geometry.push_back(cur_pos);

        if (check_collision && i < geometry_size - 1) {
          const geometry::Vector2D nxt_pos(geometry_json[i + 1]["x"],
                                           geometry_json[i + 1]["y"]);
          line_segments_.push_back(
              std::make_shared<geometry::LineSegment>(cur_pos, nxt_pos));
        }
      }
      // TODO: Try different sample rate.
      std::shared_ptr<RoadLine> road_line = std::make_shared<RoadLine>(
          road_type, std::move(geometry), sample_every_n_, check_collision);
      road_lines_.push_back(road_line);
    }
  }

  road_network_bounds_ =
      sf::FloatRect(min_x, min_y, max_x - min_x, max_y - min_y);

  // Now create the BVH for the line segments
  // Since the line segments never move we only need to define this once
  line_segment_bvh_.Reset(line_segments_);
}

}  // namespace nocturne
