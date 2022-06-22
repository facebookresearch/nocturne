// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <memory>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/circle.h"
#include "geometry/circular_sector.h"
#include "geometry/point_like.h"
#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {

class ViewField : public geometry::AABBInterface {
 public:
  ViewField() = default;
  ViewField(const geometry::Vector2D& center, float radius, float heading,
            float theta);

  geometry::AABB GetAABB() const override { return vision_->GetAABB(); }

  std::vector<const ObjectBase*> VisibleObjects(
      const std::vector<const ObjectBase*>& objects) const;
  void FilterVisibleObjects(std::vector<const ObjectBase*>& objects) const;

  std::vector<const ObjectBase*> VisibleNonblockingObjects(
      const std::vector<const ObjectBase*>& objects) const;
  void FilterVisibleNonblockingObjects(
      std::vector<const ObjectBase*>& objects) const;

  std::vector<const geometry::PointLike*> VisiblePoints(
      const std::vector<const geometry::PointLike*>& objects) const;
  void FilterVisiblePoints(
      std::vector<const geometry::PointLike*>& objects) const;

 protected:
  std::vector<geometry::Vector2D> ComputeSightEndpoints(
      const std::vector<const ObjectBase*>& objects) const;

  std::unique_ptr<geometry::CircleLike> vision_ = nullptr;
  const bool panoramic_view_ = false;
};

}  // namespace nocturne
