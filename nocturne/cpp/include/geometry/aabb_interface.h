// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include "geometry/aabb.h"

namespace nocturne {
namespace geometry {

class AABBInterface {
 public:
  virtual AABB GetAABB() const = 0;
};

}  // namespace geometry
}  // namespace nocturne
