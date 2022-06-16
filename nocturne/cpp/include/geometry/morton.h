// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <cstdint>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {
namespace morton {

uint64_t Morton2D(const Vector2D& v);

}  // namespace morton
}  // namespace geometry
}  // namespace nocturne
