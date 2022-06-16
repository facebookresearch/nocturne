// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "geometry/morton.h"

#include <cstring>

namespace nocturne {
namespace geometry {
namespace morton {

namespace {

// https://stackoverflow.com/questions/26856268/morton-index-from-2d-point-with-floats

uint32_t FloatToInt(float x) {
  // const uint32_t ix = *(const uint32_t*)&x;
  // const uint32_t ix = reinterpret_cast<const uint32_t&>(x);
  uint32_t ix;
  std::memcpy(&ix, &x, sizeof(x));
  const int32_t ixs = static_cast<int32_t>(ix) >> 31;
  return (((ix & 0x7FFFFFFFL) ^ ixs) - ixs) + 0x7FFFFFFFL;
}

constexpr uint64_t ExpandBits(uint64_t x) {
  x = (x | (x << 16)) & 0x0000ffff0000ffffLL;
  x = (x | (x << 8)) & 0x00ff00ff00ff00ffLL;
  x = (x | (x << 4)) & 0x0f0f0f0f0f0f0f0fLL;
  x = (x | (x << 2)) & 0x3333333333333333LL;
  x = (x | (x << 1)) & 0x5555555555555555LL;
  return x;
}

}  // namespace

uint64_t Morton2D(const Vector2D& v) {
  const uint32_t ix = FloatToInt(v.x());
  const uint32_t iy = FloatToInt(v.y());
  uint64_t xx = static_cast<uint64_t>(ix);
  uint64_t yy = static_cast<uint64_t>(iy);
  xx = ExpandBits(xx);
  yy = ExpandBits(yy);
  return (xx << 1) | yy;
}

}  // namespace morton
}  // namespace geometry
}  // namespace nocturne
