// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace nocturne {
namespace geometry {

class PointLike;
class Polygon;
class Vector2D;

namespace utils {

// Use uint32_t instead of bool for MaskType to avoid API limits of
// std::vector<bool> and also reach better performance.
using MaskType = uint32_t;

constexpr double kEps = 1e-8;
constexpr double kPi = M_PI;
constexpr double kTwoPi = 2.0 * kPi;
constexpr double kHalfPi = M_PI_2;
constexpr double kQuarterPi = M_PI_4;

template <typename T>
inline bool AlmostEquals(T lhs, T rhs) {
  return std::fabs(lhs - rhs) < std::numeric_limits<T>::epsilon() * T(32);
}

template <typename T>
constexpr T Radians(T d) {
  return d / 180.0 * kPi;
}

template <typename T>
constexpr T Degrees(T r) {
  return r / kPi * 180.0;
}

// Check if angle is in the range of [-Pi, Pi].
template <typename T>
constexpr bool IsNormalizedAngle(T angle) {
  return angle >= -kPi && angle <= kPi;
}

template <typename T>
inline float NormalizeAngle(T angle) {
  const T ret = std::fmod(angle, kTwoPi);
  return ret > kPi ? ret - kTwoPi : (ret < -kPi ? ret + kTwoPi : ret);
}

template <typename T>
inline T AngleAdd(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs + rhs);
}

template <typename T>
inline T AngleSub(T lhs, T rhs) {
  return NormalizeAngle<T>(lhs - rhs);
}

// Pack the coordinates of points into x and y.
std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<Vector2D>& points);
std::pair<std::vector<float>, std::vector<float>> PackCoordinates(
    const std::vector<const PointLike*>& points);

template <int64_t N>
std::pair<std::array<float, N>, std::array<float, N>> PackSmallPolygon(
    const Polygon& polygon);

}  // namespace utils
}  // namespace geometry
}  // namespace nocturne
