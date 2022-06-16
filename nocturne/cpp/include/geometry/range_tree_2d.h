// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <algorithm>
#include <memory>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/point_like.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

// 2d Range Tree for efficient range search operations.
// https://en.wikipedia.org/wiki/Range_tree
//
// The Outer Tree (X-Tree) is implemented by a non-recursive segment tree.
// The Inner Tree (Y-Tree) is an std::vector sorted by y-coordinate where we
// can do binary search for lower_bound an upper_bound.
//
// The non-recursive segment tree implementaion is adapted from the link blow.
// https://codeforces.com/blog/entry/18051
//
// Time complexity for Reset operation: O(NlogN).
// Time complexity for RangeSearch operation: O(log^2(N) + K).
// Space complexity: O(NlogN).

class RangeTree2d {
 public:
  RangeTree2d() = default;

  template <class PointType>
  RangeTree2d(const std::vector<PointType>& points) {
    Reset(points);
  }

  template <class PointType>
  RangeTree2d(const std::vector<const PointType*>& points) {
    Reset(points);
  }

  template <class PointType>
  RangeTree2d(const std::vector<std::shared_ptr<PointType>>& points) {
    Reset(points);
  }

  int64_t size() const { return size_; }
  int64_t capacity() const { return capacity_; }

  bool Empty() const { return size_ == 0; }

  void Clear() {
    points_.clear();
    nodes_.clear();
    size_ = 0;
    capacity_ = 0;
  }

  template <class PointType>
  void Reset(const std::vector<PointType>& points) {
    ResetImpl(points, [](const PointType& p) { return &p; });
  }

  template <class PointType>
  void Reset(const std::vector<const PointType*>& points) {
    ResetImpl(points, [](const PointType* p) { return p; });
  }

  template <class PointType>
  void Reset(const std::vector<std::shared_ptr<PointType>>& points) {
    ResetImpl(points,
              [](const std::shared_ptr<PointType>& p) { return p.get(); });
  }

  // Time complexity: O(Log^2(N) + K)
  template <class PointType>
  std::vector<const PointType*> RangeSearch(const AABB& aabb) const {
    std::vector<const PointType*> ret;
    const auto l_ptr = std::lower_bound(
        points_.cbegin(), points_.cend(), aabb.MinX(),
        [](const PointLike* a, float b) { return a->X() < b; });
    const auto r_ptr = std::upper_bound(
        points_.cbegin(), points_.cend(), aabb.MaxX(),
        [](float a, const PointLike* b) { return a < b->X(); });
    int64_t l = (std::distance(points_.cbegin(), l_ptr) | capacity_);
    int64_t r = (std::distance(points_.cbegin(), r_ptr) | capacity_);
    while (l < r) {
      if (l & 1) {
        NodeRangeSearch<PointType>(aabb, nodes_[l++], ret);
      }
      if (r & 1) {
        NodeRangeSearch<PointType>(aabb, nodes_[--r], ret);
      }
      l >>= 1;
      r >>= 1;
    }
    return ret;
  }

  template <class PointType>
  std::vector<const PointType*> RangeSearch(const AABBInterface& object) const {
    return RangeSearch<PointType>(object.GetAABB());
  }

 protected:
  // Time complexity: O(NlogN)
  template <class PointType, class PtrFunc>
  void ResetImpl(const std::vector<PointType>& points, PtrFunc ptr_func) {
    Clear();
    size_ = points.size();
    points_.reserve(size_);
    for (const auto& point : points) {
      points_.push_back(dynamic_cast<const PointLike*>(ptr_func(point)));
    }
    // Sort points_ by x-coordinate.
    std::sort(
        points_.begin(), points_.end(),
        [](const PointLike* a, const PointLike* b) { return a->X() < b->X(); });
    for (capacity_ = 1; capacity_ < size_; capacity_ <<= 1)
      ;
    nodes_.assign(2 * capacity_, std::vector<const PointLike*>());
    for (int64_t i = 0; i < size_; ++i) {
      nodes_[i | capacity_].push_back(points_[i]);
    }
    for (int64_t i = capacity_ - 1; i > 0; --i) {
      // Combine nodes by y-coordinate.
      CombineNodes(nodes_[(i << 1)], nodes_[(i << 1) | 1], nodes_[i]);
    }
  }

  void CombineNodes(const std::vector<const PointLike*>& lhs,
                    const std::vector<const PointLike*>& rhs,
                    std::vector<const PointLike*>& ret) const {
    ret.assign(lhs.size() + rhs.size(), nullptr);
    std::merge(
        lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend(), ret.begin(),
        [](const PointLike* a, const PointLike* b) { return a->Y() < b->Y(); });
  }

  template <class PointType>
  void NodeRangeSearch(const AABB& aabb,
                       const std::vector<const PointLike*>& node,
                       std::vector<const PointType*>& ret) const {
    const auto l = std::lower_bound(
        node.cbegin(), node.cend(), aabb.MinY(),
        [](const PointLike* a, float b) { return a->Y() < b; });
    const auto r = std::upper_bound(
        node.cbegin(), node.cend(), aabb.MaxY(),
        [](float a, const PointLike* b) { return a < b->Y(); });
    for (auto it = l; it != r; ++it) {
      ret.push_back(dynamic_cast<const PointType*>(*it));
    }
  }

  int64_t size_ = 0;
  int64_t capacity_ = 0;
  // points_ is sorted by x-coordinate.
  std::vector<const PointLike*> points_;
  // Each element in nodes_ is sorted by y-coordinate.
  std::vector<std::vector<const PointLike*>> nodes_;
};

}  // namespace geometry
}  // namespace nocturne
