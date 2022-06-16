// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <array>
#include <memory>
#include <utility>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/aabb_interface.h"
#include "geometry/intersection.h"
#include "geometry/line_segment.h"
#include "geometry/morton.h"

namespace nocturne {
namespace geometry {

// Bounding Volume Hierarchy
// https://en.wikipedia.org/wiki/Bounding_volume_hierarchy
class BVH {
 public:
  class Node {
   public:
    Node() = default;
    explicit Node(const AABBInterface* object)
        : aabb_(object->GetAABB()), object_(object) {}
    Node(const AABB& aabb, const AABBInterface* object, Node* l_child,
         Node* r_child)
        : aabb_(aabb), object_(object) {
      children_[0] = l_child;
      children_[1] = r_child;
    }

    const AABB& aabb() const { return aabb_; }
    const AABBInterface* object() const { return object_; }

    bool IsLeaf() const { return object_ != nullptr; }

    const Node* Child(int64_t index) const { return children_.at(index); }
    Node* Child(int64_t index) { return children_.at(index); }

    const Node* LChild() const { return children_[0]; }
    Node* LChild() { return children_[0]; }

    const Node* RChild() const { return children_[1]; }
    Node* RChild() { return children_[1]; }

   protected:
    AABB aabb_;
    const AABBInterface* object_ = nullptr;
    std::array<Node*, 2> children_ = {nullptr, nullptr};
  };

  BVH() = default;
  explicit BVH(int64_t delta) : delta_(delta) {}

  template <class ObjectType>
  explicit BVH(const std::vector<ObjectType>& objects) {
    Reset(objects);
  }

  template <class ObjectType>
  BVH(const std::vector<ObjectType>& objects, int64_t delta) : delta_(delta) {
    Reset(objects);
  }

  template <class ObjectType>
  explicit BVH(const std::vector<const ObjectType*>& objects) {
    Reset(objects);
  }

  template <class ObjectType>
  BVH(const std::vector<const ObjectType*>& objects, int64_t delta)
      : delta_(delta) {
    Reset(objects);
  }

  bool Empty() const { return nodes_.empty(); }
  int64_t Size() const { return nodes_.size(); }

  void Clear() {
    root_ = nullptr;
    nodes_.clear();
  }

  template <class ObjectType>
  void Reset(const std::vector<ObjectType>& objects) {
    ResetImpl(
        objects, [](const ObjectType& obj) { return obj.GetAABB(); },
        [](const ObjectType& obj) { return &obj; });
  }

  template <class ObjectType>
  void Reset(const std::vector<const ObjectType*>& objects) {
    ResetImpl(
        objects, [](const ObjectType* obj) { return obj->GetAABB(); },
        [](const ObjectType* obj) { return obj; });
  }

  template <class ObjectType>
  void Reset(const std::vector<std::shared_ptr<ObjectType>>& objects) {
    ResetImpl(
        objects,
        [](const std::shared_ptr<ObjectType>& obj) { return obj->GetAABB(); },
        [](const std::shared_ptr<ObjectType>& obj) { return obj.get(); });
  }

  template <class ObjectType>
  std::vector<const ObjectType*> IntersectionCandidates(
      const AABBInterface& object) const {
    std::vector<const ObjectType*> candidates;
    if (root_ != nullptr) {
      IntersectionCandidatesImpl<AABB, ObjectType>(object.GetAABB(), root_,
                                                   candidates);
    }
    return candidates;
  }

  template <class ObjectType>
  std::vector<const ObjectType*> IntersectionCandidates(
      const LineSegment& segment) const {
    std::vector<const ObjectType*> candidates;
    if (root_ != nullptr) {
      IntersectionCandidatesImpl<LineSegment, ObjectType>(segment, root_,
                                                          candidates);
    }
    return candidates;
  }

 protected:
  Node* MakeNode(const AABBInterface* object) {
    nodes_.emplace_back(object);
    return &nodes_.back();
  }

  Node* MakeNode(Node* l_child, Node* r_child) {
    nodes_.emplace_back((l_child->aabb() || r_child->aabb()),
                        /*object=*/nullptr, l_child, r_child);
    return &nodes_.back();
  }

  std::vector<BVH::Node*> CombineNodes(const std::vector<BVH::Node*>& nodes,
                                       int64_t num);

  // Implement AAC algorithm in
  // http://graphics.cs.cmu.edu/projects/aac/aac_build.pdf
  template <class ObjectType, class AABBFunc, class PtrFunc>
  void ResetImpl(const std::vector<ObjectType>& objects, AABBFunc aabb_func,
                 PtrFunc ptr_func) {
    Clear();
    if (objects.empty()) {
      return;
    }

    const int64_t n = objects.size();
    nodes_.reserve(2 * n - 1);
    std::vector<std::pair<uint64_t, const AABBInterface*>> encoded_objects;
    encoded_objects.reserve(n);
    for (const ObjectType& obj : objects) {
      const AABB aabb = aabb_func(obj);
      const uint64_t morton_code = morton::Morton2D(aabb.Center());
      encoded_objects.emplace_back(
          morton_code, dynamic_cast<const AABBInterface*>(ptr_func(obj)));
    }
    std::sort(encoded_objects.begin(), encoded_objects.end());
    const std::vector<Node*> nodes =
        InitHierarchy(encoded_objects, /*l=*/0, /*r=*/n);
    const std::vector<Node*> root = CombineNodes(nodes, 1);
    root_ = root[0];
  }

  // Init hierarchy in range [l, r).
  std::vector<Node*> InitHierarchy(
      const std::vector<std::pair<uint64_t, const AABBInterface*>>& objects,
      int64_t l, int64_t r);

  template <class AABBType, class ObjectType>
  void IntersectionCandidatesImpl(
      const AABBType& obj, const Node* cur,
      std::vector<const ObjectType*>& candidates) const {
    if (!Intersects(obj, cur->aabb())) {
      return;
    }
    if (cur->IsLeaf()) {
      candidates.push_back(dynamic_cast<const ObjectType*>(cur->object()));
      return;
    }
    IntersectionCandidatesImpl(obj, cur->LChild(), candidates);
    IntersectionCandidatesImpl(obj, cur->RChild(), candidates);
  }

  std::vector<Node> nodes_;
  Node* root_ = nullptr;
  const int64_t delta_ = 4;
};

}  // namespace geometry
}  // namespace nocturne
