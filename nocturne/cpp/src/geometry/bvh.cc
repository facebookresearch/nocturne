// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/bvh.h"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <limits>
#include <tuple>

#include "geometry/vector_2d.h"

namespace nocturne {
namespace geometry {

namespace {

// Binary search for smallest index who shares the same highest bit with r - 1
// in range [l, r).
int64_t FindPivot(
    const std::vector<std::pair<uint64_t, const AABBInterface*>>& objects,
    int64_t l, int64_t r) {
  const uint64_t last = objects[r - 1].first;
  const int64_t pivot_prefix = __builtin_clzll(objects[l].first ^ last);
  int64_t ret = r;
  while (l < r) {
    const int64_t mid = l + (r - l) / 2;
    const int64_t cur_prefix = __builtin_clzll(objects[mid].first ^ last);
    if (cur_prefix > pivot_prefix) {
      ret = mid;
      r = mid;
    } else {
      l = mid + 1;
    }
  }
  return ret;
}

std::pair<BVH::Node*, float> FindBestMatch(
    const std::tuple<BVH::Node*, BVH::Node*, float>* nodes, int64_t num,
    BVH::Node* p) {
  float dis = std::numeric_limits<float>::max();
  BVH::Node* ret = nullptr;
  for (int64_t i = 0; i < num; ++i) {
    BVH::Node* q = std::get<0>(nodes[i]);
    if (p != q) {
      const float cur_dis = Distance(p->aabb(), q->aabb());
      if (cur_dis < dis) {
        ret = q;
        dis = cur_dis;
      }
    }
  }
  return std::make_pair(ret, dis);
}

void RemoveNode(int64_t num, BVH::Node* p,
                std::tuple<BVH::Node*, BVH::Node*, float>* nodes) {
  const auto ip =
      std::find_if(nodes, nodes + num,
                   [p](const std::tuple<BVH::Node*, BVH::Node*, float>& cur) {
                     return std::get<0>(cur) == p;
                   });
  std::swap(*ip, nodes[num - 1]);
}

}  // namespace

std::vector<BVH::Node*> BVH::CombineNodes(const std::vector<BVH::Node*>& nodes,
                                          int64_t num) {
  const int64_t n = nodes.size();
  if (n <= num) {
    return nodes;
  }

  std::vector<std::tuple<BVH::Node*, BVH::Node*, float>> clusters;
  clusters.reserve(n);
  for (BVH::Node* p : nodes) {
    clusters.emplace_back(p, nullptr, 0.0f);
  }
  for (auto& c : clusters) {
    BVH::Node* p = std::get<0>(c);
    auto [q, d] = FindBestMatch(clusters.data(), n, p);
    std::get<1>(c) = q;
    std::get<2>(c) = d;
  }
  for (int64_t cur_size = clusters.size(); cur_size > num; --cur_size) {
    BVH::Node* u = nullptr;
    BVH::Node* v = nullptr;
    float best = std::numeric_limits<float>::max();
    for (int64_t i = 0; i < cur_size; ++i) {
      auto [p, q, d] = clusters[i];
      if (d < best) {
        best = d;
        u = p;
        v = q;
      }
    }
    BVH::Node* x = MakeNode(u, v);
    RemoveNode(cur_size, u, clusters.data());
    RemoveNode(cur_size - 1, v, clusters.data());
    const auto [y, d] = FindBestMatch(clusters.data(), cur_size - 2, x);
    clusters[cur_size - 2] = std::make_tuple(x, y, d);
    for (int64_t i = 0; i < cur_size - 2; ++i) {
      const auto [p, q, d] = clusters[i];
      if (q == u || q == v) {
        auto [qq, dd] = FindBestMatch(clusters.data(), cur_size - 1, p);
        std::get<1>(clusters[i]) = qq;
        std::get<2>(clusters[i]) = dd;
      }
    }
  }

  std::vector<BVH::Node*> ret;
  ret.reserve(num);
  for (int64_t i = 0; i < num; ++i) {
    ret.push_back(std::get<0>(clusters[i]));
  }
  return ret;
}

std::vector<BVH::Node*> BVH::InitHierarchy(
    const std::vector<std::pair<uint64_t, const AABBInterface*>>& objects,
    int64_t l, int64_t r) {
  assert(l < r);
  if (r - l <= delta_) {
    std::vector<BVH::Node*> ret;
    ret.reserve(r - l);
    for (int64_t i = l; i < r; ++i) {
      ret.push_back(MakeNode(std::get<1>(objects[i])));
    }
    return ret;
  }
  int64_t p = FindPivot(objects, l, r);
  if (p == l || p == r) {
    p = l + (r - l) / 2;
  }

  const std::vector<Node*> l_nodes = InitHierarchy(objects, l, p);
  const std::vector<Node*> r_nodes = InitHierarchy(objects, p, r);
  std::vector<Node*> nodes;
  nodes.reserve(l_nodes.size() + r_nodes.size());
  for (Node* node : l_nodes) {
    nodes.push_back(node);
  }
  for (Node* node : r_nodes) {
    nodes.push_back(node);
  }
  const int64_t num = (nodes.size() + 1) / 2;
  return CombineNodes(nodes, num);
}

}  // namespace geometry
}  // namespace nocturne
