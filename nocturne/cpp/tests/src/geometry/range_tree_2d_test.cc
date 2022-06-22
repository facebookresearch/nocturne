// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "geometry/range_tree_2d.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <random>
#include <vector>

#include "geometry/aabb.h"
#include "geometry/point_like.h"

namespace nocturne {
namespace geometry {
namespace {

class MockPoint : public PointLike {
 public:
  MockPoint(const Vector2D& point) : point_(point) {}

  Vector2D Coordinate() const override { return point_; }

 protected:
  Vector2D point_;
};

std::vector<MockPoint> MakeRandomPoinst(int64_t n, float l, float r) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> dis(l, r);
  std::vector<MockPoint> points;
  for (int64_t i = 0; i < n; ++i) {
    const float x = dis(gen);
    const float y = dis(gen);
    points.emplace_back(Vector2D(x, y));
  }
  return points;
}

std::vector<const MockPoint*> RangeSearch(const std::vector<MockPoint>& points,
                                          const AABB& aabb) {
  std::vector<const MockPoint*> ret;
  for (const auto& p : points) {
    if (aabb.Contains(p.Coordinate())) {
      ret.push_back(&p);
    }
  }
  std::sort(ret.begin(), ret.end(), [](const MockPoint* a, const MockPoint* b) {
    return a->Coordinate() < b->Coordinate();
  });
  return ret;
}

TEST(RangeTree2dTest, RangeSearchTest) {
  const int64_t n = 1000;
  const int64_t cap = 1024;
  const float l = -20.0;
  const float r = 20.0;
  std::vector<MockPoint> points = MakeRandomPoinst(n, l, r);

  RangeTree2d tree(points);
  ASSERT_EQ(tree.size(), n);
  ASSERT_EQ(tree.capacity(), cap);

  const AABB aabb1(-10.0, -10.0, 10.0, 10.0);
  std::vector<const MockPoint*> ret = tree.RangeSearch<MockPoint>(aabb1);
  std::sort(ret.begin(), ret.end(), [](const MockPoint* a, const MockPoint* b) {
    return a->Coordinate() < b->Coordinate();
  });
  std::vector<const MockPoint*> ans = RangeSearch(points, aabb1);
  ASSERT_EQ(ret.size(), ans.size());
  for (size_t i = 0; i < ret.size(); ++i) {
    EXPECT_EQ(ret[i], ans[i]);
  }
}

}  // namespace
}  // namespace geometry
}  // namespace nocturne
