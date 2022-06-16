// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include "view_field.h"

#include <gmock/gmock-matchers.h>
#include <gtest/gtest.h>

#include <cmath>
#include <string>
#include <vector>

#include "geometry/geometry_utils.h"
#include "geometry/vector_2d.h"
#include "object_base.h"

namespace nocturne {
namespace {

using geometry::ConvexPolygon;
using geometry::Vector2D;
using geometry::utils::kHalfPi;
using geometry::utils::kPi;
using geometry::utils::kTwoPi;
using testing::ElementsAre;

class MockObject : public ObjectBase {
 public:
  MockObject() = default;

  MockObject(float length, float width, const Vector2D& position,
             bool can_block_sight)
      : ObjectBase(position, can_block_sight,
                   /*can_be_collided=*/true, /*check_collision=*/true),
        length_(length),
        width_(width) {}

  MockObject(float length, float width, const Vector2D& position, float heading,
             bool can_block_sight)
      : ObjectBase(position, can_block_sight,
                   /*can_be_collided=*/true, /*check_collision=*/true),
        length_(length),
        width_(width),
        heading_(heading) {}

  float heading() const { return heading_; }

  float Radius() const override {
    return std::sqrt(length_ * length_ + width_ * width_) * 0.5f;
  }

  ConvexPolygon BoundingPolygon() const override {
    const geometry::Vector2D p0 =
        geometry::Vector2D(length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
        position_;
    const geometry::Vector2D p1 =
        geometry::Vector2D(-length_ * 0.5f, width_ * 0.5f).Rotate(heading_) +
        position_;
    const geometry::Vector2D p2 =
        geometry::Vector2D(-length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
        position_;
    const geometry::Vector2D p3 =
        geometry::Vector2D(length_ * 0.5f, -width_ * 0.5f).Rotate(heading_) +
        position_;
    return ConvexPolygon({p0, p1, p2, p3});
  }

 protected:
  void draw(sf::RenderTarget& /*target*/,
            sf::RenderStates /*states*/) const override {}

  const float length_ = 0.0f;
  const float width_ = 0.0f;
  const float heading_ = 0.0f;
};

TEST(ViewFieldTest, VisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi,
                     geometry::utils::Radians(120.0f));

  const MockObject obj1(2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  auto ret = vf.VisibleObjects({&obj1, &obj2, &obj3, &obj4, &obj5});
  EXPECT_THAT(ret, ElementsAre(&obj1, &obj4, &obj5));

  const MockObject obj6(10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  ret = vf.VisibleObjects({&obj6});
  EXPECT_THAT(ret, ElementsAre(&obj6));
}

TEST(ViewFieldTest, FilterVisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi,
                     geometry::utils::Radians(120.0f));

  const MockObject obj1(2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  std::vector<const ObjectBase*> objects = {&obj1, &obj2, &obj3, &obj4, &obj5};
  vf.FilterVisibleObjects(objects);
  EXPECT_THAT(objects, ElementsAre(&obj1, &obj4, &obj5));

  const MockObject obj6(10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  objects = std::vector<const ObjectBase*>{&obj6};
  vf.FilterVisibleObjects(objects);
  EXPECT_THAT(objects, ElementsAre(&obj6));
}

TEST(ViewFieldTest, PanoramicViewVisibleObjectsTest) {
  const ViewField vf(Vector2D(1.0f, 1.0f), 10.0f, kHalfPi, kTwoPi);

  const MockObject obj1(2.0f, 1.0f, Vector2D(1.0f, 3.0f), true);
  const MockObject obj2(2.0f, 1.0f, Vector2D(1.0f, -1.0f), true);
  const MockObject obj3(2.0f, 1.0f, Vector2D(1.0f, 4.0f), true);
  const MockObject obj4(1.5f, 1.0f, Vector2D(1.0f, 2.0f), false);
  const MockObject obj5(2.0f, 1.0f, Vector2D(4.5f, 4.0f), true);
  auto ret = vf.VisibleObjects({&obj1, &obj2, &obj3, &obj4, &obj5});
  EXPECT_THAT(ret, ElementsAre(&obj1, &obj2, &obj4, &obj5));

  const MockObject obj6(10.0f, 1.0f, Vector2D(1.0f, 10.4f), true);
  ret = vf.VisibleObjects({&obj6});
  EXPECT_THAT(ret, ElementsAre(&obj6));
}

TEST(ViewFieldTest, NumericTest) {
  const std::vector<MockObject> objects = {
      MockObject(4.740863800048828, 2.0649945735931396,
                 Vector2D(2669.646484, -2188.851074),
                 /*heading=*/-2.0404860973358154,
                 /*can_block_sight=*/true),
      MockObject(4.4499664306640625, 2.0561788082122803,
                 Vector2D(2680.234863, -2184.793213),
                 /*heading=*/1.1528973579406738,
                 /*can_block_sight=*/true),
      MockObject(4.795155048370361, 2.0164151191711426,
                 Vector2D(2684.984131, -2204.270996),
                 /*heading=*/-0.4758952260017395,
                 /*can_block_sight=*/true),
      MockObject(4.927265167236328, 2.2062973976135254,
                 Vector2D(2680.350098, -2196.872803),
                 /*heading=*/2.579761028289795,
                 /*can_block_sight=*/true),
      MockObject(4.465244293212891, 2.112093687057495,
                 Vector2D(2675.792480, -2176.175049),
                 /*heading=*/-2.0115230083465576,
                 /*can_block_sight=*/true),
      MockObject(4.5914106369018555, 2.0008256435394287,
                 Vector2D(2666.556152, -2212.434570),
                 /*heading=*/1.1701767444610596,
                 /*can_block_sight=*/true),
      MockObject(8.120285987854004, 2.95621919631958,
                 Vector2D(2669.058105, -2197.881348),
                 /*heading=*/1.1092040538787842,
                 /*can_block_sight=*/true),
      MockObject(4.83572244644165, 2.193498373031616,
                 Vector2D(2695.201172, -2209.294189),
                 /*heading=*/-0.45424920320510864,
                 /*can_block_sight=*/true),
      MockObject(4.4054365158081055, 1.9970016479492188,
                 Vector2D(2685.656006, -2173.814941),
                 /*heading=*/1.087992787361145,
                 /*can_block_sight=*/true),
      MockObject(4.6685895919799805, 2.0569701194763184,
                 Vector2D(2679.008789, -2169.271240),
                 /*heading=*/-2.0146853923797607,
                 /*can_block_sight=*/true),
      MockObject(4.503179550170898, 2.0618410110473633,
                 Vector2D(2662.907715, -2219.484863),
                 /*heading=*/1.1390262842178345,
                 /*can_block_sight=*/true),
      MockObject(4.805270195007324, 2.0598297119140625,
                 Vector2D(2684.970215, -2157.089111),
                 /*heading=*/-2.0285773277282715,
                 /*can_block_sight=*/true),
      MockObject(4.365901470184326, 1.957434892654419,
                 Vector2D(2681.695557, -2163.795898),
                 /*heading=*/-1.9882886409759521,
                 /*can_block_sight=*/true),
      MockObject(4.495063781738281, 2.1045844554901123,
                 Vector2D(2649.875488, -2246.350586),
                 /*heading=*/1.0837059020996094,
                 /*can_block_sight=*/true),
      MockObject(4.618539810180664, 2.068289279937744,
                 Vector2D(2697.755615, -2149.247314),
                 /*heading=*/1.1281996965408325,
                 /*can_block_sight=*/true),
      MockObject(4.5271992683410645, 2.0338592529296875,
                 Vector2D(2694.705811, -2137.984131),
                 /*heading=*/-2.0358753204345703,
                 /*can_block_sight=*/true),
      MockObject(4.541479587554932, 2.026360273361206,
                 Vector2D(2700.818604, -2211.965576),
                 /*heading=*/-0.44259434938430786,
                 /*can_block_sight=*/true),
      MockObject(4.526176452636719, 2.0519537925720215,
                 Vector2D(2687.597168, -2152.175293),
                 /*heading=*/-2.0117440223693848,
                 /*can_block_sight=*/true),
      MockObject(4.353062152862549, 1.9422996044158936,
                 Vector2D(2715.791748, -2219.446533),
                 /*heading=*/-0.4740070104598999,
                 /*can_block_sight=*/true),
      MockObject(4.705846309661865, 2.112614154815674,
                 Vector2D(2700.483154, -2205.825439),
                 /*heading=*/2.683105707168579,
                 /*can_block_sight=*/true),
      MockObject(4.342007637023926, 1.9726860523223877,
                 Vector2D(2685.356445, -2140.802979),
                 /*heading=*/2.5876576900482178,
                 /*can_block_sight=*/true),
      MockObject(4.648506164550781, 2.091207981109619,
                 Vector2D(2695.039551, -2203.134033),
                 /*heading=*/2.6856439113616943,
                 /*can_block_sight=*/true),
      MockObject(4.55119514465332, 2.0358266830444336,
                 Vector2D(2640.011475, -2248.565674),
                 /*heading=*/-2.0497934818267822,
                 /*can_block_sight=*/true),
      MockObject(4.393487930297852, 1.9973748922348022,
                 Vector2D(2648.627686, -2235.096924),
                 /*heading=*/0.05755229666829109,
                 /*can_block_sight=*/true),
      MockObject(5.119008541107178, 2.211367607116699,
                 Vector2D(2654.697998, -2221.887207),
                 /*heading=*/-2.021545171737671,
                 /*can_block_sight=*/true),
      MockObject(4.545588970184326, 1.9355685710906982,
                 Vector2D(2638.943604, -2232.335938),
                 /*heading=*/-0.5254319906234741,
                 /*can_block_sight=*/true),
      MockObject(4.695493698120117, 2.0350232124328613,
                 Vector2D(2631.950684, -2228.144287),
                 /*heading=*/-0.4927181005477905,
                 /*can_block_sight=*/true),
      MockObject(5.285999774932861, 2.3320000171661377,
                 Vector2D(2673.315674, -2200.980957),
                 /*heading=*/0.6411342620849609,
                 /*can_block_sight=*/true)};

  const int64_t n = objects.size();
  constexpr int64_t kVehicleIndex = 21;
  const MockObject& ego = objects[kVehicleIndex];
  std::vector<const ObjectBase*> ptrs;
  ptrs.reserve(n - 1);
  for (int64_t i = 0; i < n; ++i) {
    if (i != kVehicleIndex) {
      ptrs.push_back(dynamic_cast<const ObjectBase*>(&objects[i]));
    }
  }

  constexpr float kViewDist1 = 80.0f;
  const ViewField vf1(ego.position(), kViewDist1,
                      geometry::utils::AngleAdd(ego.heading(), 0.3f),
                      geometry::utils::Radians(120.0f));
  const auto visible_objects1 = vf1.VisibleObjects(ptrs);
  EXPECT_EQ(visible_objects1.size(), 15);

  constexpr float kViewDist2 = 50.0f;
  const ViewField vf2(ego.position(), kViewDist2, ego.heading(),
                      geometry::utils::Radians(120.0f));
  const auto visible_objects2 = vf2.VisibleObjects(ptrs);
  EXPECT_EQ(visible_objects2.size(), 14);
}

}  // namespace
}  // namespace nocturne
