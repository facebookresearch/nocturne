// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#include "utils/sf_utils.h"

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace nocturne {
namespace utils {

namespace {

#if defined(__APPLE__)  // OSX

std::vector<std::string> GetFontPaths() {
  return {"/System/Library/Fonts/Supplemental/", "~/Library/Fonts/"};
}

#elif defined(_WIN32)  // Windows 32 bit or 64 bit

std::vector<std::string> GetFontPaths() { return {"C:/Windows/Fonts/"}; }

#else  // Linux

std::vector<std::string> GetFontPaths() {
  const std::string username = std::getenv("USER");
  return {"/usr/share/fonts", "/usr/local/share/fonts",
          "/home/" + username + "/.fonts/",
          "/private/home/" + username + "/.fonts/"};
}

#endif

bool FindFontPath(const std::string& font_name, std::string& font_path) {
  const std::vector<std::string> font_paths = GetFontPaths();
  for (const std::string& fp : font_paths) {
    const std::string cur_path = fp + font_name;
    std::ifstream font(cur_path);
    if (font.is_open()) {
      font_path = cur_path;
      return true;
    }
  }
  return false;
}

}  // namespace

sf::Font LoadFont(const std::string& font_name) {
  std::string font_path;
  sf::Font font;
  if (!FindFontPath(font_name, font_path) || !font.loadFromFile(font_path)) {
    throw std::invalid_argument("Could not load font file " + font_name + ".");
  }
  return font;
}

std::unique_ptr<sf::CircleShape> MakeCircleShape(geometry::Vector2D position,
                                                 float radius, sf::Color color,
                                                 bool filled) {
  auto circle_shape = std::make_unique<sf::CircleShape>(radius);
  circle_shape->setOrigin(radius, radius);
  if (filled) {
    circle_shape->setFillColor(color);
  } else {
    circle_shape->setFillColor(sf::Color::Transparent);
    circle_shape->setOutlineColor(color);
    circle_shape->setOutlineThickness(0.5);
  }
  circle_shape->setPosition(utils::ToVector2f(position));
  return circle_shape;
}

std::vector<std::unique_ptr<sf::VertexArray>> MakeInvertedConeShape(
    float radius, float angle, float tilt, sf::Color fill_color,
    int64_t n_points) {
  std::vector<std::unique_ptr<sf::VertexArray>> drawables;

  // fill around the circle (ie. enclosing square minus circle)
  for (int64_t quadrant = 0; quadrant < 4; ++quadrant) {
    auto quadrant_drawable = std::make_unique<sf::VertexArray>(sf::TriangleFan);
    const float quadrant_start_angle = quadrant * geometry::utils::kHalfPi;

    // corner point of the square
    geometry::Vector2D corner = geometry::PolarToVector2D(
        std::sqrt(2 * radius * radius),
        quadrant_start_angle + geometry::utils::kQuarterPi);
    quadrant_drawable->append(
        sf::Vertex(utils::ToVector2f(corner), fill_color));

    // quarter of circle approximation
    for (int i = 0; i < n_points; ++i) {
      const float point_angle = quadrant_start_angle +
                                i * (geometry::utils::kHalfPi) / (n_points - 1);
      geometry::Vector2D pt = geometry::PolarToVector2D(radius, point_angle);
      quadrant_drawable->append(sf::Vertex(utils::ToVector2f(pt), fill_color));
    }

    drawables.push_back(std::move(quadrant_drawable));
  }

  // fill around cone (ie. part within circle that is not part of the cone)
  if (angle < geometry::utils::kTwoPi) {
    auto cone_drawable = std::make_unique<sf::VertexArray>(sf::TriangleFan);

    // origin of the cone (center of the circle)
    cone_drawable->append(sf::Vertex(sf::Vector2f(0.0f, 0.0f), fill_color));

    // circular arc
    const float start_angle = geometry::utils::kHalfPi + angle / 2.0f + tilt;
    const float end_angle = geometry::utils::kHalfPi + geometry::utils::kTwoPi -
                            angle / 2.0f + tilt;
    const int n_points_circle = 4 * n_points;
    for (int i = 0; i < n_points_circle; ++i) {
      const float point_angle =
          start_angle + i * (end_angle - start_angle) / (n_points_circle - 1);
      geometry::Vector2D pt = geometry::PolarToVector2D(radius, point_angle);
      cone_drawable->append(sf::Vertex(utils::ToVector2f(pt), fill_color));
    }

    drawables.push_back(std::move(cone_drawable));
  }

  return drawables;
}

std::vector<std::unique_ptr<sf::ConvexShape>> MakeObstructionShape(
    geometry::Vector2D source_pos, std::vector<geometry::LineSegment> obj_lines,
    float radius, sf::Color fill_color, int64_t n_points) {
  std::vector<std::unique_ptr<sf::ConvexShape>> obscurity_drawables;

  // check each pair of lines, and if one line is behind the other from the
  // point of view of the source, then fill all the area behind that line as it
  // is obscured by the other line.
  for (const auto& line1 : obj_lines) {
    const geometry::Vector2D& pt1 = line1.Endpoint0();
    const geometry::Vector2D& pt2 = line1.Endpoint1();
    int nIntersections = 0;
    for (const auto& line2 : obj_lines) {
      const geometry::Vector2D& pt3 = line2.Endpoint0();
      const geometry::Vector2D& pt4 = line2.Endpoint1();
      if (pt1 != pt3 && pt1 != pt4 &&
          geometry::LineSegment(pt1, source_pos).Intersects(line2)) {
        nIntersections++;
        break;
      }
    }
    for (const auto& line2 : obj_lines) {
      const geometry::Vector2D& pt3 = line2.Endpoint0();
      const geometry::Vector2D& pt4 = line2.Endpoint1();
      if (pt2 != pt3 && pt2 != pt4 &&
          geometry::LineSegment(pt2, source_pos).Intersects(line2)) {
        nIntersections++;
        break;
      }
    }

    if (nIntersections >= 1) {
      // line1 is behind another line of the object -> fill behind it
      auto hiddenArea = std::make_unique<sf::ConvexShape>();

      float angle1 = (pt1 - source_pos).Angle();
      float angle2 = (pt2 - source_pos).Angle();

      // we want to go from angle1 to angle2
      float dAngle = geometry::utils::AngleSub(angle2, angle1) / (n_points - 1);

      hiddenArea->setPointCount(n_points + 2);
      hiddenArea->setFillColor(fill_color);
      // first point of line1
      hiddenArea->setPoint(0, utils::ToVector2f((pt1 - source_pos)));
      // circular arc
      for (int i = 0; i < n_points; ++i) {
        float angle = angle1 + i * dAngle;
        geometry::Vector2D pt = geometry::PolarToVector2D(radius, angle);
        hiddenArea->setPoint(1 + i, utils::ToVector2f(pt));
      }
      // second point of line1
      hiddenArea->setPoint(n_points + 1, utils::ToVector2f((pt2 - source_pos)));

      obscurity_drawables.push_back(std::move(hiddenArea));
    }
  }
  return obscurity_drawables;
}

}  // namespace utils
}  // namespace nocturne
