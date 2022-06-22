// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <SFML/Graphics.hpp>

#include "geometry/geometry_utils.h"
#include "geometry/line_segment.h"
#include "geometry/vector_2d.h"

namespace nocturne {
namespace utils {

// Converts a `geometry::Vector2D` to a `sf::Vector2f`. If `flip_y` is true,
// then the y coordinate is flipped (y becomes -y).
inline sf::Vector2f ToVector2f(const geometry::Vector2D& vec,
                               bool flip_y = false) {
  return sf::Vector2f(vec.x(), flip_y ? -vec.y() : vec.y());
}

// Loads a font file `font_name` from the system (eg Arial.ttf).
// Font files are currently searched in standard Linux, macOS and Windows paths.
sf::Font LoadFont(const std::string& font_name);

// Creates and returns a pointer to an `sf::CircleShape` object. The circle is
// centered at `position`, has radius `radius` and color `color`.
std::unique_ptr<sf::CircleShape> MakeCircleShape(geometry::Vector2D position,
                                                 float radius, sf::Color color,
                                                 bool filled = true);

// Creates and returns drawables for a cone shape. The cone has its center
// pointing upwards (+pi/2) with a tilt `tilt` (in radians), has an angle
// `angle` (in radians) and a radius `radius`. This function fills in
// `fill_color` everything that is not part of the cone, within the square
// containing a circle of radius `radius` centered in (0, 0). Each quarter of
// circle is approximated with `n_points` points.
std::vector<std::unique_ptr<sf::VertexArray>> MakeInvertedConeShape(
    float radius, float angle, float tilt = 0.0f,
    sf::Color fill_color = sf::Color::Black, int64_t n_points = 20);

// Creates and returns obstruction drawables. Given a source position
// `source_pos` and a convex polygon object represented by its sides
// `obj_lines`, fills in `fill_color` all the area that is hidden by the object
// from the view of the source (up to a distance `radius`). Circular arcs are
// approximated by `n_points`.
std::vector<std::unique_ptr<sf::ConvexShape>> MakeObstructionShape(
    geometry::Vector2D source_pos, std::vector<geometry::LineSegment> obj_lines,
    float radius, sf::Color fill_color = sf::Color::Black,
    int64_t n_points = 80);

}  // namespace utils
}  // namespace nocturne
