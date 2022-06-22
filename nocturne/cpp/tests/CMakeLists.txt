cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

project(nocturne)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

add_executable(
  nocturne_test 
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/bvh_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/circular_sector_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/intersection_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/line_segment_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/polygon_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/range_tree_2d_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/object_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/road_test.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/view_field_test.cc
)
target_include_directories(
  nocturne_test
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../include
)
target_link_libraries(
  nocturne_test
  PUBLIC
  gmock
  gtest
  gtest_main
  nocturne_core
)

include(GoogleTest)
gtest_discover_tests(nocturne_test)
