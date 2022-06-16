cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(
  CMAKE_CXX_FLAGS
  "${CMAKE_CXX_FLAGS} -O3 -Wall -Wextra -Wno-register -Wno-comment -fPIC \
  -march=native -Wfatal-errors -fvisibility=hidden"
)

project(nocturne)

add_library(
  nocturne_geometry
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/bvh.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/circle.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/circular_sector.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/geometry_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/intersection.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/line_segment.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/morton.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/geometry/polygon.cc
)
target_include_directories(
  nocturne_geometry 
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Build and link nocturne
# add_library(nocturne SHARED ${SOURCE_FILES})
add_library(
  nocturne_core
  STATIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/object.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/road.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/scenario.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/simulation.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/stop_sign.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/traffic_light.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/utils/sf_utils.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/src/view_field.cc
)
# set_property(TARGET nocturne_lib PROPERTY POSITION_INDEPENDENT_CODE 1)
target_include_directories(
  nocturne_core
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/json/include
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link SFML
find_package(SFML 2.5 COMPONENTS window system graphics audio REQUIRED)
set(SFML_LIBS sfml-window sfml-system sfml-graphics sfml-audio)

target_link_libraries(nocturne_core PUBLIC ${SFML_LIBS} nocturne_geometry)

# Google Test
include(FetchContent)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

enable_testing()
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/tests
  ${CMAKE_CURRENT_BINARY_DIR}/tests
)
