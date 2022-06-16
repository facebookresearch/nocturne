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

add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp
  ${CMAKE_CURRENT_BINARY_DIR}/cpp
)

# pybind11
add_subdirectory(
  ${CMAKE_CURRENT_SOURCE_DIR}/../third_party/pybind11
  ${CMAKE_CURRENT_BINARY_DIR}/third_party/pybind11
)

pybind11_add_module(
  nocturne_cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/action.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/nocturne.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/object.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/road.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/scenario.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/simulation.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/vector_2d.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/src/vehicle.cc
)
target_include_directories(
  nocturne_cpp
  PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/cpp/include
  ${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include
)
target_link_libraries(nocturne_cpp PUBLIC nocturne_core)
