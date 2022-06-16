// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstring>
#include <memory>
#include <unordered_map>
#include <vector>

#include "ndarray.h"

namespace py = pybind11;

namespace nocturne {
namespace utils {

template <typename T>
py::array_t<T> AsNumpyArray(const std::vector<T>& vec) {
  py::array_t<T> arr(vec.size());
  std::memcpy(arr.mutable_data(), vec.data(), vec.size() * sizeof(T));
  return arr;
}

// Move a std::vector to numpy array without copy.
// https://github.com/ssciwr/pybind11-numpy-example/blob/main/python/pybind11-numpy-example_python.cpp
template <typename T>
py::array_t<T> AsNumpyArray(std::vector<T>&& vec) {
  const int64_t size = vec.size();
  const T* data = vec.data();
  std::unique_ptr<std::vector<T>> vec_ptr =
      std::make_unique<std::vector<T>>(std::move(vec));
  auto capsule = py::capsule(vec_ptr.get(), [](void* p) {
    std::unique_ptr<std::vector<T>>(reinterpret_cast<std::vector<T>*>(p));
  });
  vec_ptr.release();
  return py::array(size, data, capsule);
}

template <typename T>
py::array_t<T> AsNumpyArray(const NdArray<T>& arr) {
  py::array_t<T> ret = AsNumpyArray<T>(arr.data());
  ret.resize(arr.shape());
  return ret;
}

// Move a NdArray to numpy array without copy.
template <typename T>
py::array_t<T> AsNumpyArray(NdArray<T>&& arr) {
  py::array_t<T> ret = AsNumpyArray<T>(std::move(arr.data()));
  ret.resize(arr.shape());
  arr.Clear();
  return ret;
}

template <typename T>
py::dict AsNumpyArrayDict(
    const std::unordered_map<std::string, NdArray<T>>& src) {
  py::dict dst;
  for (const auto& [k, v] : src) {
    dst[py::str(k)] = AsNumpyArray<T>(v);
  }
  return dst;
}

template <typename T>
py::dict AsNumpyArrayDict(std::unordered_map<std::string, NdArray<T>>&& src) {
  py::dict dst;
  for (auto& [k, v] : src) {
    dst[py::str(k)] = AsNumpyArray<T>(std::move(v));
  }
  return dst;
}

}  // namespace utils
}  // namespace nocturne
