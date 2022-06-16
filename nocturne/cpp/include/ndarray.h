// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <numeric>
#include <vector>

namespace nocturne {

template <typename T>
class NdArray {
 public:
  NdArray() = default;

  explicit NdArray(const std::vector<T>& data)
      : size_(data.size()), shape_{data.size()}, data_(data) {}
  explicit NdArray(std::vector<T>&& data)
      : size_(data.size()), shape_{data.size()}, data_(std::move(data)) {}

  NdArray(const std::initializer_list<int64_t>& shape, T val)
      : size_(std::accumulate(shape.begin(), shape.end(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(size_, val) {}
  NdArray(const std::vector<int64_t>& shape, T val)
      : size_(std::accumulate(shape.cbegin(), shape.cend(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(size_, val) {}

  NdArray(const std::initializer_list<int64_t>& shape, const T* data)
      : size_(std::accumulate(shape.begin(), shape.end(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(data, data + size_) {}
  NdArray(const std::vector<int64_t>& shape, const T* data)
      : size_(std::accumulate(shape.cbegin(), shape.cend(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(data, data + size_) {}

  NdArray(const std::initializer_list<int64_t>& shape,
          const std::vector<T>& data)
      : size_(std::accumulate(shape.begin(), shape.end(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(data) {}
  NdArray(const std::vector<int64_t>& shape, const std::vector<T>& data)
      : size_(std::accumulate(shape.cbegin(), shape.cend(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(data) {}

  NdArray(const std::initializer_list<int64_t>& shape, std::vector<T>&& data)
      : size_(std::accumulate(shape.begin(), shape.end(), int64_t(1),
                              std::multiplies<int64_t>())),
        shape_(shape),
        data_(std::move(data)) {}

  NdArray(const NdArray& arr)
      : size_(arr.size_), shape_(arr.shape_), data_(arr.data_) {}
  NdArray(NdArray&& arr)
      : size_(arr.size_),
        shape_(std::move(arr.shape_)),
        data_(std::move(arr.data_)) {
    arr.size_ = 0;
  }

  NdArray& operator=(const NdArray& arr) {
    size_ = arr.size_;
    shape_ = arr.shape_;
    data_ = arr.data_;
    return *this;
  }
  NdArray& operator=(NdArray&& arr) {
    size_ = arr.size_;
    arr.size_ = 0;
    shape_ = std::move(arr.shape_);
    data_ = std::move(arr.data_);
    return *this;
  }

  int64_t size() const { return size_; }

  const std::vector<int64_t>& shape() const { return shape_; }

  const std::vector<T>& data() const { return data_; }
  std::vector<T>& data() { return data_; }

  int64_t Ndim() const { return shape_.size(); }

  const T* DataPtr() const { return data_.data(); }
  T* DataPtr() { return data_.data(); }

  void Clear() {
    size_ = 0;
    shape_.clear();
    data_.clear();
  }

  void Resize(const std::initializer_list<int64_t>& shape) {
    const int64_t size = std::accumulate(shape.begin(), shape.end(), int64_t(1),
                                         std::multiplies<int64_t>());
    assert(size == size_);
    shape_ = shape;
  }

  void Resize(const std::vector<int64_t>& shape) {
    const int64_t size = std::accumulate(
        shape.cbegin(), shape.cend(), int64_t(1), std::multiplies<int64_t>());
    assert(size == size_);
    shape_ = shape;
  }

 protected:
  int64_t size_ = 0;
  std::vector<int64_t> shape_;
  std::vector<T> data_;
};

}  // namespace nocturne
