// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
#pragma once

#include <algorithm>
#include <vector>

namespace nocturne {
namespace utils {

template <typename MaskType, typename DataType>
int64_t MaskedPartition(const std::vector<MaskType>& mask,
                        std::vector<DataType>& data) {
  const int64_t n = data.size();
  int64_t pivot = 0;
  for (; pivot < n && static_cast<bool>(mask[pivot]); ++pivot)
    ;
  for (int64_t i = pivot + 1; i < n; ++i) {
    if (static_cast<bool>(mask[i])) {
      std::swap(data[i], data[pivot++]);
    }
  }
  return pivot;
}

// std::partial_sort use heap_sort which slower than std::sort especially when k
// is large.
template <class RandomIt, class CompareFunc>
void PartialSort(RandomIt first, RandomIt middle, RandomIt last,
                 CompareFunc cmp) {
  std::nth_element(first, middle, last, cmp);
  std::sort(first, middle, cmp);
}

// The function FindWithDefault is adapted from protobuf's map_util.h
// implementation.
//
// That code is under the following copyright:
// Copyright 2008 Google Inc.  All rights reserved.
//
// https://github.com/protocolbuffers/protobuf/blob/bd85edfbd9ba449f7a7bda61b8f7e7b9986c0dc1/src/google/protobuf/stubs/map_util.h#L123
template <class Collection>
const typename Collection::value_type::second_type& FindWithDefault(
    const Collection& collection,
    const typename Collection::value_type::first_type& key,
    const typename Collection::value_type::second_type& value) {
  typename Collection::const_iterator it = collection.find(key);
  if (it == collection.end()) {
    return value;
  }
  return it->second;
}

}  // namespace utils
}  // namespace nocturne
