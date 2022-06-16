// Copyright (c) Facebook, Inc. and its affiliates.
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <cstdint>

namespace nocturne {
namespace utils {

class UniqueID {
 public:
  UniqueID() { id_ = next_id_++; }
  UniqueID(const UniqueID& id) : id_(id.id()) {}

  UniqueID& operator=(const UniqueID& id) {
    id_ = id.id();
    return *this;
  }

  int64_t id() const { return id_; }

 protected:
  inline static int64_t next_id_ = 0;
  int64_t id_;
};

}  // namespace utils
}  // namespace nocturne
