// SPDX-License-Identifier: Apache 2.0
// Copyright 2022-Present Light Transport Entertainment Inc.
#pragma once

#include <cstdint>
#include <algorithm>
//#include <iostream>
//#include <cassert>
#include <limits>
#include <vector>

namespace tinyusdz {

///
/// Simple handle resource manager
/// Assume T is an unsigned integer type.
/// TODO(LTE): Allocate handle for a given value range. e.g. [minVal, maxVal)
///
template<typename T = uint32_t>
class HandleAllocator {
public:
  // id = 0 is reserved.
  HandleAllocator() : counter_(static_cast<T>(1)){}
  //~HandleAllocator(){}

  /// Allocates handle object.
  bool Allocate(T *dst) {

    if (!dst) {
      return false;
    }

    T handle = 0;

    if (!freeList_.empty()) {
      // Reuse last element.
      handle = freeList_.back();
      freeList_.pop_back();
      // Delay sort until required
      dirty_ = true;
      (*dst) = handle;
      return true;
    }

    handle = counter_;
    if ((handle >= static_cast<T>(1)) && (handle < (std::numeric_limits<T>::max)())) {
      counter_++;
      //std::cout << "conter = " << counter_ << "\n";
      (*dst) = handle;
      return true;
    }

    return false;
  }

  /// Release handle object.
  bool Release(const T handle) {
    if (handle == counter_ - static_cast<T>(1)) {
      if (counter_ > static_cast<T>(1)) {
        counter_--;
      } else {
        return false;
      }
    } else {
      if (handle >= static_cast<T>(1)) {
        freeList_.push_back(handle);
        // Delay sort until required
        dirty_ = true;
      } else {
        // invalid handle
        return false;
      }
    }

    return true;
  }

  bool Has(const T handle) const {
    if (dirty_) {
      std::sort(freeList_.begin(), freeList_.end());
      dirty_ = false;
    }

    if (handle < 1) {
      return false;
    }

    // Do binary search.
    if (std::binary_search(freeList_.begin(), freeList_.end(), handle)) {
      return false;
    }

    if (handle >= counter_) {
      return false;
    }

    return true;
  }

  int64_t Size() const {
    return counter_ - freeList_.size() - 1;
  }

private:
  // TODO: Use unorderd_set or unorderd_map for efficiency?
  // worst case complexity is still c.size() though.
  mutable std::vector<T> freeList_; // will be sorted in `Has` call.
  T counter_{};
  mutable bool dirty_{true};
};

} // namespace tinyusdz
