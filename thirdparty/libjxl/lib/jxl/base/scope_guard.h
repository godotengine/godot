// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_SCOPE_GUARD_H_
#define LIB_JXL_BASE_SCOPE_GUARD_H_

#include <utility>

namespace jxl {

template <typename Callback>
class ScopeGuard {
 public:
  // Discourage unnecessary moves / copies.
  ScopeGuard(const ScopeGuard &) = delete;
  ScopeGuard &operator=(const ScopeGuard &) = delete;
  ScopeGuard &operator=(ScopeGuard &&) = delete;

  // Pre-C++17 does not guarantee RVO -> require move constructor.
  ScopeGuard(ScopeGuard &&other) noexcept
      : callback_(std::move(other.callback_)) {
    other.armed_ = false;
  }

  template <typename CallbackParam>
  ScopeGuard(CallbackParam &&callback, bool armed)
      : callback_(std::forward<CallbackParam>(callback)), armed_(armed) {}

  ~ScopeGuard() {
    if (armed_) callback_();
  }

  void Disarm() { armed_ = false; }

 private:
  Callback callback_;
  bool armed_;
};

template <typename Callback>
ScopeGuard<Callback> MakeScopeGuard(Callback &&callback) {
  return ScopeGuard<Callback>{std::forward<Callback>(callback), true};
}

}  // namespace jxl

#endif  // LIB_JXL_BASE_SCOPE_GUARD_H_
