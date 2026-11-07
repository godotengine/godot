// Copyright (c) the JPEG XL Project Authors. All rights reserved.
//
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#ifndef LIB_JXL_BASE_C_CALLBACK_SUPPORT_H_
#define LIB_JXL_BASE_C_CALLBACK_SUPPORT_H_

#include <utility>

namespace jxl {
namespace detail {

template <typename T>
struct MethodToCCallbackHelper {};

template <typename T, typename R, typename... Args>
struct MethodToCCallbackHelper<R (T::*)(Args...)> {
  template <R (T::*method)(Args...)>
  static R Call(void *opaque, Args... args) {
    return (reinterpret_cast<T *>(opaque)->*method)(
        std::forward<Args>(args)...);
  }
};

}  // namespace detail
}  // namespace jxl

#define METHOD_TO_C_CALLBACK(method) \
  ::jxl::detail::MethodToCCallbackHelper<decltype(method)>::Call<method>

#endif  // LIB_JXL_BASE_C_CALLBACK_SUPPORT_H_
