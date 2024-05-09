// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX_CXX_ATOMIC_H
#define _LIBCUDACXX_CXX_ATOMIC_H

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_impl {
  using __underlying_t = _Tp;
  using __temporary_t = __cxx_atomic_base_impl<_Tp, _Sco>;
  using __wrap_t = __cxx_atomic_base_impl<_Tp, _Sco>;

  static constexpr int __sco = _Sco;

#if !defined(_LIBCUDACXX_COMPILER_GCC) || (__GNUC__ >= 5)
  static_assert(is_trivially_copyable<_Tp>::value,
    "std::atomic<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _LIBCUDACXX_CONSTEXPR
  __cxx_atomic_base_impl() _NOEXCEPT = default;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
  __cxx_atomic_base_impl(_Tp value) _NOEXCEPT : __a_value(value) {}

  _ALIGNAS(sizeof(_Tp)) _Tp __a_value;
};

template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __cxx_get_underlying_atomic(__cxx_atomic_base_impl<_Tp, _Sco> * __a) _NOEXCEPT {
  return &__a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
volatile _Tp* __cxx_get_underlying_atomic(__cxx_atomic_base_impl<_Tp, _Sco> volatile* __a) _NOEXCEPT {
  return &__a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const _Tp* __cxx_get_underlying_atomic(__cxx_atomic_base_impl<_Tp, _Sco> const* __a) _NOEXCEPT {
  return &__a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const volatile _Tp* __cxx_get_underlying_atomic(__cxx_atomic_base_impl<_Tp, _Sco> const volatile* __a) _NOEXCEPT {
  return &__a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
__cxx_atomic_base_impl<_Tp, _Sco>* __cxx_atomic_unwrap(__cxx_atomic_base_impl<_Tp, _Sco>* __a) _NOEXCEPT {
  return __a;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
volatile __cxx_atomic_base_impl<_Tp, _Sco>* __cxx_atomic_unwrap(__cxx_atomic_base_impl<_Tp, _Sco> volatile* __a) _NOEXCEPT {
  return __a;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const __cxx_atomic_base_impl<_Tp, _Sco>* __cxx_atomic_unwrap(__cxx_atomic_base_impl<_Tp, _Sco> const* __a) _NOEXCEPT {
  return __a;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const volatile __cxx_atomic_base_impl<_Tp, _Sco>* __cxx_atomic_unwrap(__cxx_atomic_base_impl<_Tp, _Sco> const volatile* __a) _NOEXCEPT {
  return __a;
}

template <typename _Tp, int _Sco>
struct __cxx_atomic_ref_base_impl {
  using __underlying_t = _Tp;
  using __temporary_t = _Tp;
  using __wrap_t = _Tp;

  static constexpr int __sco = _Sco;

#if !defined(_LIBCUDACXX_COMPILER_GCC) || (__GNUC__ >= 5)
  static_assert(is_trivially_copyable<_Tp>::value,
    "std::atomic_ref<Tp> requires that 'Tp' be a trivially copyable type");
#endif

  _LIBCUDACXX_CONSTEXPR
  __cxx_atomic_ref_base_impl() _NOEXCEPT = delete;

  _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
  __cxx_atomic_ref_base_impl(_Tp& value) _NOEXCEPT : __a_value(&value) {}

  _Tp* __a_value;
};

template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __cxx_get_underlying_atomic(__cxx_atomic_ref_base_impl<_Tp, _Sco>* __a) _NOEXCEPT {
  return __a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
volatile _Tp* __cxx_get_underlying_atomic(__cxx_atomic_ref_base_impl<_Tp, _Sco> volatile* __a) _NOEXCEPT {
  return __a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const _Tp* __cxx_get_underlying_atomic(__cxx_atomic_ref_base_impl<_Tp, _Sco> const* __a) _NOEXCEPT {
  return __a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const volatile _Tp* __cxx_get_underlying_atomic(__cxx_atomic_ref_base_impl<_Tp, _Sco> const volatile* __a) _NOEXCEPT {
  return __a->__a_value;
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __cxx_atomic_unwrap(__cxx_atomic_ref_base_impl<_Tp, _Sco>* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(__a);
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
volatile _Tp* __cxx_atomic_unwrap(__cxx_atomic_ref_base_impl<_Tp, _Sco> volatile* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(__a);
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const _Tp* __cxx_atomic_unwrap(__cxx_atomic_ref_base_impl<_Tp, _Sco> const* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(__a);
}
template <typename _Tp, int _Sco>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const volatile _Tp* __cxx_atomic_unwrap(__cxx_atomic_ref_base_impl<_Tp, _Sco> const volatile* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(__a);
}

template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __cxx_get_underlying_atomic(_Tp* __a) _NOEXCEPT {
  return __a;
}

template <typename _Tp, typename _Up>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
auto __cxx_atomic_wrap_to_base(_Tp*, _Up __val) _NOEXCEPT -> typename _Tp::__wrap_t {
  return typename _Tp::__wrap_t(__val);
}
template <typename _Tp>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
auto __cxx_atomic_base_temporary(_Tp*) _NOEXCEPT -> typename _Tp::__temporary_t {
  return typename _Tp::__temporary_t();
}

template <typename _Tp>
using __cxx_atomic_underlying_t = typename _Tp::__underlying_t;

#endif //_LIBCUDACXX_CXX_ATOMIC_H
