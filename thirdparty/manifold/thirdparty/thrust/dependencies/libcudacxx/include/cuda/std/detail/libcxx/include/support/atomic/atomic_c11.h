// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Atomics for C11

template <typename _Tp>
struct __cxx_atomic_base_impl {

  _LIBCUDACXX_INLINE_VISIBILITY
#ifndef _LIBCUDACXX_CXX03_LANG
    __cxx_atomic_base_impl() _NOEXCEPT = default;
#else
    __cxx_atomic_base_impl() _NOEXCEPT : __a_value() {}
#endif // _LIBCUDACXX_CXX03_LANG
  _LIBCUDACXX_CONSTEXPR explicit __cxx_atomic_base_impl(_Tp value) _NOEXCEPT
    : __a_value(value) {}
  _LIBCUDACXX_DISABLE_EXTENSION_WARNING _Atomic(_Tp) __a_value;
};

#ifndef _LIBCUDACXX_ATOMIC_IS_LOCK_FREE
#define _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(__x) __c11_atomic_is_lock_free(__x, 0)
#endif

_LIBCUDACXX_INLINE_VISIBILITY inline
void __cxx_atomic_thread_fence(memory_order __order) _NOEXCEPT {
    __c11_atomic_thread_fence(static_cast<__memory_order_underlying_t>(__order));
}

_LIBCUDACXX_INLINE_VISIBILITY inline
void __cxx_atomic_signal_fence(memory_order __order) _NOEXCEPT {
    __c11_atomic_signal_fence(static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_init(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __val) _NOEXCEPT {
    __c11_atomic_init(&__a->__a_value, __val);
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_init(__cxx_atomic_base_impl<_Tp> * __a, _Tp __val) _NOEXCEPT {
    __c11_atomic_init(&__a->__a_value, __val);
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_store(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __val, memory_order __order) _NOEXCEPT {
    __c11_atomic_store(&__a->__a_value, __val, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
void __cxx_atomic_store(__cxx_atomic_base_impl<_Tp> * __a, _Tp __val, memory_order __order) _NOEXCEPT {
    __c11_atomic_store(&__a->__a_value, __val, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_load(__cxx_atomic_base_impl<_Tp> const volatile* __a, memory_order __order) _NOEXCEPT {
    using __ptr_type = typename remove_const<decltype(__a->__a_value)>::type*;
    return __c11_atomic_load(const_cast<__ptr_type>(&__a->__a_value), static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_load(__cxx_atomic_base_impl<_Tp> const* __a, memory_order __order) _NOEXCEPT {
    using __ptr_type = typename remove_const<decltype(__a->__a_value)>::type*;
    return __c11_atomic_load(const_cast<__ptr_type>(&__a->__a_value), static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_exchange(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __value, memory_order __order) _NOEXCEPT {
    return __c11_atomic_exchange(&__a->__a_value, __value, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_exchange(__cxx_atomic_base_impl<_Tp> * __a, _Tp __value, memory_order __order) _NOEXCEPT {
    return __c11_atomic_exchange(&__a->__a_value, __value, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return __c11_atomic_compare_exchange_strong(&__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_impl<_Tp> * __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return __c11_atomic_compare_exchange_strong(&__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return __c11_atomic_compare_exchange_weak(&__a->__a_value, __expected, __value, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_impl<_Tp> * __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) _NOEXCEPT {
    return __c11_atomic_compare_exchange_weak(&__a->__a_value, __expected, __value,  static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp> * __a, _Tp __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp* __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp*> volatile* __a, ptrdiff_t __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp* __cxx_atomic_fetch_add(__cxx_atomic_base_impl<_Tp*> * __a, ptrdiff_t __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_add(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_sub(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp> * __a, _Tp __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_sub(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp* __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp*> volatile* __a, ptrdiff_t __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_sub(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp* __cxx_atomic_fetch_sub(__cxx_atomic_base_impl<_Tp*> * __a, ptrdiff_t __delta, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_sub(&__a->__a_value, __delta, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_and(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_and(__cxx_atomic_base_impl<_Tp> * __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_and(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_or(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_or(__cxx_atomic_base_impl<_Tp> * __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_or(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}

template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl<_Tp> volatile* __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_xor(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}
template<class _Tp>
_LIBCUDACXX_INLINE_VISIBILITY
_Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_impl<_Tp> * __a, _Tp __pattern, memory_order __order) _NOEXCEPT {
    return __c11_atomic_fetch_xor(&__a->__a_value, __pattern, static_cast<__memory_order_underlying_t>(__order));
}
