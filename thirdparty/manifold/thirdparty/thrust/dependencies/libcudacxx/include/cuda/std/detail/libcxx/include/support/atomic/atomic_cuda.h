//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#if defined(__CUDA_MINIMUM_ARCH__) && ((!defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 600) || (defined(_MSC_VER) && __CUDA_MINIMUM_ARCH__ < 700))
#  error "CUDA atomics are only supported for sm_60 and up on *nix and sm_70 and up on Windows."
#endif

#if !defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)
#define ATOMIC_BOOL_LOCK_FREE      2
#define ATOMIC_CHAR_LOCK_FREE      2
#define ATOMIC_CHAR16_T_LOCK_FREE  2
#define ATOMIC_CHAR32_T_LOCK_FREE  2
#define ATOMIC_WCHAR_T_LOCK_FREE   2
#define ATOMIC_SHORT_LOCK_FREE     2
#define ATOMIC_INT_LOCK_FREE       2
#define ATOMIC_LONG_LOCK_FREE      2
#define ATOMIC_LLONG_LOCK_FREE     2
#define ATOMIC_POINTER_LOCK_FREE   2
#endif //!defined(__CLANG_ATOMIC_BOOL_LOCK_FREE) && !defined(__GCC_ATOMIC_BOOL_LOCK_FREE)

#ifndef __ATOMIC_RELAXED
#define __ATOMIC_RELAXED 0
#define __ATOMIC_CONSUME 1
#define __ATOMIC_ACQUIRE 2
#define __ATOMIC_RELEASE 3
#define __ATOMIC_ACQ_REL 4
#define __ATOMIC_SEQ_CST 5
#endif //__ATOMIC_RELAXED

inline __host__ __device__ int __stronger_order_cuda(int __a, int __b) {
    int const __max = __a > __b ? __a : __b;
    if(__max != __ATOMIC_RELEASE)
        return __max;
    static int const __xform[] = {
        __ATOMIC_RELEASE,
        __ATOMIC_ACQ_REL,
        __ATOMIC_ACQ_REL,
        __ATOMIC_RELEASE };
    return __xform[__a < __b ? __a : __b];
}

// pre-define lock free query for heterogeneous compatibility
#ifndef _LIBCUDACXX_ATOMIC_IS_LOCK_FREE
#define _LIBCUDACXX_ATOMIC_IS_LOCK_FREE(__x) (__x <= 8)
#endif

// Wrap host atomic implementations into a sub-namespace
namespace __host {
#if defined(_LIBCUDACXX_COMPILER_MSVC)
#  include "atomic_msvc.h"
#elif defined (_LIBCUDACXX_HAS_GCC_ATOMIC_IMP)
#  include "atomic_gcc.h"
#elif defined (_LIBCUDACXX_HAS_C11_ATOMIC_IMP)
//TODO
// #  include "atomic_c11.h"
#elif defined(_LIBCUDACXX_COMPILER_NVRTC)
#  include "atomic_nvrtc.h"
#endif
}

#include "atomic_cuda_generated.h"
#include "atomic_cuda_derived.h"

_LIBCUDACXX_INLINE_VISIBILITY
inline
 void __cxx_atomic_thread_fence(memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_thread_fence_cuda(static_cast<__memory_order_underlying_t>(__order), __thread_scope_system_tag());
        ),
        NV_IS_HOST, (
            __host::__cxx_atomic_thread_fence(__order);
        )
    )
}

_LIBCUDACXX_INLINE_VISIBILITY
inline
 void __cxx_atomic_signal_fence(memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_signal_fence_cuda(static_cast<__memory_order_underlying_t>(__order));
        ),
        NV_IS_HOST, (
            __host::__cxx_atomic_signal_fence(__order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref = false>
struct __cxx_atomic_base_heterogeneous_impl {
    __cxx_atomic_base_heterogeneous_impl() noexcept = default;

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
      __cxx_atomic_base_heterogeneous_impl(_Tp __value) : __a_value(__value) {
    }

    __host::__cxx_atomic_base_impl<_Tp, _Sco> __a_value;
};

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, true> {
    __cxx_atomic_base_heterogeneous_impl() noexcept = default;

    static_assert(sizeof(_Tp) >= 4, "atomic_ref does not support 1 or 2 byte types");
    static_assert(sizeof(_Tp) <= 8, "atomic_ref does not support types larger than 8 bytes");

    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
      __cxx_atomic_base_heterogeneous_impl(_Tp& __value) : __a_value(__value) {
    }

    __host::__cxx_atomic_ref_base_impl<_Tp, _Sco> __a_value;
};

template <typename _Tp, int _Sco, bool _Ref>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
_Tp* __cxx_get_underlying_device_atomic(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> * __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(&__a->__a_value);
}

template <typename _Tp, int _Sco, bool _Ref>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
volatile _Tp* __cxx_get_underlying_device_atomic(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(&__a->__a_value);
}

template <typename _Tp, int _Sco, bool _Ref>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const _Tp* __cxx_get_underlying_device_atomic(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> const* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(&__a->__a_value);
}

template <typename _Tp, int _Sco, bool _Ref>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR
const volatile _Tp* __cxx_get_underlying_device_atomic(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> const volatile* __a) _NOEXCEPT {
  return __cxx_get_underlying_atomic(&__a->__a_value);
}

template <typename _Tp, int _Sco>
struct __cxx_atomic_base_small_impl {
    __cxx_atomic_base_small_impl() noexcept = default;
    _LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR explicit
      __cxx_atomic_base_small_impl(_Tp __value) : __a_value(__value) {
    }

    __cxx_atomic_base_heterogeneous_impl<uint32_t, _Sco, false> __a_value;
};

template <typename _Tp>
using __cxx_small_proxy = typename conditional<sizeof(_Tp) == 1,
                                               uint8_t,
                                               typename conditional<sizeof(_Tp) == 2,
                                                                    uint16_t,
                                                                    void>::type >::type;

template <typename _Tp, int _Sco>
using __cxx_atomic_base_impl = typename conditional<sizeof(_Tp) < 4,
                                    __cxx_atomic_base_small_impl<_Tp, _Sco>,
                                    __cxx_atomic_base_heterogeneous_impl<_Tp, _Sco> >::type;


template <typename _Tp, int _Sco>
using __cxx_atomic_ref_base_impl = __cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, true>;

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 void __cxx_atomic_init(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __val) {
    alignas(_Tp) auto __tmp = __val;
    __cxx_atomic_assign_volatile(*__cxx_get_underlying_device_atomic(__a), __tmp);
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 void __cxx_atomic_store(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __val, memory_order __order) {
    alignas(_Tp) auto __tmp = __val;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            __atomic_store_n_cuda(__cxx_get_underlying_device_atomic(__a), __tmp, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            __host::__cxx_atomic_store(&__a->__a_value, __tmp, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_load(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> const volatile* __a, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_load_n_cuda(__cxx_get_underlying_device_atomic(__a), static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_load(&__a->__a_value, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_exchange(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __val, memory_order __order) {
    alignas(_Tp) auto __tmp = __val;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_exchange_n_cuda(__cxx_get_underlying_device_atomic(__a), __tmp, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_exchange(&__a->__a_value, __tmp, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp* __expected, _Tp __val, memory_order __success, memory_order __failure) {
    alignas(_Tp) auto __tmp = *__expected;
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            alignas(_Tp) auto __tmp_v = __val;
            __result = __atomic_compare_exchange_cuda(__cxx_get_underlying_device_atomic(__a), &__tmp, &__tmp_v, false, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            __result = __host::__cxx_atomic_compare_exchange_strong(&__a->__a_value, &__tmp, __val, __success, __failure);
        )
    )
    *__expected = __tmp;
    return __result;
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp* __expected, _Tp __val, memory_order __success, memory_order __failure) {
    alignas(_Tp) auto __tmp = *__expected;
    bool __result = false;
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            alignas(_Tp) auto __tmp_v = __val;
            __result = __atomic_compare_exchange_cuda(__cxx_get_underlying_device_atomic(__a), &__tmp, &__tmp_v, true, static_cast<__memory_order_underlying_t>(__success), static_cast<__memory_order_underlying_t>(__failure), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            __result = __host::__cxx_atomic_compare_exchange_weak(&__a->__a_value, &__tmp, __val, __success, __failure);
        )
    )
    *__expected = __tmp;
    return __result;
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __delta, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_add_cuda(__cxx_get_underlying_device_atomic(__a), __delta, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_add(&__a->__a_value, __delta, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp* __cxx_atomic_fetch_add(__cxx_atomic_base_heterogeneous_impl<_Tp*, _Sco, _Ref> volatile* __a, ptrdiff_t __delta, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_add_cuda(__cxx_get_underlying_device_atomic(__a), __delta, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_add(&__a->__a_value, __delta, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __delta, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_sub_cuda(__cxx_get_underlying_device_atomic(__a), __delta, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_sub(&__a->__a_value, __delta, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp* __cxx_atomic_fetch_sub(__cxx_atomic_base_heterogeneous_impl<_Tp*, _Sco, _Ref> volatile* __a, ptrdiff_t __delta, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_sub_cuda(__cxx_get_underlying_device_atomic(__a), __delta, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_sub(&__a->__a_value, __delta, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __pattern, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_and_cuda(__cxx_get_underlying_device_atomic(__a), __pattern, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_and(&__a->__a_value, __pattern, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __pattern, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_or_cuda(__cxx_get_underlying_device_atomic(__a), __pattern, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_or(&__a->__a_value, __pattern, __order);
        )
    )
}

template <typename _Tp, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Tp __pattern, memory_order __order) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_xor_cuda(__cxx_get_underlying_device_atomic(__a), __pattern, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ),
        NV_IS_HOST, (
            return __host::__cxx_atomic_fetch_xor(&__a->__a_value, __pattern, __order);
        )
    )
}

template <typename _Tp, typename _Delta, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_max(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Delta __val, memory_order __order) {
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_max_cuda(__cxx_get_underlying_device_atomic(__a), __val, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ), (
            return __host::__cxx_atomic_fetch_max(&__a->__a_value, __val, __order);
        )
    )
}

template <typename _Tp, typename _Delta, int _Sco, bool _Ref>
__host__ __device__
 _Tp __cxx_atomic_fetch_min(__cxx_atomic_base_heterogeneous_impl<_Tp, _Sco, _Ref> volatile* __a, _Delta __val, memory_order __order) {
    NV_IF_TARGET(
        NV_IS_DEVICE, (
            return __atomic_fetch_min_cuda(__cxx_get_underlying_device_atomic(__a), __val, static_cast<__memory_order_underlying_t>(__order), __scope_tag<_Sco>());
        ), (
            return __host::__cxx_atomic_fetch_min(&__a->__a_value, __val, __order);
        )
    )
}

template<class _Tp>
__host__ __device__ inline uint32_t __cxx_small_to_32(_Tp __val) {
    __cxx_small_proxy<_Tp> __temp = 0;
    memcpy(&__temp, &__val, sizeof(_Tp));
    return __temp;
}

template<class _Tp>
__host__ __device__ inline _Tp __cxx_small_from_32(uint32_t __val) {
    __cxx_small_proxy<_Tp> __temp = static_cast<__cxx_small_proxy<_Tp>>(__val);
    _Tp __result;
    memcpy(&__result, &__temp, sizeof(_Tp));
    return __result;
}

template <typename _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_init(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __val) {
    __cxx_atomic_init(&__a->__a_value, __cxx_small_to_32(__val));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline void __cxx_atomic_store(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __val, memory_order __order) {
    __cxx_atomic_store(&__a->__a_value, __cxx_small_to_32(__val), __order);
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_load(__cxx_atomic_base_small_impl<_Tp, _Sco> const volatile* __a, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_load(&__a->__a_value, __order));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_exchange(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __value, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_exchange(&__a->__a_value, __cxx_small_to_32(__value), __order));
}
__host__ __device__
inline int __cuda_memcmp(void const * __lhs, void const * __rhs, size_t __count) {
    NV_DISPATCH_TARGET(
        NV_IS_DEVICE, (
            auto __lhs_c = reinterpret_cast<unsigned char const *>(__lhs);
            auto __rhs_c = reinterpret_cast<unsigned char const *>(__rhs);
            while (__count--) {
                auto const __lhs_v = *__lhs_c++;
                auto const __rhs_v = *__rhs_c++;
                if (__lhs_v < __rhs_v) { return -1; }
                if (__lhs_v > __rhs_v) { return 1; }
            }
            return 0;
        ),
        NV_IS_HOST, (
            return memcmp(__lhs, __rhs, __count);
        )
    )
}

template <typename _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_weak(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) {
    auto __temp = __cxx_small_to_32(*__expected);
    auto const __ret = __cxx_atomic_compare_exchange_weak(&__a->__a_value, &__temp, __cxx_small_to_32(__value), __success, __failure);
    auto const __actual = __cxx_small_from_32<_Tp>(__temp);
    if(!__ret) {
        if(0 == __cuda_memcmp(&__actual, __expected, sizeof(_Tp)))
            __cxx_atomic_fetch_and(&__a->__a_value, (1u << (8*sizeof(_Tp))) - 1, memory_order_relaxed);
        else
            *__expected = __actual;
    }
    return __ret;
}

template <typename _Tp, int _Sco>
__host__ __device__ inline bool __cxx_atomic_compare_exchange_strong(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp* __expected, _Tp __value, memory_order __success, memory_order __failure) {
    auto const __old = *__expected;
    while(1) {
        if(__cxx_atomic_compare_exchange_weak(__a, __expected, __value, __success, __failure))
            return true;
        if(0 != __cuda_memcmp(&__old, __expected, sizeof(_Tp)))
            return false;
    }
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_add(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __delta, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_add(&__a->__a_value, __cxx_small_to_32(__delta), __order));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_sub(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __delta, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_sub(&__a->__a_value, __cxx_small_to_32(__delta), __order));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_and(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __pattern, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_and(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_or(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __pattern, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_or(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}

template <typename _Tp, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_xor(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Tp __pattern, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_xor(&__a->__a_value, __cxx_small_to_32(__pattern), __order));
}

template <typename _Tp, typename _Delta, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_max(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Delta __val, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_max(&__a->__a_value, __cxx_small_to_32(__val), __order));
}

template <typename _Tp, typename _Delta, int _Sco>
__host__ __device__ inline _Tp __cxx_atomic_fetch_min(__cxx_atomic_base_small_impl<_Tp, _Sco> volatile* __a, _Delta __val, memory_order __order) {
    return __cxx_small_from_32<_Tp>(__cxx_atomic_fetch_min(&__a->__a_value, __cxx_small_to_32(__val), __order));
}
