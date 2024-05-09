//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

template<class _Type, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type) <= 2, int>::type = 0>
bool __device__ __atomic_compare_exchange_cuda(_Type volatile *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, _Scope __s) {

    auto const __aligned = (uint32_t*)((intptr_t)__ptr & ~(sizeof(uint32_t) - 1));
    auto const __offset = uint32_t((intptr_t)__ptr & (sizeof(uint32_t) - 1)) * 8;
    auto const __mask = ((1 << sizeof(_Type)*8) - 1) << __offset;

    uint32_t __old = *__expected << __offset;
    uint32_t __old_value;
    while (1) {
        __old_value = (__old & __mask) >> __offset;
        if (__old_value != *__expected)
            break;
        uint32_t const __attempt = (__old & ~__mask) | (*__desired << __offset);
        if (__atomic_compare_exchange_cuda(__aligned, &__old, &__attempt, true, __success_memorder, __failure_memorder, __s))
            return true;
    }
    *__expected = __old_value;
    return false;
}

template<class _Type, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
void __device__ __atomic_exchange_cuda(_Type volatile *__ptr, _Type *__val, _Type *__ret, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, __val, true, __memorder, __memorder, __s))
        ;
    *__ret = __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __device__ __atomic_fetch_add_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected + __val;
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s))
        __desired = __expected + __val;
    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __host__ __device__ __atomic_fetch_max_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {
    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected > __val ? __expected : __val;

    while(__desired == __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s)) {
        __desired = __expected > __val ? __expected : __val;
    }

    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __host__ __device__ __atomic_fetch_min_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {
    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected < __val ? __expected : __val;

    while(__desired != __val &&
            !__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s)) {
        __desired = __expected < __val ? __expected : __val;
    }

    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __device__ __atomic_fetch_sub_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected - __val;
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s))
        __desired = __expected - __val;
    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __device__ __atomic_fetch_and_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected & __val;
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s))
        __desired = __expected & __val;
    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __device__ __atomic_fetch_xor_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected ^ __val;
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s))
        __desired = __expected ^ __val;
    return __expected;
}

template<class _Type, class _Delta, class _Scope, typename _CUDA_VSTD::enable_if<sizeof(_Type)<=2, int>::type = 0>
_Type __device__ __atomic_fetch_or_cuda(_Type volatile *__ptr, _Delta __val, int __memorder, _Scope __s) {

    _Type __expected = __atomic_load_n_cuda(__ptr, __ATOMIC_RELAXED, __s);
    _Type __desired = __expected | __val;
    while(!__atomic_compare_exchange_cuda(__ptr, &__expected, &__desired, true, __memorder, __memorder, __s))
        __desired = __expected | __val;
    return __expected;
}

template<class _Type, class _Scope>
_Type __device__ __atomic_load_n_cuda(const _Type volatile *__ptr, int __memorder, _Scope __s) {
    _Type __ret;
    __atomic_load_cuda(__ptr, &__ret, __memorder, __s);
    return __ret;
}

template<class _Type, class _Scope>
void __device__ __atomic_store_n_cuda(_Type volatile *__ptr, _Type __val, int __memorder, _Scope __s) {
    __atomic_store_cuda(__ptr, &__val, __memorder, __s);
}

template<class _Type, class _Scope>
bool __device__ __atomic_compare_exchange_n_cuda(_Type volatile *__ptr, _Type *__expected, _Type __desired, bool __weak, int __success_memorder, int __failure_memorder, _Scope __s) {
    return __atomic_compare_exchange_cuda(__ptr, __expected, &__desired, __weak, __success_memorder, __failure_memorder, __s);
}

template<class _Type, class _Scope>
_Type __device__ __atomic_exchange_n_cuda(_Type volatile *__ptr, _Type __val, int __memorder, _Scope __s) {
    _Type __ret;
    __atomic_exchange_cuda(__ptr, &__val, &__ret, __memorder, __s);
    return __ret;
}

static inline __device__ void __atomic_signal_fence_cuda(int) {
    asm volatile("":::"memory");
}

