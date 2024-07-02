//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//


static inline __device__ void __cuda_membar_block() { asm volatile("membar.cta;":::"memory"); }
static inline __device__ void __cuda_fence_acq_rel_block() { asm volatile("fence.acq_rel.cta;":::"memory"); }
static inline __device__ void __cuda_fence_sc_block() { asm volatile("fence.sc.cta;":::"memory"); }
static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_block_tag) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_fence_sc_block(); break;
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_fence_acq_rel_block(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST:
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_membar_block(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    )
  )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.cta.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.cta.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    uint32_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_32_block(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_32_block(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_32_block(__ptr, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_32_block(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.cta.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.cta.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    uint64_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_64_block(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_64_block(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_64_block(__ptr, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_64_block(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_32_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.cta.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_32_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.cta.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_32_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_block_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_32_block(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_32_block(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_block();
          case __ATOMIC_RELAXED: __cuda_store_volatile_32_block(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_64_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.cta.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_64_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.cta.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_64_block(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_block_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_64_block(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_64_block(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_block();
          case __ATOMIC_RELAXED: __cuda_store_volatile_64_block(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.cta.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.cta.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.cta.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.cta.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.cta.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_block_tag) {
    uint32_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 4);
    memcpy(&__old, __expected, 4);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_32_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_32_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_32_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_32_block(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_32_block(__ptr, __old, __old_tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_compare_exchange_volatile_32_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_32_block(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_exchange_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_add_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_and_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_max_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_min_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_or_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.cta.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_sub_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_32_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.cta.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_32_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_xor_volatile_32_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_32_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.cta.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.cta.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.cta.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.cta.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.cta.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_block_tag) {
    uint64_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 8);
    memcpy(&__old, __expected, 8);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_64_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_64_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_64_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_64_block(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_64_block(__ptr, __old, __old_tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_compare_exchange_volatile_64_block(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_64_block(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_block_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_exchange_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_and_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_max_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_min_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_or_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.cta.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_sub_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_64_block(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.cta.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_block_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_xor_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_block_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_block(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_block_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp = -__tmp;
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_block(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_block();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); __cuda_membar_block(); break;
          case __ATOMIC_RELEASE: __cuda_membar_block(); __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_block(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
static inline __device__ void __cuda_membar_device() { asm volatile("membar.gl;":::"memory"); }
static inline __device__ void __cuda_fence_acq_rel_device() { asm volatile("fence.acq_rel.gpu;":::"memory"); }
static inline __device__ void __cuda_fence_sc_device() { asm volatile("fence.sc.gpu;":::"memory"); }
static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_device_tag) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_fence_sc_device(); break;
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_fence_acq_rel_device(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST:
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_membar_device(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    )
  )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.gpu.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.gpu.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    uint32_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_32_device(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_32_device(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_32_device(__ptr, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_32_device(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.gpu.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.gpu.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    uint64_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_64_device(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_64_device(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_64_device(__ptr, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_64_device(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_32_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.gpu.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_32_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.gpu.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_32_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_device_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_32_device(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_32_device(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_device();
          case __ATOMIC_RELAXED: __cuda_store_volatile_32_device(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_64_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.gpu.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_64_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.gpu.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_64_device(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_device_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_64_device(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_64_device(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_device();
          case __ATOMIC_RELAXED: __cuda_store_volatile_64_device(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.gpu.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.gpu.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.gpu.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.gpu.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.gpu.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_device_tag) {
    uint32_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 4);
    memcpy(&__old, __expected, 4);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_32_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_32_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_32_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_32_device(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_32_device(__ptr, __old, __old_tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_compare_exchange_volatile_32_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_32_device(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_exchange_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_add_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_and_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_max_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_min_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_or_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.gpu.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_sub_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_32_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.gpu.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_32_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_xor_volatile_32_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_32_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.gpu.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.gpu.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.gpu.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.gpu.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.gpu.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_device_tag) {
    uint64_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 8);
    memcpy(&__old, __expected, 8);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_64_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_64_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_64_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_64_device(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_64_device(__ptr, __old, __old_tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_compare_exchange_volatile_64_device(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_64_device(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_device_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_exchange_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_and_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_max_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_min_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_or_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.gpu.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_sub_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_64_device(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.gpu.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_device_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_xor_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_device(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_device_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp = -__tmp;
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_device(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_device();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); __cuda_membar_device(); break;
          case __ATOMIC_RELEASE: __cuda_membar_device(); __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_device(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
static inline __device__ void __cuda_membar_system() { asm volatile("membar.sys;":::"memory"); }
static inline __device__ void __cuda_fence_acq_rel_system() { asm volatile("fence.acq_rel.sys;":::"memory"); }
static inline __device__ void __cuda_fence_sc_system() { asm volatile("fence.sc.sys;":::"memory"); }
static inline __device__ void __atomic_thread_fence_cuda(int __memorder, __thread_scope_system_tag) {
  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST: __cuda_fence_sc_system(); break;
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_fence_acq_rel_system(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    ),
    NV_IS_DEVICE, (
      switch (__memorder) {
        case __ATOMIC_SEQ_CST:
        case __ATOMIC_CONSUME:
        case __ATOMIC_ACQUIRE:
        case __ATOMIC_ACQ_REL:
        case __ATOMIC_RELEASE: __cuda_membar_system(); break;
        case __ATOMIC_RELAXED: break;
        default: assert(0);
      }
    )
  )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.sys.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.sys.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b32 %0,[%1];" : "=r"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    uint32_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_32_system(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_32_system(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_32_system(__ptr, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_32_system(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.acquire.sys.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.relaxed.sys.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_load_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst) {asm volatile("ld.volatile.b64 %0,[%1];" : "=l"(__dst) : "l"(__ptr) : "memory"); }
template<class _Type, typename _CUDA_VSTD::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_load_cuda(const volatile _Type *__ptr, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    uint64_t __tmp = 0;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_acquire_64_system(__ptr, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_load_relaxed_64_system(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_load_volatile_64_system(__ptr, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELAXED: __cuda_load_volatile_64_system(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_32_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.sys.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_32_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.sys.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_32_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b32 [%0], %1;" :: "l"(__ptr),"r"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_system_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_32_system(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_32_system(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_system();
          case __ATOMIC_RELAXED: __cuda_store_volatile_32_system(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_relaxed_64_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.relaxed.sys.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_release_64_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.release.sys.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _CUDA_A, class _CUDA_B> static inline __device__ void __cuda_store_volatile_64_system(_CUDA_A __ptr, _CUDA_B __src) { asm volatile("st.volatile.b64 [%0], %1;" :: "l"(__ptr),"l"(__src) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_store_cuda(volatile _Type *__ptr, _Type *__val, int __memorder, __thread_scope_system_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_RELEASE: __cuda_store_release_64_system(__ptr, __tmp); break;
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_RELAXED: __cuda_store_relaxed_64_system(__ptr, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_RELEASE:
          case __ATOMIC_SEQ_CST: __cuda_membar_system();
          case __ATOMIC_RELAXED: __cuda_store_volatile_64_system(__ptr, __tmp); break;
          default: assert(0);
        }
      )
    )
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.sys.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.sys.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.sys.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.sys.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.sys.b32 %0,[%1],%2,%3;" : "=r"(__dst) : "l"(__ptr),"r"(__cmp),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_system_tag) {
    uint32_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 4);
    memcpy(&__old, __expected, 4);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_32_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_32_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_32_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_32_system(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_32_system(__ptr, __old, __old_tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_compare_exchange_volatile_32_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_32_system(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    uint32_t __tmp = 0;
    memcpy(&__tmp, __val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_exchange_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 4);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_add_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_and_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_max_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_min_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_or_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.sys.u32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_sub_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_32_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.sys.b32 %0,[%1],%2;" : "=r"(__dst) : "l"(__ptr),"r"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==4, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint32_t __tmp = 0;
    memcpy(&__tmp, &__val, 4);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_32_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_xor_volatile_32_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_32_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 4);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acq_rel.sys.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.acquire.sys.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.relaxed.sys.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.release.sys.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C, class _CUDA_D> static inline __device__ void __cuda_compare_exchange_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __cmp, _CUDA_D __op) { asm volatile("atom.cas.sys.b64 %0,[%1],%2,%3;" : "=l"(__dst) : "l"(__ptr),"l"(__cmp),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ bool __atomic_compare_exchange_cuda(volatile _Type *__ptr, _Type *__expected, const _Type *__desired, bool, int __success_memorder, int __failure_memorder, __thread_scope_system_tag) {
    uint64_t __tmp = 0, __old = 0, __old_tmp;
    memcpy(&__tmp, __desired, 8);
    memcpy(&__old, __expected, 8);
    __old_tmp = __old;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_acquire_64_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_compare_exchange_acq_rel_64_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_compare_exchange_release_64_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_relaxed_64_system(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__stronger_order_cuda(__success_memorder, __failure_memorder)) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_compare_exchange_volatile_64_system(__ptr, __old, __old_tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_compare_exchange_volatile_64_system(__ptr, __old, __old_tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_compare_exchange_volatile_64_system(__ptr, __old, __old_tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    bool const __ret = __old == __old_tmp;
    memcpy(__expected, &__old, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acq_rel.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.acquire.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.relaxed.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.release.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_exchange_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.exch.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ void __atomic_exchange_cuda(volatile _Type *__ptr, _Type *__val, _Type *__ret, int __memorder, __thread_scope_system_tag) {
    uint64_t __tmp = 0;
    memcpy(&__tmp, __val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_exchange_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_exchange_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_exchange_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_exchange_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_exchange_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(__ret, &__tmp, 8);
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acq_rel.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.acquire.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.relaxed.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.release.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_add_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.add.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_add_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acq_rel.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.acquire.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.relaxed.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.release.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_and_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.and.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_and_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_and_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_and_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_and_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_and_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_and_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acq_rel.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.acquire.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.relaxed.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.release.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_max_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.max.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_max_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_max_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_max_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_max_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_max_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_max_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acq_rel.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.acquire.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.relaxed.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.release.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_min_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.min.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_min_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_min_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_min_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_min_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_min_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_min_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acq_rel.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.acquire.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.relaxed.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.release.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_or_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.or.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_or_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_or_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_or_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_or_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_or_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_or_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acq_rel.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.acquire.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.relaxed.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.release.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_sub_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { __op = -__op;
asm volatile("atom.add.sys.u64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_sub_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_sub_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_sub_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_sub_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_sub_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_sub_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acq_rel_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acq_rel.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_acquire_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.acquire.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_relaxed_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.relaxed.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_release_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.release.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _CUDA_A, class _CUDA_B, class _CUDA_C> static inline __device__ void __cuda_fetch_xor_volatile_64_system(_CUDA_A __ptr, _CUDA_B& __dst, _CUDA_C __op) { asm volatile("atom.xor.sys.b64 %0,[%1],%2;" : "=l"(__dst) : "l"(__ptr),"l"(__op) : "memory"); }
template<class _Type, typename cuda::std::enable_if<sizeof(_Type)==8, int>::type = 0>
__device__ _Type __atomic_fetch_xor_cuda(volatile _Type *__ptr, _Type __val, int __memorder, __thread_scope_system_tag) {
    _Type __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_xor_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_xor_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_relaxed_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_xor_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_xor_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_xor_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_add_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_system(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
template<class _Type>
__device__ _Type* __atomic_fetch_sub_cuda(_Type *volatile *__ptr, ptrdiff_t __val, int __memorder, __thread_scope_system_tag) {
    _Type* __ret;
    uint64_t __tmp = 0;
    memcpy(&__tmp, &__val, 8);
    __tmp = -__tmp;
    __tmp *= sizeof(_Type);
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_70, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST: __cuda_fence_sc_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_acquire_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_ACQ_REL: __cuda_fetch_add_acq_rel_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELEASE: __cuda_fetch_add_release_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_relaxed_64_system(__ptr, __tmp, __tmp); break;
        }
      ),
      NV_IS_DEVICE, (
        switch (__memorder) {
          case __ATOMIC_SEQ_CST:
          case __ATOMIC_ACQ_REL: __cuda_membar_system();
          case __ATOMIC_CONSUME:
          case __ATOMIC_ACQUIRE: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); __cuda_membar_system(); break;
          case __ATOMIC_RELEASE: __cuda_membar_system(); __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          case __ATOMIC_RELAXED: __cuda_fetch_add_volatile_64_system(__ptr, __tmp, __tmp); break;
          default: assert(0);
        }
      )
    )
    memcpy(&__ret, &__tmp, 8);
    return __ret;
}
