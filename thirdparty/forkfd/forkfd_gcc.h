/****************************************************************************
**
** Copyright (C) 2015 Intel Corporation
**
** Permission is hereby granted, free of charge, to any person obtaining a copy
** of this software and associated documentation files (the "Software"), to deal
** in the Software without restriction, including without limitation the rights
** to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
** copies of the Software, and to permit persons to whom the Software is
** furnished to do so, subject to the following conditions:
**
** The above copyright notice and this permission notice shall be included in
** all copies or substantial portions of the Software.
**
** THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
** IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
** FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
** AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
** LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
** OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
** THE SOFTWARE.
**
****************************************************************************/

#ifndef FFD_ATOMIC_GCC_H
#define FFD_ATOMIC_GCC_H

/* atomics */
/* we'll use the GCC 4.7 atomic builtins
 * See http://gcc.gnu.org/onlinedocs/gcc/_005f_005fatomic-Builtins.html#_005f_005fatomic-Builtins
 * Or in texinfo: C Extensions > __atomic Builtins
 */
typedef int ffd_atomic_int;
#define ffd_atomic_pointer(type)    type*

#define FFD_ATOMIC_INIT(val)    (val)
// acq_rel & cst not necessary

#if !defined(__GNUC__) || \
    ((__GNUC__ - 0) * 100 + (__GNUC_MINOR__ - 0)) < 407 || \
    (defined(__INTEL_COMPILER) && __INTEL_COMPILER-0 < 1310) || \
    (defined(__clang__) && ((__clang_major__-0) * 100 + (__clang_minor-0)) < 303)

#define FFD_ATOMIC_RELAXED  ((void)0)
#define FFD_ATOMIC_ACQUIRE  ((void)0)
#define FFD_ATOMIC_RELEASE  ((void)0)

#define ffd_atomic_load(ptr,order) *(ptr)
#define ffd_atomic_store(ptr,val,order) (*(ptr) = (val), (void)0)
#define ffd_atomic_exchange(ptr,val,order) __sync_lock_test_and_set(ptr, val)
#define ffd_atomic_compare_exchange(ptr,expected,desired,order1,order2) \
    __sync_bool_compare_and_swap(ptr, *(expected), desired) ? 1 : \
    (*(expected) = *(ptr), 0)
#define ffd_atomic_add_fetch(ptr,val,order) __sync_add_and_fetch(ptr, val)
#else

#define FFD_ATOMIC_RELAXED  __ATOMIC_RELAXED
#define FFD_ATOMIC_ACQUIRE  __ATOMIC_ACQUIRE
#define FFD_ATOMIC_RELEASE  __ATOMIC_RELEASE

#define ffd_atomic_load(ptr,order) __atomic_load_n(ptr, order)
#define ffd_atomic_store(ptr,val,order) __atomic_store_n(ptr, val, order)
#define ffd_atomic_exchange(ptr,val,order) __atomic_exchange_n(ptr, val, order)
#define ffd_atomic_compare_exchange(ptr,expected,desired,order1,order2) \
    __atomic_compare_exchange_n(ptr, expected, desired, 1, order1, order2)
#define ffd_atomic_add_fetch(ptr,val,order) __atomic_add_fetch(ptr, val, order)
#endif

#endif
