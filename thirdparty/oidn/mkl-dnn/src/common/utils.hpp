/*******************************************************************************
* Copyright 2016-2018 Intel Corporation
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*     http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*******************************************************************************/

#ifndef UTILS_HPP
#define UTILS_HPP

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <stdint.h>

#if defined(__x86_64__) || defined(_M_X64)
#define MKLDNN_X86_64
#endif

#define MSAN_ENABLED 0
#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#undef MSAN_ENABLED
#define MSAN_ENABLED 1
#include <sanitizer/msan_interface.h>
#endif
#endif

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "z_magic.hpp"

namespace mkldnn {
namespace impl {

// Sanity check for 64 bits
static_assert(sizeof(void*) == 8, "Intel(R) MKL-DNN supports 64 bit only");

#define CHECK(f) do { \
    status_t status = f; \
    if (status != status::success) \
    return status; \
} while (0)

#define IMPLICATION(cause, effect) (!(cause) || !!(effect))

namespace utils {

/* a bunch of std:: analogues to be compliant with any msvs version
 *
 * Rationale: msvs c++ (and even some c) headers contain special pragma that
 * injects msvs-version check into object files in order to abi-mismatches
 * during the static linking. This makes sense if e.g. std:: objects are passed
 * through between application and library, which is not the case for mkl-dnn
 * (since there is no any c++-rt dependent stuff, ideally...). */

/* SFINAE helper -- analogue to std::enable_if */
template<bool expr, class T = void> struct enable_if {};
template<class T> struct enable_if<true, T> { typedef T type; };

/* analogue std::conditional */
template <bool, typename, typename> struct conditional {};
template <typename T, typename F> struct conditional<true, T, F>
{ typedef T type; };
template <typename T, typename F> struct conditional<false, T, F>
{ typedef F type; };

template <bool, typename, bool, typename, typename> struct conditional3 {};
template <typename T, typename FT, typename FF>
struct conditional3<true, T, false, FT, FF> { typedef T type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, true, FT, FF> { typedef FT type; };
template <typename T, typename FT, typename FF>
struct conditional3<false, T, false, FT, FF> { typedef FF type; };

template <bool, typename U, U, U> struct conditional_v {};
template <typename U, U t, U f> struct conditional_v<true, U, t, f>
{ static constexpr U value = t; };
template <typename U, U t, U f> struct conditional_v<false, U, t, f>
{ static constexpr U value = f; };

template <typename T> struct remove_reference { typedef T type; };
template <typename T> struct remove_reference<T&> { typedef T type; };
template <typename T> struct remove_reference<T&&> { typedef T type; };

template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &t)
{ return static_cast<T&&>(t); }
template <typename T>
inline T&& forward(typename utils::remove_reference<T>::type &&t)
{ return static_cast<T&&>(t); }

template <typename T>
inline typename remove_reference<T>::type zero()
{ auto zero = typename remove_reference<T>::type(); return zero; }

template <typename T, typename P>
inline bool everyone_is(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
inline bool everyone_is(T val, P item, Args... item_others) {
    return val == item && everyone_is(val, item_others...);
}

template <typename T, typename P>
constexpr bool one_of(T val, P item) { return val == item; }
template <typename T, typename P, typename... Args>
constexpr bool one_of(T val, P item, Args... item_others) {
    return val == item || one_of(val, item_others...);
}

template <typename... Args>
inline bool any_null(Args... ptrs) { return one_of(nullptr, ptrs...); }

template<typename T>
inline void array_copy(T *dst, const T *src, size_t size) {
    for (size_t i = 0; i < size; ++i) dst[i] = src[i];
}
template<typename T>
inline bool array_cmp(const T *a1, const T *a2, size_t size) {
    for (size_t i = 0; i < size; ++i) if (a1[i] != a2[i]) return false;
    return true;
}
template<typename T, typename U>
inline void array_set(T *arr, const U& val, size_t size) {
    for (size_t i = 0; i < size; ++i) arr[i] = static_cast<T>(val);
}

namespace product_impl {
template<size_t> struct int2type{};

template <typename T>
constexpr int product_impl(const T *arr, int2type<0>) { return arr[0]; }

template <typename T, size_t num>
inline T product_impl(const T *arr, int2type<num>) {
    return arr[0]*product_impl(arr+1, int2type<num-1>()); }
}

template <size_t num, typename T>
inline T array_product(const T *arr) {
    return product_impl::product_impl(arr, product_impl::int2type<num-1>());
}

template<typename T, typename R = T>
inline R array_product(const T *arr, size_t size) {
    R prod = 1;
    for (size_t i = 0; i < size; ++i) prod *= arr[i];
    return prod;
}

/** sorts an array of values using @p comparator. While sorting the array
 * of value, the function permutes an array of @p keys accordingly.
 *
 * @note The arrays of @p keys can be omitted. In this case the function
 *       sorts the array of @vals only.
 */
template <typename T, typename U, typename F>
inline void simultaneous_sort(T *vals, U *keys, size_t size, F comparator) {
    if (size == 0) return;

    for (size_t i = 0; i < size - 1; ++i) {
        bool swapped = false;

        for (size_t j = 0; j < size - i - 1; j++) {
            if (comparator(vals[j], vals[j + 1]) > 0) {
                nstl::swap(vals[j], vals[j + 1]);
                if (keys) nstl::swap(keys[j], keys[j + 1]);
                swapped = true;
            }
        }

        if (swapped == false) break;
    }
}

template <typename T, typename U>
inline typename remove_reference<T>::type div_up(const T a, const U b) {
    assert(b);
    return (a + b - 1) / b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_up(const T a, const U b) {
    return div_up(a, b) * b;
}

template <typename T, typename U>
inline typename remove_reference<T>::type rnd_dn(const T a, const U b) {
    return (a / b) * b;
}

template <typename T> T *align_ptr(T *ptr, uintptr_t alignment)
{ return (T *)(((uintptr_t)ptr + alignment - 1) & ~(alignment - 1)); }

template <typename T, typename U, typename V>
inline U this_block_size(const T offset, const U max, const V block_size) {
    assert(offset < max);
    // TODO (Roma): can't use nstl::max() due to circular dependency... we
    // need to fix this
    const T block_boundary = offset + block_size;
    if (block_boundary > max)
        return max - offset;
    else
        return block_size;
}

template<typename T>
inline T nd_iterator_init(T start) { return start; }
template<typename T, typename U, typename W, typename... Args>
inline T nd_iterator_init(T start, U &x, const W &X, Args &&... tuple) {
    start = nd_iterator_init(start, utils::forward<Args>(tuple)...);
    x = start % X;
    return start / X;
}

inline bool nd_iterator_step() { return true; }
template<typename U, typename W, typename... Args>
inline bool nd_iterator_step(U &x, const W &X, Args &&... tuple) {
    if (nd_iterator_step(utils::forward<Args>(tuple)...) ) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template<typename U, typename W, typename Y>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X)
{
    U max_jump = end - cur;
    U dim_jump = X - x;
    if (dim_jump <= max_jump) {
        x = 0;
        cur += dim_jump;
        return true;
    } else {
        cur += max_jump;
        x += max_jump;
        return false;
    }
}
template<typename U, typename W, typename Y, typename... Args>
inline bool nd_iterator_jump(U &cur, const U end, W &x, const Y &X,
        Args &&... tuple)
{
    if (nd_iterator_jump(cur, end, utils::forward<Args>(tuple)...)) {
        x = (x + 1) % X;
        return x == 0;
    }
    return false;
}

template <typename T>
inline T pick(size_t i, const T &x0) { return x0; }
template <typename T, typename ...Args>
inline T pick(size_t i, const T &x0, Args &&... args) {
    return i == 0 ? x0 : pick(i - 1, utils::forward<Args>(args)...);
}

template <typename T>
T pick_by_prop_kind(prop_kind_t prop_kind, const T &val_fwd_inference,
        const T &val_fwd_training, const T &val_bwd_d, const T &val_bwd_w) {
    switch (prop_kind) {
    case prop_kind::forward_inference: return val_fwd_inference;
    case prop_kind::forward_training: return val_fwd_training;
    case prop_kind::backward_data: return val_bwd_d;
    case prop_kind::backward_weights: return val_bwd_w;
    default: assert(!"unsupported prop_kind");
    }
    return T();
}

template <typename T>
T pick_by_prop_kind(prop_kind_t prop_kind,
        const T &val_fwd, const T &val_bwd_d, const T &val_bwd_w)
{ return pick_by_prop_kind(prop_kind, val_fwd, val_fwd, val_bwd_d, val_bwd_w); }

template <typename Telem, size_t Tdims>
struct array_offset_calculator {
    template <typename... Targs>
    array_offset_calculator(Telem *base, Targs... Fargs) : _dims{ Fargs... }
    {
        _base_ptr = base;
    }
    template <typename... Targs>
    inline Telem &operator()(Targs... Fargs)
    {
        return *(_base_ptr + _offset(1, Fargs...));
    }

private:
    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t element)
    {
        return element;
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element)
    {
        return element + (_dims[dimension] * theta);
    }

    template <typename... Targs>
    inline size_t _offset(size_t const dimension, size_t theta, size_t element,
            Targs... Fargs)
    {
        size_t t_prime = element + (_dims[dimension] * theta);
        return _offset(dimension + 1, t_prime, Fargs...);
    }

    Telem *_base_ptr;
    const int _dims[Tdims];
};

}

int32_t fetch_and_add(int32_t *dst, int32_t val);
inline void yield_thread() {}

// Reads an environment variable 'name' and stores its string value in the
// 'buffer' of 'buffer_size' bytes on success.
//
// - Returns the length of the environment variable string value (excluding
// the terminating 0) if it is set and its contents (including the terminating
// 0) can be stored in the 'buffer' without truncation.
//
// - Returns negated length of environment variable string value and writes
// "\0" to the buffer (if it is not NULL) if the 'buffer_size' is to small to
// store the value (including the terminating 0) without truncation.
//
// - Returns 0 and writes "\0" to the buffer (if not NULL) if the environment
// variable is not set.
//
// - Returns INT_MIN if the 'name' is NULL.
//
// - Returns INT_MIN if the 'buffer_size' is negative.
//
// - Returns INT_MIN if the 'buffer' is NULL and 'buffer_size' is greater than
// zero. Passing NULL 'buffer' with 'buffer_size' set to 0 can be used to
// retrieve the length of the environment variable value string.
//
int getenv(const char *name, char *buffer, int buffer_size);
// Reads an integer from the environment
int getenv_int(const char *name, int default_value = 0);
bool jit_dump_enabled();
FILE *fopen(const char *filename, const char *mode);

constexpr int msan_enabled = MSAN_ENABLED;
inline void msan_unpoison(void *ptr, size_t size) {
#if MSAN_ENABLED
    __msan_unpoison(ptr, size);
#endif
}

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
