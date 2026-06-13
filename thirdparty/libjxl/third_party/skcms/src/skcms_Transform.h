/*
 * Copyright 2018 Google LLC
 *
 * Use of this source code is governed by a BSD-style license that can be
 * found in the LICENSE file.
 */

#pragma once

#include <stddef.h>
#include <stdint.h>

// skcms_Transform.h contains skcms implementation details.
// Please don't use this header from outside the skcms repo.

namespace skcms_private {

/** All transform ops */

#define SKCMS_WORK_OPS(M) \
    M(load_a8)            \
    M(load_g8)            \
    M(load_ga88)          \
    M(load_4444)          \
    M(load_565)           \
    M(load_888)           \
    M(load_8888)          \
    M(load_1010102)       \
    M(load_101010x_XR)    \
    M(load_10101010_XR)   \
    M(load_161616LE)      \
    M(load_16161616LE)    \
    M(load_161616BE)      \
    M(load_16161616BE)    \
    M(load_hhh)           \
    M(load_hhhh)          \
    M(load_fff)           \
    M(load_ffff)          \
                          \
    M(swap_rb)            \
    M(clamp)              \
    M(invert)             \
    M(force_opaque)       \
    M(premul)             \
    M(unpremul)           \
    M(matrix_3x3)         \
    M(matrix_3x4)         \
                          \
    M(lab_to_xyz)         \
    M(xyz_to_lab)         \
                          \
    M(gamma_r)            \
    M(gamma_g)            \
    M(gamma_b)            \
    M(gamma_a)            \
    M(gamma_rgb)          \
                          \
    M(tf_r)               \
    M(tf_g)               \
    M(tf_b)               \
    M(tf_a)               \
    M(tf_rgb)             \
                          \
    M(pq_r)               \
    M(pq_g)               \
    M(pq_b)               \
    M(pq_a)               \
    M(pq_rgb)             \
                          \
    M(hlg_r)              \
    M(hlg_g)              \
    M(hlg_b)              \
    M(hlg_a)              \
    M(hlg_rgb)            \
                          \
    M(hlginv_r)           \
    M(hlginv_g)           \
    M(hlginv_b)           \
    M(hlginv_a)           \
    M(hlginv_rgb)         \
                          \
    M(table_r)            \
    M(table_g)            \
    M(table_b)            \
    M(table_a)            \
                          \
    M(clut_A2B)           \
    M(clut_B2A)

#define SKCMS_STORE_OPS(M) \
    M(store_a8)            \
    M(store_g8)            \
    M(store_ga88)          \
    M(store_4444)          \
    M(store_565)           \
    M(store_888)           \
    M(store_8888)          \
    M(store_1010102)       \
    M(store_161616LE)      \
    M(store_16161616LE)    \
    M(store_161616BE)      \
    M(store_16161616BE)    \
    M(store_101010x_XR)    \
    M(store_hhh)           \
    M(store_hhhh)          \
    M(store_fff)           \
    M(store_ffff)

enum class Op : int {
#define M(op) op,
    SKCMS_WORK_OPS(M)
    SKCMS_STORE_OPS(M)
#undef M
};

/** Constants */

#if defined(__clang__) || defined(__GNUC__)
    static constexpr float INFINITY_ = __builtin_inff();
#else
    static const union {
        uint32_t bits;
        float    f;
    } inf_ = { 0x7f800000 };
    #define INFINITY_ inf_.f
#endif

/** Vector type */

#if defined(__clang__)
    template <int N, typename T> using Vec = T __attribute__((ext_vector_type(N)));
#elif defined(__GNUC__)
    // Unfortunately, GCC does not allow us to omit the struct. This will not compile:
    //   template <int N, typename T> using Vec = T __attribute__((vector_size(N*sizeof(T))));
    template <int N, typename T> struct VecHelper {
        typedef T __attribute__((vector_size(N * sizeof(T)))) V;
    };
    template <int N, typename T> using Vec = typename VecHelper<N, T>::V;
#endif

/** Interface */

namespace baseline {

void run_program(const Op* program, const void** contexts, ptrdiff_t programSize,
                 const char* src, char* dst, int n,
                 const size_t src_bpp, const size_t dst_bpp);

}
namespace hsw {

void run_program(const Op* program, const void** contexts, ptrdiff_t programSize,
                 const char* src, char* dst, int n,
                 const size_t src_bpp, const size_t dst_bpp);

}
namespace skx {

void run_program(const Op* program, const void** contexts, ptrdiff_t programSize,
                 const char* src, char* dst, int n,
                 const size_t src_bpp, const size_t dst_bpp);

}
}  // namespace skcms_private
