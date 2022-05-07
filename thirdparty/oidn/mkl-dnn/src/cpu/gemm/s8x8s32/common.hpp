/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef COMMON_H
#define COMMON_H

#define GEMM_CODE_SIZE          (4096L * 32)

#define AVX512_UNROLL_M                   48
#define AVX512_UNROLL_N                    8
#define AVX512_UNROLL_K                    1
#define AVX512_BM                       9984
#define AVX512_BN                        384
#define AVX512_BK                        768
#define AVX512_BK_VNNI                  1536
#define AVX512_BK_TRADITIONAL            384
#define AVX512_BLOCKING_SMALL_K           48
#define AVX512_BN_SMALL_K                 24


#define PAGESIZE 4096

#define PADD_BYTESIZE_ONPAGE(x, size) (((x) * (size) + PAGESIZE - 1) / PAGESIZE) * PAGESIZE
#define NEXT_THR_STRIDE(x, size) (PADD_BYTESIZE_ONPAGE(x, size)) / size

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

enum {
    PARTITION_1D_ROW,
    PARTITION_1D_COL,
    PARTITION_2D_COL_MAJOR,
    PARTITION_2D = PARTITION_2D_COL_MAJOR,
};

enum {
    COPY_NONE,
    COPY_A,
};

enum {
    NO_OFFSET,
    FIX_OFFSET,
    COL_OFFSET,
    ROW_OFFSET,
};

// Alias for any dimension related variable.
typedef long long int dim_t;

typedef struct {
    // Interface arguments.
    int transa, transb, offsetc;
    dim_t m, n, k;
    dim_t lda, ldb, ldc;
    const int8_t *a;
    const uint8_t *b;
    int32_t *c;
    const float *alpha, *beta;

    int8_t ao, bo;
    const int32_t *co;

    // Kernel parameters.
    dim_t um, un, uk, bm, bn, bk;
    dim_t bn_small_k, bk_traditional, blocking_small_k;

    int (*copyA)(const dim_t *m, const dim_t *n, const int8_t *a,
            const dim_t *lda, const int8_t *alpha, int8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    int (*copyB)(const dim_t *m, const dim_t *n, const uint8_t *a,
            const dim_t *lda, const uint8_t *alpha, uint8_t *b,
            const dim_t *dummy1, const dim_t *dummy2, int32_t *row_col_sum);

    int (*kernel)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_b)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_r)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    int (*kernel_b0_c)(const dim_t *m, const dim_t *n, const dim_t *k,
            const float *alpha, const int8_t *a, const uint8_t *b, int32_t *c,
            const dim_t ldc, const int32_t *col_offset,
            const int32_t *row_offset);

    // Gemv kernels
    void (*gemv_s8u8s32_kernel)(const dim_t, const dim_t, const float,
                                const int8_t*, const dim_t, const uint8_t*,
                                const float, int32_t*);

    void (*gemv_u8s8s32_kernel)(const dim_t, const dim_t, const float,
                                const uint8_t*, const dim_t, const int8_t*,
                                const float, int32_t*);

    // Gemv parameters
    int swap;

} blas_t;


class jit_avx512_core_u8_copy_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_an_kern);

    public:
        jit_avx512_core_u8_copy_an_kern();
};

class jit_avx512_core_u8_copy_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_at_kern);

    public:
        jit_avx512_core_u8_copy_at_kern();
};

class jit_avx512_core_u8_copy_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_bn_kern);

    public:
        jit_avx512_core_u8_copy_bn_kern();
};

class jit_avx512_core_u8_copy_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_bt_kern);

    public:
        jit_avx512_core_u8_copy_bt_kern();
};

class jit_avx512_core_u8_copy_sum_an_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_an_kern);

    public:
        jit_avx512_core_u8_copy_sum_an_kern();
};

class jit_avx512_core_u8_copy_sum_at_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_at_kern);

    public:
        jit_avx512_core_u8_copy_sum_at_kern();
};

class jit_avx512_core_u8_copy_sum_bn_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_bn_kern);

    public:
        jit_avx512_core_u8_copy_sum_bn_kern();
};

class jit_avx512_core_u8_copy_sum_bt_kern : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8_copy_sum_bt_kern);

    public:
        jit_avx512_core_u8_copy_sum_bt_kern();
};

}
}
}
#endif
