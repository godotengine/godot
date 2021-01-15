/*******************************************************************************
* Copyright 2018 Intel Corporation
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

#include <cstdint>

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "../f32/ref_gemm_f32.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename b_dt>
mkldnn_status_t ref_gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const b_dt *B, const int *LDB, const int8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co) {

    if (*M == 0 || *N == 0 || *K == 0)
        return mkldnn_success;

    bool OCisR = (*offsetc == 'R' || *offsetc == 'r');
    bool OCisC = (*offsetc == 'C' || *offsetc == 'c');
    bool AisN = (*transa == 'N' || *transa == 'n');
    bool BisN = (*transb == 'N' || *transb == 'n');

    int m = *M, n = *N, k = *K, lda = *LDA, ldb = *LDB, ldc = *LDC;
    size_t sizeA = AisN ? lda * k : lda * m;
    size_t sizeB = BisN ? ldb * n : ldb * k;
    size_t sizeC = ldc * n;

    double *dA = (double *)malloc(sizeA * sizeof(double), PAGE_4K);
    double *dB = (double *)malloc(sizeB * sizeof(double), PAGE_4K);
    double *dC = (double *)malloc(sizeC * sizeof(double), PAGE_4K);

    if (utils::any_null(dA, dB, dC)) {
        free(dA);
        free(dB);
        free(dC);
        return mkldnn_out_of_memory;
    }

    auto da_setter = [=] (int i, int j, double v) { dA[j * lda + i] = v; };
    auto db_setter = [=] (int i, int j, double v) { dB[j * ldb + i] = v; };

    auto ia_accessor = [=] (int i, int j) { return A[j * lda + i]; };
    auto ib_accessor = [=] (int i, int j) { return B[j * ldb + i]; };

    const int a_rows = AisN ? m : k;
    const int a_cols = AisN ? k : m;
    mkldnn::impl::parallel_nd(a_cols, a_rows, [&](int j, int i) {
        da_setter(i, j,
            static_cast<double>(ia_accessor(i, j)) + static_cast<double>(ao[0]));
    });

    const int b_rows = BisN ? k : n;
    const int b_cols = BisN ? n : k;
    mkldnn::impl::parallel_nd(b_cols, b_rows, [&](int j, int i) {
        db_setter(i, j,
            static_cast<double>(ib_accessor(i, j)) + static_cast<double>(bo[0]));
    });
    double one = 1.0, zero = 0.0;
    ref_gemm<double>(transa, transb, M, N, K, &one, dA, LDA, dB, LDB, &zero,
        dC, LDC, nullptr);

    auto i2d = [=] (int32_t v) { return static_cast<double>(v); };
    auto f2d = [=] (float v)   { return static_cast<double>(v); };

    mkldnn::impl::parallel_nd(n, m, [&] (int j, int i) {
        double coffset = OCisR ? i2d(co[j]) : OCisC ? i2d(co[i]) : i2d(co[0]);
        double val = ((*beta == 0.0f) ? 0.0 : f2d(*beta) * i2d(C[i + j * ldc]))
            + f2d(*alpha) * dC[i + j * ldc] + coffset;
        C[i + j * ldc] = math::out_round<int32_t>(math::saturate<int32_t>(val));
    });

    free(dA);
    free(dB);
    free(dC);
    return mkldnn_success;
}

template mkldnn_status_t ref_gemm_s8x8s32<uint8_t>(
        const char *transa, const char *transb, const char *offsetc,
        const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const uint8_t *B, const int *LDB, const int8_t *bo,
        const float *beta, int32_t *C, const int *LDC, const int32_t *co);

template mkldnn_status_t ref_gemm_s8x8s32<int8_t>(
        const char *transa, const char *transb, const char *offsetc,
        const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const int8_t *B, const int *LDB, const int8_t *bo,
        const float *beta, int32_t *C, const int *LDC, const int32_t *co);

}
}
}
