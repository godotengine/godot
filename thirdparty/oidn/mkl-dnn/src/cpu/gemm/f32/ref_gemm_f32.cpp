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

#include "mkldnn_types.h"

#include "mkldnn_thread.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "gemm_utils_f32.hpp"
#include "ref_gemm_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace gemm_utils;

namespace {

template <typename data_t>
void copy_A(
        bool isTransA, int K, const data_t *A, const dim_t lda, data_t *ws) {
    for (int k = 0; k < K; k++) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < unroll_factor<data_t>::m; i++) {
            ws[i] = isTransA ? A[i * lda + k] : A[i + k * lda];
        }
        ws += unroll_factor<data_t>::m;
    }
}

template <typename data_t, bool isTransA, bool isTransB>
void kernel_mxn(int K, const data_t *A, const dim_t lda,
        const data_t *B, const dim_t ldb, data_t *C, const dim_t ldc,
        const data_t alpha, const data_t beta) {
    data_t c[unroll_factor<data_t>::m * unroll_factor<data_t>::n] =
        { static_cast<data_t>(0.) };
    for (int k = 0; k < K; k++) {
        for (int j = 0; j < unroll_factor<data_t>::n; j++) {
            data_t b = isTransB ? B[j + k * ldb] : B[k + j * ldb];
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < unroll_factor<data_t>::m; i++) {
                data_t a = isTransA ? A[i * lda + k] : A[i + lda * k];
                c[i + unroll_factor<data_t>::m * j] += a * b;
            }
        }
    }
    for (int j = 0; j < unroll_factor<data_t>::n; j++) {
        PRAGMA_OMP_SIMD()
        for (int i = 0; i < unroll_factor<data_t>::m; i++) {
            C[i + j * ldc] = (beta == static_cast<data_t>(0.))
            ? alpha * c[i + unroll_factor<data_t>::m * j]
            : alpha * c[i + unroll_factor<data_t>::m * j]
                + beta * C[i + j * ldc];
        }
    }
}

template <typename data_t, bool isTransA, bool isTransB>
void block_ker(const int M, const int N, const int K,
        const data_t *A, const dim_t lda, const data_t *B, const dim_t ldb,
        data_t *C, const dim_t ldc, const data_t alpha, const data_t beta,
        data_t *ws, bool do_copy) {
    int Nu = rnd_dn(N, unroll_factor<data_t>::n);
    int Mu = rnd_dn(M, unroll_factor<data_t>::m);
    for (int i = 0; i < Mu; i += unroll_factor<data_t>::m) {
        for (int j = 0; j < Nu; j += unroll_factor<data_t>::n) {
            const data_t *b = isTransB ? &B[j] : &B[j * ldb];
            const data_t *a = isTransA ? &A[i * lda] : &A[i];
            if (do_copy) {
                if (j == 0) {
                    copy_A<data_t>(isTransA, K, a, lda, ws);
                }
                kernel_mxn<data_t, false, isTransB>(
                        K, ws, unroll_factor<data_t>::m, b, ldb,
                        &C[i + j * ldc], ldc, alpha, beta);
            } else {
                kernel_mxn<data_t, isTransA, isTransB>(
                        K, a, lda, b, ldb, &C[i + j * ldc], ldc, alpha, beta);
            }
        }
    }
    // tail processing
    for (int i = 0; i < M; i++) {
        for (int j = Nu; j < N; j++) {
            data_t c = beta == static_cast<data_t>(0.)
                ? static_cast<data_t>(0.)
                : beta * C[i + j * ldc];
            for (int p = 0; p < K; p++) {
                data_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                data_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
    for (int i = Mu; i < M; i++) {
        for (int j = 0; j < Nu; j++) {
            data_t c = beta == static_cast<data_t>(0.)
                ? static_cast<data_t>(0.)
                : beta * C[i + j * ldc];
            for (int p = 0; p < K; p++) {
                data_t b = isTransB ? B[j + p * ldb] : B[p + j * ldb];
                data_t a = isTransA ? A[p + i * lda] : A[i + p * lda];
                c += alpha * a * b;
            }
            C[i + j * ldc] = c;
        }
    }
}

template <typename data_t, bool isTransA, bool isTransB>
void gemm_ithr(const int M, const int N, const int K, const data_t alpha,
        const data_t *A, const dim_t lda, const data_t *B, const dim_t ldb,
        const data_t beta, data_t *C, const dim_t ldc, bool do_copy,
        data_t *ws) {
    constexpr int BM = gemm_traits<data_t, isTransA, isTransB>::BM;
    constexpr int BN = gemm_traits<data_t, isTransA, isTransB>::BN;
    constexpr int BK = gemm_traits<data_t, isTransA, isTransB>::BK;

    const data_t *curA;
    const data_t *curB;
    data_t *curC;

    if ((M <= 0) || (N <= 0))
        return;

    if ((K <= 0) || (alpha == static_cast<data_t>(0))) {
        dim_t MN = N * M;
        if (beta == static_cast<data_t>(0.)) {
            for (dim_t j = 0; j < MN; j++)
                C[j] = static_cast<data_t>(0.);
        } else if (beta != static_cast<data_t>(1.)) {
            for (dim_t j = 0; j < MN; j++)
                C[j] *= beta;
        }
        return;
    }

    for (int Bk = 0; Bk < K; Bk += BK) {
        int kb = nstl::min(K - Bk, BK);
        for (int Bm = 0; Bm < M; Bm += BM) {
            int mb = nstl::min(M - Bm, BM);
            for (int Bn = 0; Bn < N; Bn += BN) {
                int nb = nstl::min(N - Bn, BN);
                curA = isTransA ? A + Bk + Bm * lda : A + Bm + Bk * lda;
                curB = isTransB ? B + Bn + Bk * ldb : B + Bk + Bn * ldb;
                curC = C + Bm + Bn * ldc;
                if (Bk == 0) {
                    block_ker<data_t, isTransA, isTransB>(mb, nb, kb, curA, lda,
                        curB, ldb, curC, ldc, alpha, beta, ws, do_copy);
                } else {
                    block_ker<data_t, isTransA, isTransB>(mb, nb, kb, curA, lda,
                        curB, ldb, curC, ldc, alpha, static_cast<data_t>(1.0),
                        ws, do_copy);
                }
            }
        }
    }
}

}

template <typename data_t>
mkldnn_status_t ref_gemm(
        const char *transa_, const char *transb_, const int *M_,
        const int *N_, const int *K_, const data_t *alpha_, const data_t *A,
        const int *lda_, const data_t *B, const int *ldb_, const data_t *beta_,
        data_t *C, const int *ldc_, const data_t *bias) {

    bool isTransA = (*transa_ == 'T' || *transa_ == 't');
    bool isTransB = (*transb_ == 'T' || *transb_ == 't');
    const int M = *M_, N = *N_, K = *K_;
    const dim_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const data_t alpha = *alpha_, beta = *beta_;

    int max_nthr = mkldnn_in_parallel() ? 1 : mkldnn_get_max_threads();
    int nthr_m, nthr_n, nthr_k;
    int MB, NB, KB;
    // thread balancing over M, N, K & size of blocking dimensions
    calc_nthr_nocopy_avx(
            M, N, K, max_nthr, &nthr_m, &nthr_n, &nthr_k, &MB, &NB, &KB);
    assert(IMPLICATION(!mkldnn_thr_syncable(), nthr_k == 1));

    data_t *c_buffers = nullptr;
    data_t *ws_buffers = nullptr;
    if (nthr_k > 1) {
        c_buffers = (data_t *)malloc(nthr_m * nthr_n * (nthr_k - 1) * MB * NB
                * sizeof(data_t), PAGE_4K);
        if (!c_buffers) {
            nthr_k = 1;
            KB = K;
        }
    }

    bool do_copy = (NB / unroll_factor<data_t>::n > 3);
    const int nthr_mn = nthr_m * nthr_n;
    const int nthr = nthr_mn * nthr_k;
    const size_t ws_elems_per_thr = K * unroll_factor<data_t>::m;
    const size_t ws_size_per_thr
            = rnd_up(ws_elems_per_thr * sizeof(data_t), PAGE_4K);
    if (do_copy) {
        ws_buffers = (data_t*)malloc(nthr * ws_size_per_thr, PAGE_4K);
        if (!ws_buffers)
            do_copy = false;
    }

    auto get_thr_block = [&](int &from, int &to, int &myN, int NB, int N,
                             int ithr) {
        from = NB * (ithr);
        to = NB * (ithr + 1);
        if (to > N)
            to = N;
        myN = to - from;
    };

    parallel_nd(nthr, [&](const int ithr) {
        int ithr_mn = ithr % nthr_mn;
        int ithr_m = ithr_mn % nthr_m;
        int ithr_n = ithr_mn / nthr_m;
        int ithr_k = ithr / nthr_mn;

        int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

        data_t *ws = do_copy
                ? ws_buffers + ithr * ws_size_per_thr / sizeof(data_t)
                : nullptr;

        int m_from = 0, m_to = 0, myM = 0, n_from = 0, n_to = 0, myN = 0,
                k_from = 0, k_to = 0, myK = 0;

        get_thr_block(m_from, m_to, myM, MB, M, ithr_m);
        get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
        get_thr_block(k_from, k_to, myK, KB, K, ithr_k);

        if (myM > 0 && myN > 0) {
            data_t myBeta, *myC;
            dim_t ld;
            if (ithr_k == 0) {
                myC = &(C[m_from + n_from * ldc]);
                myBeta = beta;
                ld = ldc;
            } else {
                myC = c_buffers + (dim_t)MB * NB * (cbase + ithr_k - 1);
                myBeta = 0.0f;
                ld = MB;
            }
            const data_t *myA = isTransA
                    ? &(A[k_from + m_from * lda])
                    : &(A[m_from + k_from * lda]);
            const data_t *myB = isTransB
                    ? &(B[n_from + k_from * ldb])
                    : &(B[k_from + n_from * ldb]);

            if (!isTransA) {
                if (!isTransB) {
                    gemm_ithr<data_t, false, false>(myM, myN, myK, alpha, myA,
                        lda, myB, ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<data_t, false, true>(myM, myN, myK, alpha, myA,
                        lda, myB, ldb, myBeta, myC, ld, do_copy, ws);
                }
            } else {
                if (!isTransB) {
                    gemm_ithr<data_t, true, false>(myM, myN, myK, alpha, myA,
                        lda, myB, ldb, myBeta, myC, ld, do_copy, ws);
                } else {
                    gemm_ithr<data_t, true, true>(myM, myN, myK, alpha, myA,
                        lda, myB, ldb, myBeta, myC, ld, do_copy, ws);
                }
            }
        }
    });

    if (nthr_k > 1) {
        parallel_nd(nthr, [&](const int ithr) {
            int ithr_mn = ithr % nthr_mn;
            int ithr_m = ithr_mn % nthr_m;
            int ithr_k = ithr / nthr_mn;
            int ithr_n = ithr_mn / nthr_m;

            int n_from = 0, n_to = 0, myN = 0;
            int m_from = 0, m_to = 0, myM = 0;

            int cbase = (ithr_m + nthr_m * ithr_n) * (nthr_k - 1);

            get_thr_block(n_from, n_to, myN, NB, N, ithr_n);
            get_thr_block(m_from, m_to, myM, MB, M, ithr_m);

            // sum matrices partitioned along K dimension
            int offset = 0, block = 0;
            gemm_utils::partition_unit_diff(ithr_k, nthr_k, myN, &offset,
                    &block);
            for (int ik = 1; ik < nthr_k; ++ik) {
                data_t *myC = c_buffers
                            + MB * ((dim_t)NB * (cbase + ik - 1) + offset);

                gemm_utils::sum_two_matrices(myM, block, myC, MB,
                        &C[m_from + (n_from + offset) * ldc], ldc);
            }
        });
    }

    if (bias) {
        parallel_nd(N, M, [&](int i, int j) {
            C[i*ldc + j] += bias[j];
        });
    }

    free(ws_buffers);
    free(c_buffers);

    return mkldnn_success;
}

template mkldnn_status_t ref_gemm<float>(
        const char *transa_, const char *transb_,
        const int *M_, const int *N_, const int *K_, const float *alpha_,
        const float *A, const int *lda_, const float *B, const int *ldb_,
        const float *beta_, float *C, const int *ldc_, const float *bias);

template mkldnn_status_t ref_gemm<double>(
        const char *transa_, const char *transb_,
        const int *M_, const int *N_, const int *K_, const double *alpha_,
        const double *A, const int *lda_, const double *B, const int *ldb_,
        const double *beta_, double *C, const int *ldc_, const double *bias);
}
}
}
