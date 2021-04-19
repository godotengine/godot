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

#include "gemv.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

int gemm_s8u8s32_jump_to_gemv_s8u8s32(blas_t *arg) {

    blas_t arg_gemv = *arg;

    if ((arg -> offsetc == FIX_OFFSET) && // Fix offset
        (arg -> ao == 0) &&
        (arg -> bo == 0) &&
        (arg -> co[0] == 0) &&
        (*(arg -> alpha) == 1.0f) &&
        ((*(arg -> beta) == 1.0f) || *(arg -> beta) == 0.0f)) {

        if (arg -> n == 1) {

            if (arg -> transa == 1) { // A transpose
                arg_gemv.n = arg -> k;
                arg_gemv.ldc = 1;
                arg_gemv.swap = 0;
                if (arg -> transb == 0) { // B non transpose
                    arg_gemv.ldb = 1;
                }
                // B transpose arg_gemv.ldb = arg -> ldb
                gemv_threading_driver(&arg_gemv);
                return 1;
            }
        }

        if (arg -> m == 1) {

            if (arg -> transb == 0) { // B non transpose
                arg_gemv.transa = 1;
                arg_gemv.m = arg -> n;
                arg_gemv.n = arg -> k;
                arg_gemv.a = (int8_t *) arg -> b;
                arg_gemv.lda = arg -> ldb;
                arg_gemv.b = (uint8_t *) arg -> a;
                arg_gemv.swap = 1;
                if (arg -> transa == 0) { // A non transpose
                    arg_gemv.ldb = arg -> lda;
                }
                else { // A transpose
                    arg_gemv.ldb = 1;
                }
                gemv_threading_driver(&arg_gemv);
                return 1;
            }
        }
    }

    return 0;
}


int gemv_kernel_driver(blas_t *arg) {

    dim_t m = arg -> m;
    dim_t n = arg -> n;
    uint8_t *a = (uint8_t *) arg -> a;
    dim_t lda = arg -> lda;
    int8_t *b = (int8_t *) arg -> b;
    float beta = *(arg -> beta);

    if (arg -> swap) {
        arg -> gemv_u8s8s32_kernel(m, n, 1.0f, a, lda, b, beta, arg -> c);
    }
    else {
        arg -> gemv_s8u8s32_kernel(arg -> m, arg -> n, 1.0f, arg -> a,
                                   arg -> lda, arg -> b, *(arg -> beta), arg -> c);
    }

    return 0;
}

int gemv_threading_driver(blas_t *arg) {

    dim_t nthr_m, nthr_n = 1;
    dim_t MB, NB, UM = 16, UN = 64;
    dim_t BLOCKM = 192, BLOCKN = 3072;
    int status;
    dim_t i;

    dim_t nthr = (mkldnn_in_parallel()) ? 1 : mkldnn_get_max_threads();

    uint8_t *new_x = NULL;
    int32_t *tmp_y = NULL, *new_y = NULL;

    dim_t m = arg -> m, n = arg -> n;

    blas_t arg_seq = *arg;
    float zero = 0.0f;

    nthr_m = std::min(std::max(m / BLOCKM, (dim_t) 1), nthr);
    MB = m / nthr_m;
    MB = (((MB / UM) * UM) == MB) ? MB : (MB / UM) * UM + UM;
    nthr_m = (((m / MB) * MB) == m) ? m / MB : m / MB + 1;
    nthr_m = std::min(std::max(nthr_m, (dim_t) 1), nthr);

    while ((nthr_m * (nthr_n + 1) <= nthr) && ((n / (nthr_n + 1)) >= BLOCKN)) {
        nthr_n++;
    }

    NB = n / nthr_n;
    NB = (((NB / UN) * UN) == NB) ? NB : (NB / UN) * UN + UN;
    nthr_n = (((n / NB) * NB) == n) ? n / NB : n / NB + 1;
    nthr_n = std::min(std::max(nthr_n, (dim_t) 1), nthr / nthr_m);

    nthr = nthr_m * nthr_n;

    if (arg -> ldb != 1) {
        new_x = (uint8_t *)malloc(n, 64);
        if (new_x == NULL)
            return 1;
        for (i = 0; i < n; i++) {
            new_x[i] = (arg -> b)[i * arg -> ldb];
        }
        arg_seq.b = new_x;
        arg_seq.ldb = 1;
    }
    else new_x = (uint8_t *) arg -> b;

    if (arg -> ldc != 1) {
        new_y = (int32_t *) malloc(nthr_m * PADD_BYTESIZE_ONPAGE(MB, sizeof(int32_t)), 64);
        if (new_y == NULL) {
            if (arg -> ldb != 1) {
                free(new_x);
            }
            return 1;
        }
    }

    // GEMV computation
    if (nthr == 1) {

        if (arg -> ldc != 1) {
            if (*(arg -> beta) != 0.0f) {
                for (i = 0; i < m; i++) {
                    new_y[i] = arg -> c[i * arg -> ldc];
                }
            }
        }

        status = gemv_kernel_driver(&arg_seq);

        if (arg -> ldc != 1) {
            for (i = 0; i < m; i++) {
                arg -> c[i * arg -> ldc] = new_y[i];
            }
        }

        if (arg -> ldb != 1) {
            free(new_x);
        }
        if (arg -> ldc != 1) {
            free(new_y);
        }
        return status;
    }

    if (nthr_n > 1) {
        tmp_y = (int32_t *) malloc((nthr_n - 1) * PADD_BYTESIZE_ONPAGE(m, sizeof(int32_t)), PAGESIZE);
        if (tmp_y == NULL) {
            if (arg -> ldb != 1) {
                free(new_x);
            }
            return 1;
        }
    }

    parallel_nd((int) nthr, [&](const dim_t ithr) {

            dim_t m_from, m_to, myM;
            dim_t n_from, n_to, myN;

            dim_t n_id, m_id;
            dim_t loc_incy = 1;
            int32_t *loc_y;

            blas_t arg_loc = arg_seq;
            int j;

            m_id = ithr / nthr_n;
            n_id = ithr % nthr_n;

            m_from = MB * m_id;
            m_to = MB * (m_id + 1);
            if ((m_to > m) || (m_id == nthr_m - 1))
                m_to = m;

            myM = m_to - m_from;

            n_from = NB * n_id;
            n_to = NB * (n_id + 1);
            if ((n_to > n) || (n_id == nthr_n - 1))
                n_to = n;

            myN = n_to - n_from;

            if (n_id != 0) {
                arg_loc.beta = &zero;
                loc_y = tmp_y + (NEXT_THR_STRIDE(m, sizeof(int32_t))) * (n_id - 1) + m_from;
            }
            else {
                if (arg -> ldc == 1) {
                    loc_y = arg_seq.c + m_from;
                }
                else {
                    // need to copy the block of c in new_y
                    loc_y = new_y + m_id * NEXT_THR_STRIDE(MB, sizeof(int32_t));
                    if (*(arg -> beta) != 0.0f) {
                        for (j = 0; j < myM; j++) {
                            loc_y[j] = arg -> c[(m_from + j) * arg -> ldc];
                        }
                    }
                }
            }

            arg_loc.m = myM;
            arg_loc.n = myN;
            arg_loc.a = arg_seq.a + m_from * arg_seq.lda + n_from;
            arg_loc.b = arg_seq.b + n_from;
            arg_loc.c = loc_y;
            arg_loc.ldc = loc_incy;

            gemv_kernel_driver(&arg_loc);

            if ((n_id == 0) && (arg -> ldc != 1)) {
                for (j = 0; j < myM; j++) {
                    arg -> c[(m_from + j) * arg -> ldc] = loc_y[j];
                }
            }

        });

    if (nthr_n > 1) {
        parallel_nd((int) nthr_m, [&](const dim_t ithr) {

                dim_t j, j_from, j_to, ii;
                int32_t acc;

                j_from = MB * ithr;
                j_to = MB * (ithr + 1);
                if ((j_to > m) || (ithr == nthr - 1))
                    j_to = m;

                for (j = j_from; j < j_to; j++) {
                    acc = 0;
                    for (ii = 0; ii < nthr_n - 1; ii++) {
                        acc += tmp_y[ii * NEXT_THR_STRIDE(m, sizeof(int32_t)) + j];
                    }
                    (arg -> c)[j * arg -> ldc] += acc;
                }
            });
        free(tmp_y);
    }

    if (arg -> ldb != 1) {
        free(new_x);
    }

    if (arg -> ldc != 1) {
        free(new_y);
    }

    return 0;
}

}
}
}
