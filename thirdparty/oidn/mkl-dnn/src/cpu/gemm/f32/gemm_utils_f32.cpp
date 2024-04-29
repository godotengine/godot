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
#include <cmath>

#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "gemm_utils_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
namespace gemm_utils {
#define BM_NOCOPY_AVX 64
#define BN_NOCOPY_AVX 48
#define BK_NOCOPY_AVX 384
#define BN_LARGE_NOCOPY_AVX 192
#define BM_SMALL_NOCOPY_AVX 16
#define BN_SMALL_NOCOPY_AVX 1
#define BK_SMALL_NOCOPY_AVX 4
// Determine number of threads for each dimension of a 3-D partitioning
// algorithm based on input parameters
// m/n/k - First/second/third parameter for GEMM
// nthrs - total available number of threads
// nthrs_m/nthrs_n/nthrs_k - number of threads to use in each dimension
// BM/BN/BK - blocking values
void calc_nthr_nocopy_avx(int m, int n, int k,
        int nthrs, int *nthrs_m, int *nthrs_n, int *nthrs_k, int *BM, int *BN,
        int *BK)
{
    int nthr, nthr_m, nthr_n, nthr_k;
    int MB, NB, KB;

    nthr = nthrs;
    nthr_m = (m + BM_NOCOPY_AVX - 1) / BM_NOCOPY_AVX;
    nthr_n = (n + BN_NOCOPY_AVX - 1) / BN_NOCOPY_AVX;
    nthr_k = 1;

    // Partition along K dimension
    //  - if threading allows having barriers (e.g. OMP)
    //  - if there is not enough parallelism along M or N
    if (mkldnn_thr_syncable()) {
        int nthr_other = nthr_k = 1;
        while ((nthr_m * nthr_n * nthr_other < nthr)
                && (k / (nthr_other + 1) > BK_NOCOPY_AVX)) {
            nthr_other++;
            if ((nthr / nthr_other) * nthr_other > 0.9 * nthr)
                nthr_k = nthr_other;
        }
    }
    nthr /= nthr_k;

    if (nthr_m == 1)
        nthr_n = nthr;
    if (nthr_n == 1)
        nthr_m = nthr;

    // Simple partition reduction
    while (nthr_m * nthr_n > nthr)
        if (nthr_m > nthr_n)
            nthr_m--;
        else
            nthr_n--;
    while (nthr_m * nthr_n < nthr)
        if (nthr_m < nthr_n)
            nthr_m++;
        else
            nthr_n++;

    if ((nthr_m * nthr_n > nthr) && (nthr_m > 1) && (nthr_n > 1)) {

        if (nthr_m <= nthr_n) {
            nthr_m = (int)sqrt((double)nthr);
            if (nthr_m > (m + BM_SMALL_NOCOPY_AVX - 1) / BM_SMALL_NOCOPY_AVX)
                nthr_m = (m + BM_SMALL_NOCOPY_AVX - 1) / BM_SMALL_NOCOPY_AVX;
            nthr_n = nthr / nthr_m;

            while ((nthr_m > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_m--;
                nthr_n = nthr / nthr_m;
            }
        } else {
            nthr_n = (int)sqrt((double)nthr);
            if (nthr_n > (n + BN_SMALL_NOCOPY_AVX - 1) / BN_SMALL_NOCOPY_AVX)
                nthr_n = (n + BN_SMALL_NOCOPY_AVX - 1) / BN_SMALL_NOCOPY_AVX;
            nthr_m = nthr / nthr_n;

            while ((nthr_n > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_n--;
                nthr_m = nthr / nthr_n;
            }
        }
    }

    MB = (m + nthr_m - 1) / nthr_m + BM_SMALL_NOCOPY_AVX - 1;
    MB -= MB % BM_SMALL_NOCOPY_AVX;
    NB = (n + nthr_n - 1) / nthr_n + BN_SMALL_NOCOPY_AVX - 1;
    NB -= NB % BN_SMALL_NOCOPY_AVX;
    KB = (k + nthr_k - 1) / nthr_k + BK_SMALL_NOCOPY_AVX - 1;
    KB -= KB % BK_SMALL_NOCOPY_AVX;

    if (MB * nthr_m > m)
        nthr_m = (m + MB - 1) / MB;
    if (NB * nthr_n > n)
        nthr_n = (n + NB - 1) / NB;
    if (KB * nthr_k > k)
        nthr_k = (k + KB - 1) / KB;

    *nthrs_m = nthr_m;
    *nthrs_n = nthr_n;
    *nthrs_k = nthr_k;

    *BM = MB;
    *BN = NB;
    *BK = KB;
}
#undef BM_NOCOPY_AVX
#undef BN_NOCOPY_AVX
#undef BK_NOCOPY_AVX
#undef BN_LARGE_NOCOPY_AVX
#undef BM_SMALL_NOCOPY_AVX
#undef BN_SMALL_NOCOPY_AVX
#undef BK_SMALL_NOCOPY_AVX

#define BM_NOCOPY_AVX512_COMMON 32
#define BN_NOCOPY_AVX512_COMMON 64
#define BK_NOCOPY_AVX512_COMMON 192
#define BN_LARGE_NOCOPY_AVX512_COMMON 192
#define BM_SMALL_NOCOPY_AVX512_COMMON 16
#define BN_SMALL_NOCOPY_AVX512_COMMON 1
#define BK_SMALL_NOCOPY_AVX512_COMMON 4
// Determine number of threads for each dimension of a 3-D partitioning
// algorithm based on input parameters
// m/n/k - First/second/third parameter for GEMM
// nthrs - total available number of threads
// nthrs_m/nthrs_n/nthrs_k - number of threads to use in each dimension
// BM/BN/BK - blocking values
void calc_nthr_nocopy_avx512_common(int m,
        int n, int k, int nthrs, int *nthrs_m, int *nthrs_n, int *nthrs_k,
        int *BM, int *BN, int *BK)
{
    int nthr, nthr_m, nthr_n, nthr_k = 1;
    int MB, NB, KB;
    nthr = nthrs;

    int counter = 0;
    float ratio_float = 1.;
    int ratio = 1;
    nthr = nthrs;
    int nthr_m_gt_n;

    // Partition along K dimension
    //  - if threading allows having barriers (e.g. OMP)
    //  - if there is not enough parallelism along M or N
    if (mkldnn_thr_syncable()) {
        if (n <= 2 * BN_NOCOPY_AVX512_COMMON &&
                m <= 2 * BM_NOCOPY_AVX512_COMMON * nthr) {
            nthr_k = k / BK_NOCOPY_AVX512_COMMON;
            if (nthr_k > nthr / 4)
                nthr_k = nthr / 4;
            if (nthr_k < 1)
                nthr_k = 1;

            while ((nthr_k > 1) && (nthr % nthr_k)) {
                nthr_k--;
            }
            nthr /= nthr_k;
        } else {
            nthr_k = 1;
        }
    }
    nthr_m = (m + BM_NOCOPY_AVX512_COMMON - 1) / BM_NOCOPY_AVX512_COMMON;
    nthr_n = (n + BN_NOCOPY_AVX512_COMMON - 1) / BN_NOCOPY_AVX512_COMMON;

    if (nthr_m < 1)
        nthr_m = 1;
    if (nthr_n < 1)
        nthr_n = 1;

    nthr_m_gt_n = nthr_m > nthr_n ? 1 : 0;
    ratio_float = (float)nthr_m / nthr_n;

    if (nthr_m_gt_n)
        ratio = (int)ratio_float;
    else
        ratio = (int)(1. / ratio_float);

    // scale down nthr_m and nthr_n if they are too large
    while (nthr_m * nthr_n > 4 * nthr) {
        nthr_m /= 2;
        nthr_n /= 2;
    }

    if (nthr_m < 1)
        nthr_m = 1;
    if (nthr_n < 1)
        nthr_n = 1;

    // Simple partition reduction
    counter = 0;
    while (nthr_m * nthr_n > nthr) {
        if (nthr_m > nthr_n) {
            if (counter < ratio)
                nthr_m--;
            else {
                nthr_n--;
                counter = -1;
            }
        } else {
            if (counter < ratio)
                nthr_n--;
            else {
                nthr_m--;
                counter = -1;
            }
        }
        counter++;
    }

    // Simple partition increment
    counter = 0;
    while (nthr_m * nthr_n < 0.95 * nthr) {
        if (nthr_m > nthr_n) {
            if (counter < ratio)
                nthr_m++;
            else {
                nthr_n++;
                counter = -1;
            }
        } else {
            if (counter < ratio)
                nthr_n++;
            else {
                nthr_m++;
                counter = -1;
            }
        }
        counter++;
    }

    // if nothing works out, then this should work
    if ((nthr_m * nthr_n > nthr)) {

        if (nthr_m <= nthr_n) {
            nthr_m = (int)sqrt((double)nthr);
            if (nthr_m > (m + BM_SMALL_NOCOPY_AVX512_COMMON - 1)
                            / BM_SMALL_NOCOPY_AVX512_COMMON)
                nthr_m = (m + BM_SMALL_NOCOPY_AVX512_COMMON - 1)
                        / BM_SMALL_NOCOPY_AVX512_COMMON;
            nthr_n = nthr / nthr_m;

            while ((nthr_m > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_m--;
                nthr_n = nthr / nthr_m;
            }
        } else {
            nthr_n = (int)sqrt((double)nthr);
            if (nthr_n > (n + BN_SMALL_NOCOPY_AVX512_COMMON - 1)
                            / BN_SMALL_NOCOPY_AVX512_COMMON)
                nthr_n = (n + BN_SMALL_NOCOPY_AVX512_COMMON - 1)
                        / BN_SMALL_NOCOPY_AVX512_COMMON;
            nthr_m = nthr / nthr_n;

            while ((nthr_n > 1) && (nthr_m * nthr_n != nthr)) {
                nthr_n--;
                nthr_m = nthr / nthr_n;
            }
        }
    }

    MB = (m + nthr_m - 1) / nthr_m + BM_SMALL_NOCOPY_AVX512_COMMON - 1;
    MB -= MB % BM_SMALL_NOCOPY_AVX512_COMMON;
    NB = (n + nthr_n - 1) / nthr_n + BN_SMALL_NOCOPY_AVX512_COMMON - 1;
    NB -= NB % BN_SMALL_NOCOPY_AVX512_COMMON;
    KB = (k + nthr_k - 1) / nthr_k + BK_SMALL_NOCOPY_AVX512_COMMON - 1;
    KB -= KB % BK_SMALL_NOCOPY_AVX512_COMMON;

    if (MB * nthr_m > m)
        nthr_m = (m + MB - 1) / MB;
    if (NB * nthr_n > n)
        nthr_n = (n + NB - 1) / NB;
    if (KB * nthr_k > k)
        nthr_k = (k + KB - 1) / KB;

    *nthrs_m = nthr_m;
    *nthrs_n = nthr_n;
    *nthrs_k = nthr_k;

    *BM = MB;
    *BN = NB;
    *BK = KB;
}
#undef BM_NOCOPY_AVX512_COMMON
#undef BN_NOCOPY_AVX512_COMMON
#undef BK_NOCOPY_AVX512_COMMON
#undef BN_LARGE_NOCOPY_AVX512_COMMON
#undef BM_SMALL_NOCOPY_AVX512_COMMON
#undef BN_SMALL_NOCOPY_AVX512_COMMON
#undef BK_SMALL_NOCOPY_AVX512_COMMON

// Partition n values as equally as possible among nthr threads
// and set the offset (t_offset) and number of values (t_block) for ithr
// Assumption: 0 <= ithr < nthr
void partition_unit_diff(
        int ithr, int nthr, int n, int *t_offset, int *t_block)
{
    int band = n / nthr;
    if (band == 0)
        band = 1;
    int tail = n - band * nthr;
    if (tail < 0)
        tail = 0;

    if (ithr < tail) {
        band++;
        *t_offset = band * ithr;
        *t_block = band;
    } else {
        *t_offset = band * ithr + tail;
        *t_block = band;
    }

    if (*t_offset >= n) {
        *t_offset = 0;
        *t_block = 0;
    }

    if (*t_offset + *t_block > n) {
        *t_block = n - *t_offset;
    }
}

// Sum the m*n values from p_src into p_dst, assuming the two-dimensional
// arrays have leading dimensions ld_src and ld_dst, respectively
template<typename data_t>
void sum_two_matrices(int m, int n,
        data_t * __restrict p_src, dim_t ld_src,
        data_t * __restrict p_dst, dim_t ld_dst)
{
    int i, j;
    for (j = 0; j < n; j++) {
        for (i = 0; i < m; i++) {
            p_dst[i + j * ld_dst] += p_src[i + j * ld_src];
        }
    }
}

template
void sum_two_matrices<float>(int m, int n,
        float * __restrict p_src, dim_t ld_src,
        float * __restrict p_dst, dim_t ld_dst);

template
void sum_two_matrices<double>(int m, int n,
        double * __restrict p_src, dim_t ld_src,
        double * __restrict p_dst, dim_t ld_dst);
}
}
}
}
