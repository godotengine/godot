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

#include "c_types_map.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "cpu_batch_normalization_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {
namespace bnorm_utils {

void cache_balance(size_t working_set_size, dim_t C_blks,
        dim_t &C_blks_per_iter, int64_t &iters) {
    int nthrs = mkldnn_get_max_threads();
    int l3_size = get_cache_size(3, true) * nthrs / 2;

    C_blks_per_iter = l3_size / working_set_size;

    if (C_blks_per_iter == 0)
        C_blks_per_iter = 1;
    if (C_blks_per_iter > C_blks)
        C_blks_per_iter = C_blks;

    iters = (C_blks + C_blks_per_iter - 1) / C_blks_per_iter;
}

bool thread_balance(bool do_blocking, bool spatial_thr_allowed, int ithr,
        int nthr, dim_t N, dim_t C_blks, dim_t SP, int &C_ithr, int &C_nthr,
        dim_t &C_blk_s, dim_t &C_blk_e, int &N_ithr, int &N_nthr, dim_t &N_s,
        dim_t &N_e, int &S_ithr, int &S_nthr, dim_t &S_s, dim_t &S_e) {
    if (nthr <= C_blks || !mkldnn_thr_syncable()) {
        C_ithr = ithr; C_nthr = nthr;
        N_ithr = 0; N_nthr = 1;
        S_ithr = 0; S_nthr = 1;
        N_s = 0; N_e = N; S_s = 0; S_e = SP;
        balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
    } else {
        if (do_blocking) {
            N_nthr = (int)nstl::min<dim_t>(N, nthr);
            C_nthr = (int)nstl::min<dim_t>(C_blks, nthr / N_nthr);
            S_nthr = (int)nstl::min<dim_t>(SP, nthr / (C_nthr * N_nthr));
        } else {
            C_nthr = (int)math::gcd((dim_t)nthr, C_blks);
            N_nthr = (int)nstl::min<dim_t>(N, nthr / C_nthr);
            S_nthr = (int)nstl::min<dim_t>(SP, nthr / (C_nthr * N_nthr));
        }

        if (!spatial_thr_allowed)
            S_nthr = 1;

        if (S_nthr < 1) S_nthr = 1;
        if (ithr < C_nthr * N_nthr * S_nthr) {
            N_ithr = (ithr / S_nthr) % N_nthr ;
            C_ithr = ithr / (N_nthr * S_nthr);
            S_ithr = ithr % S_nthr;
            balance211(C_blks, C_nthr, C_ithr, C_blk_s, C_blk_e);
            balance211(N, N_nthr, N_ithr, N_s, N_e);
            balance211(SP, S_nthr, S_ithr, S_s, S_e);
        } else {
            S_ithr = N_ithr = C_ithr = -ithr;
            S_s = S_e = N_s = N_e = C_blk_s = C_blk_e = -1;
        }
    }

    // spatial_thr_allowed is meant to help maintain
    // consistent decisions about spatial threading
    // between mutiple invocations of this routine.
    // It is caller's responsibility to check the
    // return value and pass it as a flag to the
    // next call if needed.
    if (S_nthr == 1)
        spatial_thr_allowed = false;

    return spatial_thr_allowed;
}

bool is_spatial_thr(const batch_normalization_pd_t *bdesc, int simd_w,
        int data_size) {
    if (!mkldnn_thr_syncable()) return false;

    dim_t nthr = mkldnn_get_max_threads();
    dim_t SP = bdesc->W() * bdesc->D() * bdesc->H();
    dim_t C_PADDED = memory_desc_wrapper(bdesc->src_md())
        .padded_dims()[1];
    assert(C_PADDED % simd_w == 0);

    size_t data = bdesc->MB() * C_PADDED * SP * data_size;
    size_t l3_size_ = get_cache_size(3, true) * nthr / 2;
    bool do_blocking = (data >= l3_size_ / 2 && l3_size_ > 0);
    dim_t C_blks_per_iter{ 1 }, iters{ 1 };
    dim_t C_blks = C_PADDED / simd_w;

    if (do_blocking) {
        int num_tensors = bdesc->is_fwd() ? 1 : 2;
        size_t working_set_size
            = (bdesc->MB() * SP * simd_w * data_size) * num_tensors;
        cache_balance(working_set_size, C_blks, C_blks_per_iter, iters);
    }

    // Spatial threading decision made in this function shall be consistent
    // with thread_balance() behavior.
    C_blks = do_blocking ? C_blks_per_iter : C_blks;

    if (nthr <= C_blks) return false;

    dim_t S_nthr = 1;
    if (do_blocking) {
        dim_t N_nthr = nstl::min(bdesc->MB(), nthr);
        dim_t C_nthr = nstl::min(C_blks, nthr / N_nthr);
        S_nthr = nstl::min(SP, nthr / (C_nthr * N_nthr));
    } else {
        dim_t C_nthr = math::gcd(nthr, C_blks);
        dim_t N_nthr = nstl::min(bdesc->MB(), nthr / C_nthr);
        S_nthr = nstl::min(SP, nthr / (C_nthr * N_nthr));
    }

    return S_nthr > 1;
}

}
}
}
}
