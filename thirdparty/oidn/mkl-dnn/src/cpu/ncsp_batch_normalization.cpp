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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"

#include "cpu_batch_normalization_utils.hpp"
#include "jit_generator.hpp"

#include "ncsp_batch_normalization.hpp"

// clang 6 and 7 generate incorrect code with OMP_SIMD in some particular cases
#if (defined __clang_major__) && (__clang_major__ >= 6)
#define SAFE_TO_USE_OMP_SIMD 0
#else
#define SAFE_TO_USE_OMP_SIMD 1
#endif

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace memory_tracking::names;

void ncsp_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const bool calculate_stats = !pd()->stats_is_src();
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto scaleshift = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SCALE_SHIFT);

    auto scratchpad = this->scratchpad(ctx);
    auto *ws_reduce = scratchpad.get<data_t>(key_bnorm_reduction);

    data_t *mean, *variance;
    if (!calculate_stats) {
        mean = const_cast<data_t *>(
                CTX_IN_MEM(const data_t *, MKLDNN_ARG_MEAN));
        variance = const_cast<data_t *>(
                CTX_IN_MEM(const data_t *, MKLDNN_ARG_VARIANCE));
    } else {
        if (save_stats) {
            mean = CTX_OUT_MEM(data_t *, MKLDNN_ARG_MEAN);
            variance = CTX_OUT_MEM(data_t *, MKLDNN_ARG_VARIANCE);
        } else {
            mean = scratchpad.get<data_t>(key_bnorm_tmp_mean);
            variance = scratchpad.get<data_t>(key_bnorm_tmp_var);
        }
    }

    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, MKLDNN_ARG_WORKSPACE);

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool with_relu = pd()->with_relu_post_op();
    auto maybe_post_op
            = [&](data_t res) { return (with_relu && res < 0) ? 0 : res; };
    const bool has_spatial = utils::one_of(pd()->ndims(), 4, 5);
    dim_t SP = (has_spatial) ? pd()->H() * pd()->W() * pd()->D() : 1;
    dim_t N = pd()->MB();
    dim_t C = pd()->C();

    int nthr = mkldnn_get_max_threads();
    size_t l3_size_ = get_cache_size(3, true) * nthr / 2;
    size_t data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);

    parallel(0, [&](const int ithr, const int nthr) {
        int C_ithr = 0, C_nthr = 0;
        int N_ithr = 0, N_nthr = 0;
        int S_ithr = 0, S_nthr = 0;

        dim_t C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        dim_t N_s = 0, N_e = 0;
        dim_t S_s = 0, S_e = 0;

        dim_t C_blks_per_iter = 1;
        int64_t iters = 1;

        if (do_blocking) {
            size_t working_set_size = N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int64_t last_iter_blks = C - (iters - 1) * C_blks_per_iter;
        bool spatial_thr_allowed
                = bnorm_utils::thread_balance(do_blocking, true, ithr, nthr, N,
                        C_blks_per_iter, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;
        for (int64_t it = 0; it < iters; ++it) {
            if (it == iters - 1 && iters > 1) {
                // On the last iteration the access pattern to ws_reduce
                // might change (due to re-balance on C). So sync the
                // threads if they are not synced by the algorithm.
                if (SP_N_nthr == 1 && mkldnn_thr_syncable())
                    mkldnn_thr_barrier();

                S_s = S_e = C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                        spatial_thr_allowed, ithr, nthr, N, last_iter_blks, SP,
                        C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr, N_s,
                        N_e, S_ithr, S_nthr, S_s, S_e);
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
                SP_N_ithr = N_ithr * S_nthr + S_ithr;
                SP_N_nthr = N_nthr * S_nthr;
            }
            size_t C_off = it * C_blks_per_iter;
            // On the last iteration the access pattern to ws_reduce
            // might change (due to re-balance on C). Since sync is not always
            // possible (in case of TBB) use different parts of ws for each
            // iteration if threads are not synced by the algorithm.
            size_t ws_iter_off = (mkldnn_thr_syncable() ? 0 : 1) * C_off;

            if (calculate_stats) {
                data_t *mean_blk = mean + C_off;
                data_t *variance_blk = variance + C_off;
                for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                    size_t off = (c + C_off) * SP;
                    data_t sum = 0;
                    for (dim_t n = N_s; n < N_e; ++n)
                        PRAGMA_OMP_SIMD(reduction(+ : sum))
                        for (dim_t sp = S_s; sp < S_e; ++sp) {
                            sum += src[off + n * C * SP + sp];
                        }
                    ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                        = sum;
                }

                if (SP_N_nthr > 1) mkldnn_thr_barrier();

                for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    mean_blk[c] = 0.;
                    for (dim_t n = 0; n < SP_N_nthr; n++)
                        mean_blk[c] += ws_reduce[ws_iter_off
                                + n * C_blks_per_iter + c];
                    mean_blk[c] /= (N * SP);
                }

                if (SP_N_nthr > 1) mkldnn_thr_barrier();

                for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                    size_t off = c + C_off;
                    data_t sum = 0.;
                    for (dim_t n = N_s; n < N_e; ++n)
                        PRAGMA_OMP_SIMD(reduction(+ : sum))
                        for (dim_t sp = S_s; sp < S_e; ++sp) {
                            data_t m = src[off * SP + n * C * SP + sp]
                                    - mean[off];
                            sum += m * m;
                        }
                    ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                        = sum;
                }

                if (SP_N_nthr > 1) mkldnn_thr_barrier();

                for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                    variance_blk[c] = 0.;
                    for (dim_t n = 0; n < SP_N_nthr; n++)
                        variance_blk[c] += ws_reduce[ws_iter_off
                                + n * C_blks_per_iter + c];
                    variance_blk[c] /= (N * SP);
                }

                if (SP_N_nthr > 1) mkldnn_thr_barrier();
            }

            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                data_t sqrt_variance
                        = static_cast<data_t>(sqrtf(variance[off] + eps));
                data_t sm = (use_scaleshift ? scaleshift[off] : 1.0f) / sqrt_variance;
                data_t sv = use_scaleshift ? scaleshift[C + off] : 0;
                for (dim_t n = N_s; n < N_e; ++n)
#if SAFE_TO_USE_OMP_SIMD
                    PRAGMA_OMP_SIMD()
#endif
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        size_t d_off = off * SP + n * C * SP + sp;
                        data_t bn_res
                                = sm * (src[d_off] - mean[off]) + sv;
                        if (fuse_bn_relu) {
                            if (bn_res <= 0) {
                                bn_res = 0;
                                if (is_training)
                                    ws[d_off] = 0;
                            } else {
                                if (is_training)
                                    ws[d_off] = 1;
                            }
                        }
                        dst[d_off] = maybe_post_op(bn_res);
                    }
            }
        }
    });
}

void ncsp_batch_normalization_bwd_t::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto mean = CTX_IN_MEM(const data_t *, MKLDNN_ARG_MEAN);
    auto variance = CTX_IN_MEM(const data_t *, MKLDNN_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, MKLDNN_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SCALE_SHIFT);

    auto scratchpad = this->scratchpad(ctx);
    auto *ws_reduce = scratchpad.get<data_t>(key_bnorm_reduction);

    if (diff_scaleshift == nullptr)
        diff_scaleshift = scratchpad.get<data_t>(key_bnorm_tmp_diff_ss);

    const bool has_spatial = utils::one_of(pd()->ndims(), 4, 5);
    dim_t SP = (has_spatial) ? pd()->H() * pd()->W() * pd()->D() : 1;
    dim_t C = pd()->C(), N = pd()->MB();
    const bool use_scaleshift = pd()->use_scaleshift();
    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    int nthr = mkldnn_get_max_threads();
    size_t l3_size_ = get_cache_size(3, true) * nthr / 2;
    size_t data_size = N * C * SP * sizeof(data_t);
    bool do_blocking = (data_size >= l3_size_ / 2 && l3_size_ > 0);

    parallel(0, [&](const int ithr, const int nthr) {
        int C_ithr = 0, C_nthr = 0;
        int N_ithr = 0, N_nthr = 0;
        int S_ithr = 0, S_nthr = 0;

        dim_t C_blk_gl_s = 0, C_blk_gl_e = 0, C_blk_s = 0, C_blk_e = 0;
        dim_t N_s = 0, N_e = 0;
        dim_t S_s = 0, S_e = 0;

        dim_t C_blks_per_iter = 1;
        int64_t iters = 1;

        if (do_blocking) {
            size_t working_set_size = 2 * N * SP * sizeof(data_t);
            bnorm_utils::cache_balance(
                    working_set_size, C, C_blks_per_iter, iters);
        } else
            C_blks_per_iter = C;
        int64_t last_iter_blks = C - (iters - 1) * C_blks_per_iter;
        bool spatial_thr_allowed
                = bnorm_utils::thread_balance(do_blocking, true, ithr, nthr, N,
                        C_blks_per_iter, SP, C_ithr, C_nthr, C_blk_s, C_blk_e,
                        N_ithr, N_nthr, N_s, N_e, S_ithr, S_nthr, S_s, S_e);
        balance211(C_blks_per_iter, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
        int SP_N_ithr = N_ithr * S_nthr + S_ithr;
        int SP_N_nthr = N_nthr * S_nthr;

        for (int64_t it = 0; it < iters; ++it) {
            if (it == iters - 1 && iters > 1) {
                // On the last iteration the access pattern to ws_reduce
                // might change (due to re-balance on C). So sync the
                // threads if they are not synced by the algorithm.
                if (SP_N_nthr == 1 && mkldnn_thr_syncable())
                    mkldnn_thr_barrier();

                C_blk_s = C_blk_e = N_s = N_e = 0;
                spatial_thr_allowed = bnorm_utils::thread_balance(do_blocking,
                        spatial_thr_allowed, ithr, nthr, N, last_iter_blks, SP,
                        C_ithr, C_nthr, C_blk_s, C_blk_e, N_ithr, N_nthr, N_s,
                        N_e, S_ithr, S_nthr, S_s, S_e);
                balance211(last_iter_blks, nthr, ithr, C_blk_gl_s, C_blk_gl_e);
                SP_N_ithr = N_ithr * S_nthr + S_ithr;
                SP_N_nthr = N_nthr * S_nthr;
            }
            size_t C_off = it * C_blks_per_iter;
            // On the last iteration the access pattern to ws_reduce
            // might change (due to re-balance on C). Since sync is not always
            // possible (in case of TBB) use different parts of ws for each
            // iteration if threads are not synced by the algorithm.
            size_t ws_iter_off = (mkldnn_thr_syncable() ? 0 : 1) * 2 * C_off;

            data_t *diff_gamma_blk = diff_scaleshift + C_off;
            data_t *diff_beta_blk = diff_scaleshift + C + C_off;
            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                data_t diff_gamma = 0.0, diff_beta = 0.0;
                data_t v_mean = mean[off];
                for (dim_t n = N_s; n < N_e; ++n)
                    PRAGMA_OMP_SIMD(reduction(+ : diff_gamma, diff_beta))
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        const size_t d_off = off * SP + n * C * SP + sp;
                        data_t dd;
                        if (fuse_bn_relu)
                            dd = (!ws[d_off]) ? 0 : diff_dst[d_off];
                        else
                            dd = diff_dst[d_off];
                        diff_gamma += (src[d_off] - v_mean) * dd;
                        diff_beta += dd;
                    }
                ws_reduce[ws_iter_off + SP_N_ithr * C_blks_per_iter + c]
                    = diff_gamma;
                ws_reduce[ws_iter_off + SP_N_nthr * C_blks_per_iter
                        + SP_N_ithr * C_blks_per_iter + c] = diff_beta;
            }

            if (SP_N_nthr > 1) mkldnn_thr_barrier();

            for (dim_t c = C_blk_gl_s; c < C_blk_gl_e; c++) {
                data_t sqrt_variance = static_cast<data_t>(
                        1.0f / sqrtf(variance[c + C_off] + eps));
                diff_gamma_blk[c] = 0.;
                diff_beta_blk[c] = 0.;
                for (dim_t n = 0; n < SP_N_nthr; n++) {
                    diff_gamma_blk[c] += ws_reduce[ws_iter_off
                            + n * C_blks_per_iter + c];
                    diff_beta_blk[c] += ws_reduce[ws_iter_off
                            + SP_N_nthr * C_blks_per_iter + n * C_blks_per_iter
                            + c];
                }
                diff_gamma_blk[c] *= sqrt_variance;
            }

            if (SP_N_nthr > 1) mkldnn_thr_barrier();

            for (dim_t c = C_blk_s; c < C_blk_e; c++) {
                size_t off = c + C_off;
                data_t gamma = use_scaleshift ? scaleshift[off] : 1;
                data_t sqrt_variance
                        = static_cast<data_t>(1.0f / sqrtf(variance[off] + eps));
                data_t v_mean = mean[off];
                for (dim_t n = N_s; n < N_e; ++n)
#if SAFE_TO_USE_OMP_SIMD
                    PRAGMA_OMP_SIMD()
#endif
                    for (dim_t sp = S_s; sp < S_e; ++sp) {
                        const size_t d_off = off * SP + n * C * SP + sp;

                        data_t v_diff_src;
                        if (fuse_bn_relu)
                            v_diff_src = (!ws[d_off]) ? 0 : diff_dst[d_off];
                        else
                            v_diff_src = diff_dst[d_off];
                        if (calculate_diff_stats) {
                            v_diff_src -= diff_beta_blk[c] / (SP * N)
                                    + (src[d_off] - v_mean) * diff_gamma_blk[c]
                                            * sqrt_variance / (SP * N);
                        }
                        v_diff_src *= gamma * sqrt_variance;
                        diff_src[d_off] = v_diff_src;
                    }
            }
        }
    });
}
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
