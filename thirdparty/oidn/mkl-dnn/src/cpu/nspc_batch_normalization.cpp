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

#include "nspc_batch_normalization.hpp"

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

void nspc_batch_normalization_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();
    const bool calculate_stats = !pd()->stats_is_src();
    const bool with_relu = pd()->with_relu_post_op();

    auto scratchpad = this->scratchpad(ctx);
    auto tmp_mean = scratchpad.get<data_t>(key_bnorm_tmp_mean);
    auto tmp_var = scratchpad.get<data_t>(key_bnorm_tmp_var);
    auto *ws_reduce = scratchpad.get<data_t>(key_bnorm_reduction);

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto scaleshift = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SCALE_SHIFT);

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
            mean = tmp_mean;
            variance = tmp_var;
        }
    }

    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, MKLDNN_ARG_WORKSPACE);

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t SP = pd()->H() * pd()->W() * pd()->D();

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    auto maybe_post_op
            = [&](data_t res) { return (with_relu && res < 0) ? 0 : res; };

    assert(mkldnn_thr_syncable());
    parallel(0, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);
        data_t *mean_loc = tmp_mean + nstl::max(C, (dim_t)16) * ithr;
        data_t *variance_loc = tmp_var + nstl::max(C, (dim_t)16) * ithr;

        if (calculate_stats) {
            for (dim_t c = 0; c < C; c++)
                ws_reduce[C * ithr + c] = 0.;

            for (dim_t n = N_s; n < N_e; n++)
                for (dim_t sp = 0; sp < SP; sp++)
                    PRAGMA_OMP_SIMD()
                    for (dim_t c = 0; c < C; c++)
                        ws_reduce[C * ithr + c] += src[(size_t)n * SP * C
                            + sp * C + c];

            mkldnn_thr_barrier();

            for (dim_t c = C_s; c < C_e; c++) {
                mean[c] = 0;
                for (dim_t n = 0; n < nthr; n++)
                    mean[c] += ws_reduce[C * n + c];
                mean[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (dim_t c = 0; c < C; c++) {
                mean_loc[c] = mean[c];
                ws_reduce[C * ithr + c] = 0.;
            }

            for (dim_t n = N_s; n < N_e; n++)
                for (dim_t sp = 0; sp < SP; sp++)
                    PRAGMA_OMP_SIMD()
                    for (dim_t c = 0; c < C; c++) {
                        data_t m = src[(size_t)n * SP * C + sp * C + c]
                            - mean_loc[c];
                        ws_reduce[C * ithr + c] += m * m;
                    }

            mkldnn_thr_barrier();

            for (dim_t c = C_s; c < C_e; c++) {
                variance[c] = 0;
                for (dim_t n = 0; n < nthr; n++)
                    variance[c] += ws_reduce[C * n + c];
                variance[c] /= SP * N;
            }

            mkldnn_thr_barrier();

            for (dim_t c = 0; c < C; c++)
                variance_loc[c] = variance[c];
        } else {
            variance_loc = variance;
            mean_loc = mean;
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    data_t sqrt_variance = static_cast<data_t>(
                            sqrtf(variance_loc[c] + eps));
                    data_t sm = (use_scaleshift ? scaleshift[c] : 1.0f) / sqrt_variance;
                    data_t sv = use_scaleshift ? scaleshift[C + c] : 0;
                    size_t d_off = (size_t)n * SP * C + sp * C + c;
                    data_t bn_res = sm * (src[d_off] - mean_loc[c]) + sv;
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

void nspc_batch_normalization_bwd_t::execute_backward(
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
    auto tmp_diff_ss = scratchpad.get<data_t>(key_bnorm_tmp_diff_ss);

    if (diff_scaleshift == nullptr)
        diff_scaleshift = tmp_diff_ss;

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    const dim_t SP = pd()->D() * pd()->H() * pd()->W();
    data_t *diff_gamma = diff_scaleshift, *diff_beta = diff_scaleshift + C;
    auto *ws_reduce = scratchpad.get<data_t>(key_bnorm_reduction);

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    assert(mkldnn_thr_syncable());
    parallel(0, [&](const int ithr, const int nthr) {
        dim_t N_s = 0, N_e = 0, C_s = 0, C_e = 0;
        balance211(N, nthr, ithr, N_s, N_e);
        balance211(C, nthr, ithr, C_s, C_e);

        data_t *diff_gamma_loc = tmp_diff_ss + 2 * C + C * ithr;
        data_t *diff_beta_loc = tmp_diff_ss + 2 * C + C * (nthr + ithr);

        for (dim_t c = 0; c < C; c++) {
            ws_reduce[C * ithr + c] = 0.;
            ws_reduce[C * nthr + C * ithr + c] = 0.;
        }

        for (dim_t n = N_s; n < N_e; n++)
            for (dim_t sp = 0; sp < SP; sp++)
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    const size_t d_off = (size_t)n * SP * C + sp * C + c;
                    data_t dd;
                    if (fuse_bn_relu)
                        dd = (!ws[d_off]) ? 0 : diff_dst[d_off];
                    else
                        dd = diff_dst[d_off];
                    ws_reduce[C * ithr + c] += (src[d_off] - mean[c]) * dd;
                    ws_reduce[C * nthr + C * ithr + c] += dd;
                }

        mkldnn_thr_barrier();

        for (dim_t c = C_s; c < C_e; c++) {
            data_t sqrt_variance
                    = static_cast<data_t>(1.0f / sqrtf(variance[c] + eps));
            diff_gamma[c] = 0;
            diff_beta[c] = 0;
            for (dim_t n = 0; n < nthr; n++) {
                diff_gamma[c] += ws_reduce[C * n + c];
                diff_beta[c] += ws_reduce[C * nthr + C * n + c];
            }
            diff_gamma[c] *= sqrt_variance;
        }

        mkldnn_thr_barrier();

        for (dim_t c = 0; c < C; c++) {
            diff_gamma_loc[c] = diff_gamma[c];
            diff_beta_loc[c] = diff_beta[c];
        }

        for (dim_t n = N_s; n < N_e; n++) {
            for (dim_t sp = 0; sp < SP; sp++) {
#if SAFE_TO_USE_OMP_SIMD
                PRAGMA_OMP_SIMD()
#endif
                for (dim_t c = 0; c < C; c++) {
                    const size_t d_off = (size_t)n * SP * C + sp * C + c;
                    data_t gamma = use_scaleshift ? scaleshift[c] : 1;
                    data_t sqrt_variance
                            = static_cast<data_t>(1.0f / sqrtf(variance[c] + eps));
                    data_t v_diff_src;
                    if (fuse_bn_relu)
                        v_diff_src = (!ws[d_off]) ? 0 : diff_dst[d_off];
                    else
                        v_diff_src = diff_dst[d_off];
                    if (calculate_diff_stats) {
                        v_diff_src -= diff_beta_loc[c] / (SP * N)
                                + (src[d_off] - mean[c]) * diff_gamma_loc[c]
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
