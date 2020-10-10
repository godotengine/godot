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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"

#include "ref_batch_normalization.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void ref_batch_normalization_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (this->pd()->has_zero_dim_memory()) return;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto scaleshift = CTX_IN_MEM(const float *, MKLDNN_ARG_SCALE_SHIFT);

    auto mean = pd()->stats_is_src()
        ? const_cast<float *>(CTX_IN_MEM(const float *, MKLDNN_ARG_MEAN))
        : CTX_OUT_MEM(float *, MKLDNN_ARG_MEAN);
    auto variance = pd()->stats_is_src()
        ? const_cast<float *>(CTX_IN_MEM(const float *, MKLDNN_ARG_VARIANCE))
        : CTX_OUT_MEM(float *, MKLDNN_ARG_VARIANCE);

    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(uint8_t *, MKLDNN_ARG_WORKSPACE);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper scaleshift_d(pd()->weights_md());

    const dim_t N = pd()->MB();
    const dim_t C = pd()->C();
    dim_t H = 1, W = 1, D = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 4, 5);
    if (has_spatial) {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
    }

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();;
    const bool save_stats = pd()->is_training();
    const bool is_training = pd()->is_training();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();
    const bool calculate_stats = !pd()->stats_is_src();

    const bool with_relu = pd()->with_relu_post_op();
    auto maybe_post_op = [&](float res) {
        return (with_relu && res < 0.0f) ? 0.0f : res;
    };
    const bool is_3d = data_d.ndims() == 5;

    auto data_offset = [&](const memory_desc_wrapper &data_d, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        if (has_spatial) {
            if (is_3d)
                return data_d.off(n, c, d, h, w);
            else
                return data_d.off(n, c, h, w);
        } else
            return data_d.off(n, c);
    };

    parallel_nd(C, [&](dim_t c) {
        float v_mean = calculate_stats ? 0 : mean[c];
        float v_variance = calculate_stats ? 0 : variance[c];

        if (calculate_stats) {
            for (dim_t n = 0; n < N; ++n)
            for (dim_t d = 0; d < D; ++d)
            for (dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w)
                v_mean += src[data_offset(data_d, n, c, d, h, w)];
            v_mean /= W*N*H*D;

            for (dim_t n = 0; n < N; ++n)
            for (dim_t d = 0; d < D; ++d)
            for (dim_t h = 0; h < H; ++h)
            for (dim_t w = 0; w < W; ++w) {
                float m = src[data_offset(data_d, n, c, d, h, w)] - v_mean;
                v_variance += m*m;
            }
            v_variance /= W*H*N*D;
        }

        float sqrt_variance = sqrtf(v_variance + eps);
        float sm = (use_scaleshift
            ? scaleshift[scaleshift_d.off(0, c)]
            : 1.0f) / sqrt_variance;
        float sv = use_scaleshift ? scaleshift[scaleshift_d.off(1, c)] : 0;

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            auto d_off = data_offset(data_d,n,c,d,h,w);
            float bn_res = sm * ((float)src[d_off] - v_mean) + sv;
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
            if (data_type == data_type::s8) {
                dst[d_off] = qz_a1b0<float, data_t>()(maybe_post_op(bn_res));
            } else {
                dst[d_off] = static_cast<data_t>(maybe_post_op(bn_res));
            }
        }

        if (calculate_stats) {
            if (save_stats) {
                mean[c] = v_mean;
                variance[c] = v_variance;
            }
        }
    });
}

template struct ref_batch_normalization_fwd_t<data_type::f32>;
template struct ref_batch_normalization_fwd_t<data_type::s8>;

template <impl::data_type_t data_type>
void ref_batch_normalization_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto mean = CTX_IN_MEM(const data_t *, MKLDNN_ARG_MEAN);
    auto variance = CTX_IN_MEM(const data_t *, MKLDNN_ARG_VARIANCE);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto scaleshift = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SCALE_SHIFT);
    auto ws = CTX_IN_MEM(const uint8_t *, MKLDNN_ARG_WORKSPACE);

    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);
    auto diff_scaleshift = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SCALE_SHIFT);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());
    const memory_desc_wrapper scaleshift_d(pd()->weights_md());
    const memory_desc_wrapper diff_scaleshift_d(pd()->diff_weights_md());

    const dim_t C = pd()->C();

    /* fast return */
    if (this->pd()->has_zero_dim_memory()) {
        if (diff_scaleshift) {
            for (dim_t c = 0; c < C; ++c) {
                diff_scaleshift[diff_scaleshift_d.off(0, c)] = 0;
                diff_scaleshift[diff_scaleshift_d.off(1, c)] = 0;
            }
        }
        return;
    }

    const dim_t N = pd()->MB();
    dim_t H = 1, W = 1, D = 1;
    const bool has_spatial = utils::one_of(data_d.ndims(), 4, 5);
    if (has_spatial) {
        D = pd()->D();
        H = pd()->H();
        W = pd()->W();
    }

    const float eps = pd()->desc()->batch_norm_epsilon;
    const bool use_scaleshift = pd()->use_scaleshift();
    const bool calculate_diff_stats = !pd()->use_global_stats();
    const bool fuse_bn_relu = pd()->fuse_bn_relu();

    const bool is_3d = data_d.ndims() == 5;

    auto data_offset = [&](const memory_desc_wrapper &data_d, dim_t n, dim_t c,
            dim_t d, dim_t h, dim_t w) {
        if (has_spatial) {
            if (is_3d)
                return data_d.off(n, c, d, h, w);
            else
                return data_d.off(n, c, h, w);
        } else
            return data_d.off(n, c);
    };

    parallel_nd(C, [&](dim_t c) {
        data_t v_mean = mean[c];
        data_t v_variance = variance[c];
        data_t sqrt_variance = static_cast<data_t>(1.0f / sqrtf(v_variance + eps));
        data_t gamma = use_scaleshift ? scaleshift[scaleshift_d.off(0, c)] : 1;
        data_t diff_gamma = data_t(0);
        data_t diff_beta = data_t(0);
        diff_gamma = 0.0;
        diff_beta = 0.0;

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = data_offset(data_d, n, c, d, h, w);
            data_t dd = diff_dst[data_offset(diff_data_d, n, c, d, h, w)];
            if (fuse_bn_relu && !ws[s_off])
                dd = 0;

            diff_gamma += (src[s_off] - v_mean) * dd;
            diff_beta += dd;
        }
        diff_gamma *= sqrt_variance;

        if (diff_scaleshift) {
            diff_scaleshift[diff_scaleshift_d.off(0, c)] = diff_gamma;
            diff_scaleshift[diff_scaleshift_d.off(1, c)] = diff_beta;
        }

        for (dim_t n = 0; n < N; ++n)
        for (dim_t d = 0; d < D; ++d)
        for (dim_t h = 0; h < H; ++h)
        for (dim_t w = 0; w < W; ++w) {
            const size_t s_off = data_offset(data_d, n, c, d, h, w);
            const size_t dd_off = data_offset(diff_data_d, n, c, d, h, w);
            data_t dd = diff_dst[dd_off];
            if (fuse_bn_relu && !ws[s_off])
                dd = 0;

            data_t v_diff_src = dd;
            if (calculate_diff_stats) {
                v_diff_src -= diff_beta/(D*W*H*N) +
                    (src[s_off] - v_mean) *
                    diff_gamma*sqrt_variance/(D*W*H*N);
            }
            v_diff_src *= gamma*sqrt_variance;
            diff_src[dd_off] = v_diff_src;
        }
    });
}

template struct ref_batch_normalization_bwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
