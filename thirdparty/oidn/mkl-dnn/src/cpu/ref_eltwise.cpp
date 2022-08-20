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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"

#include "ref_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace alg_kind;
using namespace math;

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(alg_kind_t alg, float alpha,
        float beta): alg_(alg), alpha_(alpha), beta_(beta) {
    assert(utils::one_of(alg_, eltwise_relu, eltwise_tanh, eltwise_elu,
                eltwise_square, eltwise_abs, eltwise_sqrt, eltwise_linear,
                eltwise_bounded_relu, eltwise_soft_relu, eltwise_logistic));
}

ref_eltwise_scalar_fwd_t::ref_eltwise_scalar_fwd_t(
        const post_ops_t::entry_t::eltwise_t &eltwise)
    : ref_eltwise_scalar_fwd_t(eltwise.alg, eltwise.alpha, eltwise.beta) {}

float ref_eltwise_scalar_fwd_t::compute_scalar(float s) {
    switch (alg_) {
        case eltwise_relu: return relu_fwd(s, alpha_);
        case eltwise_tanh: return tanh_fwd(s);
        case eltwise_elu: return elu_fwd(s, alpha_);
        case eltwise_square: return square_fwd(s);
        case eltwise_abs: return abs_fwd(s);
        case eltwise_sqrt: return sqrt_fwd(s);
        case eltwise_linear: return linear_fwd(s, alpha_, beta_);
        case eltwise_bounded_relu: return bounded_relu_fwd(s, alpha_);
        case eltwise_soft_relu: return soft_relu_fwd(s);
        case eltwise_logistic: return logistic_fwd(s);
        default: assert(!"unknown eltwise alg_kind");
    }

    return 0.f;
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_nCspBc_padded(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());
    const blocking_desc_t &blk = data_d.blocking_desc();
    const int block = blk.inner_blks[0];

    const int MB = pd()->MB();
    const int C = pd()->C() / block;
    const int C_PADDED = data_d.padded_dims()[1] / block;
    const int tail = pd()->C() % block;
    const int SP = pd()->D() * pd()->H() * pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    auto ker = [=] (data_t &d, data_t s) {
        switch (alg_kind) {
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_bounded_relu:
                d = bounded_relu_fwd(s, alpha); break;
            case eltwise_soft_relu: d = soft_relu_fwd(s); break;
            case eltwise_logistic: d = logistic_fwd(s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    };

    // FIXME: integer overflow?

    parallel_nd(MB, C_PADDED, SP,
        [&](int n, int c, int sp) {
        auto d_off = (n*C_PADDED*SP + c*SP + sp) * block;
        if (c < C) {
            for (int v = 0; v < block; v++)
                ker(dst[d_off + v], src[d_off + v]);
        } else {
            for (int v = 0; v < tail; v++)
                ker(dst[d_off + v], src[d_off + v]);
        }
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int id, int h, int w) {
        auto d_off = is_3d
            ? data_d.off(n, c, id, h, w) : data_d.off(n, c, h, w);
        data_t s = src[d_off];
        data_t &d = dst[d_off];
        switch (alg_kind) {
            case eltwise_relu: d = relu_fwd(s, alpha); break;
            case eltwise_tanh: d = tanh_fwd(s); break;
            case eltwise_elu: d = elu_fwd(s, alpha); break;
            case eltwise_square: d = square_fwd(s); break;
            case eltwise_abs: d = abs_fwd(s); break;
            case eltwise_sqrt: d = sqrt_fwd(s); break;
            case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
            case eltwise_bounded_relu:
                d = bounded_relu_fwd(s, alpha); break;
            case eltwise_soft_relu: d = soft_relu_fwd(s); break;
            case eltwise_logistic: d = logistic_fwd(s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_fwd_t<data_type>::execute_forward_dense(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper data_d(pd()->src_md());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta  = pd()->desc()->beta;

    src += data_d.offset0();
    dst += data_d.offset0();

    if (alg_kind == eltwise_relu) {
        // a fast path for relu as the most popular activation
        parallel_nd(nelems, [&](ptrdiff_t e) {
            dst[e] = relu_fwd(src[e], alpha);
        });
        return;
    }

    parallel_nd(nelems, [&](ptrdiff_t e) {
        const data_t s = src[e];
        data_t &d = dst[e];

        switch (alg_kind) {
        case eltwise_tanh: d = tanh_fwd(s); break;
        case eltwise_elu: d = elu_fwd(s, alpha); break;
        case eltwise_square: d = square_fwd(s); break;
        case eltwise_abs: d = abs_fwd(s); break;
        case eltwise_sqrt: d = sqrt_fwd(s); break;
        case eltwise_linear: d = linear_fwd(s, alpha, beta); break;
        case eltwise_bounded_relu: d = bounded_relu_fwd(s, alpha); break;
        case eltwise_soft_relu: d = soft_relu_fwd(s); break;
        case eltwise_logistic: d = logistic_fwd(s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_generic(
        const exec_ctx_t &ctx) const {
    /* fast return */
    if (pd()->has_zero_dim_memory()) return;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int D = pd()->D();
    const int H = pd()->H();
    const int W = pd()->W();
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;
    const bool is_3d = pd()->desc()->data_desc.ndims == 5;

    parallel_nd(MB, C, D, H, W,
        [&](int n, int c, int d, int h, int w) {
        auto data_off = is_3d
            ? data_d.off(n, c, d, h, w) : data_d.off(n, c, h, w);
        auto diff_data_off = is_3d
            ? diff_data_d.off(n, c, d, h, w)
            : diff_data_d.off(n, c, h, w);
        data_t s = src[data_off];
        data_t dd = diff_dst[diff_data_off];
        data_t &ds = diff_src[diff_data_off];
        switch (alg_kind) {
            case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
            case eltwise_tanh: ds = tanh_bwd(dd, s); break;
            case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
            case eltwise_square: ds = square_bwd(dd, s); break;
            case eltwise_abs: ds = abs_bwd(dd, s); break;
            case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
            case eltwise_linear:
                ds = linear_bwd(dd, s, alpha, beta); break;
            case eltwise_bounded_relu:
                ds = bounded_relu_bwd(dd, s, alpha); break;
            case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
            case eltwise_logistic: ds = logistic_bwd(dd, s); break;
            default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template <impl::data_type_t data_type>
void ref_eltwise_bwd_t<data_type>::execute_backward_dense(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper data_d(pd()->src_md());
    const memory_desc_wrapper diff_data_d(pd()->diff_src_md());

    const ptrdiff_t nelems = static_cast<ptrdiff_t>(data_d.nelems(true));
    const auto alg_kind = pd()->desc()->alg_kind;
    const float alpha = pd()->desc()->alpha;
    const float beta = pd()->desc()->beta;

    src += data_d.offset0();
    diff_dst += diff_data_d.offset0();
    diff_src += diff_data_d.offset0();

    parallel_nd(nelems, [&](ptrdiff_t e) {
        const data_t dd = diff_dst[e];
        const data_t s = src[e];
        data_t &ds = diff_src[e];

        switch (alg_kind) {
        case eltwise_relu: ds = relu_bwd(dd, s, alpha); break;
        case eltwise_tanh: ds = tanh_bwd(dd, s); break;
        case eltwise_elu: ds = elu_bwd(dd, s, alpha); break;
        case eltwise_square: ds = square_bwd(dd, s); break;
        case eltwise_abs: ds = abs_bwd(dd, s); break;
        case eltwise_sqrt: ds = sqrt_bwd(dd, s); break;
        case eltwise_linear: ds = linear_bwd(dd, s, alpha, beta); break;
        case eltwise_bounded_relu: ds = bounded_relu_bwd(dd, s, alpha); break;
        case eltwise_soft_relu: ds = soft_relu_bwd(dd, s); break;
        case eltwise_logistic: ds = logistic_bwd(dd, s); break;
        default: assert(!"unknown eltwise alg_kind");
        }
    });
}

template struct ref_eltwise_fwd_t<data_type::f32>;
template struct ref_eltwise_fwd_t<data_type::s32>;
template struct ref_eltwise_fwd_t<data_type::s8>;
template struct ref_eltwise_fwd_t<data_type::u8>;

template struct ref_eltwise_bwd_t<data_type::f32>;
template struct ref_eltwise_bwd_t<data_type::s32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
