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

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_uni_lrn_kernel_f32.hpp"
#include "jit_uni_lrn.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

template <cpu_isa_t isa>
jit_uni_lrn_fwd_t<isa>::jit_uni_lrn_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), ker_(nullptr)
    , ker_first_(nullptr), ker_last_(nullptr)
{
    using namespace alg_kind;

    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    float A = pd()->desc()->lrn_alpha / ls;
    float K = pd()->desc()->lrn_k;

    auto pk = pd()->desc()->prop_kind;
    auto ak = pd()->desc()->alg_kind;
    auto dat_tag = pd()->dat_tag_;

    if (dat_tag == nChw8c && ls == 5 && ak == lrn_across_channels) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, 0), A, K, pk);
        ker_first_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, -1), A, K, pk);
        ker_last_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_across(H, W, +1), A, K, pk);
    } else if (dat_tag == nChw8c && ak == lrn_within_channel) {
        /* within channel, local_size (x) local_size */
        A /= ls; /* XXX: why? */
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw8c_within(H, W, ls), A, K, pk);
    } else if (dat_tag == nchw && ls == 5 && ak == lrn_across_channels) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                nchw_across(C, H*W, 0), A, K, pk);
        int remind = (H*W) % VECTOR_LENGTH;
        if (remind != 0) {
            ker_last_ = new jit_uni_lrn_fwd_kernel_f32<isa>(
                        nchw_across(C, H*W, remind), A, K, pk);
        }
    } else if (true /* XXX: why */) {
        ker_ = new jit_uni_lrn_fwd_kernel_f32<isa>(nhwc_across(C), A, K, pk);
    }
}

template <cpu_isa_t isa>
jit_uni_lrn_fwd_t<isa>::~jit_uni_lrn_fwd_t()
{ delete ker_; delete ker_first_; delete ker_last_; }

template <cpu_isa_t isa>
void jit_uni_lrn_fwd_t<isa>::execute_forward(const exec_ctx_t &ctx) const {
    using namespace alg_kind;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(data_t *, MKLDNN_ARG_WORKSPACE);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int HW = pd()->H() * pd()->W();
    const int ls = pd()->desc()->local_size;

    auto ak = pd()->desc()->alg_kind;
    auto dat_tag = pd()->dat_tag_;

    if (dat_tag == nChw8c && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + c8 * HW * VECTOR_LENGTH];
            if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else if (dat_tag == nChw8c && ak == lrn_within_channel) {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + c8 * HW * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + c8 * HW * VECTOR_LENGTH];
            (*ker_)(&args);
        });
    }
    else if (dat_tag == nchw && ls == 5 && ak == lrn_across_channels) {
        parallel_nd(N, (HW + VECTOR_LENGTH - 1) / VECTOR_LENGTH,
            [&](int n, int hw8) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + hw8 * VECTOR_LENGTH];
            args.dst = &dst[n*HW*C + hw8 * VECTOR_LENGTH];
            args.scratch = &ws[n*HW*C + hw8 * VECTOR_LENGTH];
            if ((hw8 + 1)*VECTOR_LENGTH > HW)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else { // nhwc
        parallel_nd(N, HW, [&](int n, int hw) {
            jit_args_fwd_t args;
            args.src = &src[n*HW*C + hw * C];
            args.dst = &dst[n*HW*C + hw * C];
            args.scratch = &ws[n*HW*C + hw * C];
            (*ker_)(&args);
        });
    }
}

template <cpu_isa_t isa>
status_t jit_uni_lrn_fwd_t<isa>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    bool ok = true
        && mayiuse(isa)
        && is_fwd()
        && everyone_is(data_type::f32, data_d.data_type())
        && !has_zero_dim_memory()
        && data_d.ndims() == 4
        && data_d.dims()[1] % VECTOR_LENGTH == 0
        && data_d.dims()[1] >= 2 * VECTOR_LENGTH
        && desc()->lrn_beta == 0.75
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    if (desc_.prop_kind == forward_training) ws_md_ = *src_md();

    dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nChw8c, nchw, nhwc);

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && one_of(dat_tag_, nChw8c, nchw, nhwc);

    const int jit_max_local_size = 5; // bigger size triggers too big code size
    bool args_ok_within = true
        && desc()->alg_kind == lrn_within_channel
        && desc()->local_size <= ( jit_max_local_size <= MAX_LOCAL_SIZE
                                 ? jit_max_local_size : MAX_LOCAL_SIZE)
        && data_d.dims()[2] >= desc()->local_size
        && data_d.dims()[3] >= desc()->local_size
        && one_of(dat_tag_, nChw8c);

    return args_ok_across || args_ok_within ? success : unimplemented;
}

template <cpu_isa_t isa>
jit_uni_lrn_bwd_t<isa>::jit_uni_lrn_bwd_t(const pd_t *apd)
    : cpu_primitive_t(apd)
    , ker_(nullptr), ker_first_(nullptr), ker_last_(nullptr)
{
    using namespace alg_kind;
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    float A = pd()->desc()->lrn_alpha / ls;
    float B = pd()->desc()->lrn_beta;

    int use_h_parallelizm = 0;// XXX
    if (C / VECTOR_LENGTH == 1) {
        ker_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, 3), A, B, use_h_parallelizm);
    }
    else {
        ker_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, 0), A, B, use_h_parallelizm);
        ker_first_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, -1), A, B, use_h_parallelizm);
        ker_last_ = new jit_uni_lrn_bwd_kernel_f32<isa>(
            nchw8c_across(H, W, +1), A, B, use_h_parallelizm);
    }
}

template <cpu_isa_t isa>
jit_uni_lrn_bwd_t<isa>::~jit_uni_lrn_bwd_t()
{
    delete ker_; delete ker_first_; delete ker_last_;
}

template <cpu_isa_t isa>
void jit_uni_lrn_bwd_t<isa>::execute_backward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const data_t *, MKLDNN_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();

    int use_h_parallelizm = 0; // XXX
    if (use_h_parallelizm) {
        parallel_nd(N, C / VECTOR_LENGTH, H, [&](int n, int c8, int h) {
            auto offset = n*C*H*W + c8*H*W*VECTOR_LENGTH
                + h*W*VECTOR_LENGTH;
            jit_args_bwd_t args;
            args.src = &src[offset];
            args.diff_dst = &diff_dst[offset];
            args.scratch = &ws[offset];
            args.diff_src = &diff_src[offset];
            if (C / VECTOR_LENGTH == 1)
                (*ker_)(&args);
            else if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
    else {
        parallel_nd(N, C / VECTOR_LENGTH, [&](int n, int c8) {
            auto offset = n*C*H*W + c8*H*W*VECTOR_LENGTH;
            jit_args_bwd_t args;
            args.src = &src[offset];
            args.diff_dst = &diff_dst[offset];
            args.scratch = &ws[offset];
            args.diff_src = &diff_src[offset];
            if (C / VECTOR_LENGTH == 1)
                (*ker_)(&args);
            else if (c8 == 0)
                (*ker_first_)(&args);
            else if (c8 == C / VECTOR_LENGTH - 1)
                (*ker_last_)(&args);
            else
                (*ker_)(&args);
        });
    }
}

template <cpu_isa_t isa>
status_t jit_uni_lrn_bwd_t<isa>::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    bool ok = true
        && mayiuse(isa)
        && !is_fwd()
        && utils::everyone_is(data_type::f32, data_d.data_type())
        && !has_zero_dim_memory()
        && data_d.ndims() == 4
        && data_d.dims()[1] % VECTOR_LENGTH == 0
        && desc()->lrn_beta == 0.75
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    ws_md_ = *src_md();
    if (!compare_ws(hint_fwd_pd_)) return unimplemented;

    dat_tag_ = memory_desc_matches_one_of_tag(*src_md(), nChw8c);

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && utils::one_of(dat_tag_, nChw8c);

    return args_ok_across ? success : unimplemented;
}

template struct jit_uni_lrn_fwd_t<sse42>;
template struct jit_uni_lrn_fwd_t<avx2>;
template struct jit_uni_lrn_bwd_t<avx2>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
