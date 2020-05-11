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
#include "mkldnn_thread.hpp"

#include "gemm_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::prop_kind;
using namespace mkldnn::impl::data_type;
using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::primitive_kind;

template <impl::data_type_t data_type>
void gemm_inner_product_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    bool wei_tr = !memory_desc_matches_one_of_tag(
            *pd()->weights_md(), hwio, dhwio, io);

    const auto &post_ops = pd()->attr()->post_ops_;
    const bool do_relu = post_ops.len_ == 1;

    float alpha = 1.0, beta = 0.0;
    extended_sgemm(wei_tr ? "T" : "N", "N", &OC, &MB, &IC, &alpha, weights,
            wei_tr ? &IC : &OC, src, &IC, &beta, dst, &OC, bias);

    if (do_relu) {
        float nslope = post_ops.entry_[0].eltwise.alpha;
        parallel_nd(MB, OC, [&](int mb, int oc) {
            size_t dst_off = mb * OC + oc;
            if (dst[dst_off] < 0)
                dst[dst_off] *= nslope;
        });
    }
}

template <impl::data_type_t data_type>
void gemm_inner_product_bwd_data_t<data_type>::execute_backward_data(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const data_t *, MKLDNN_ARG_WEIGHTS);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    bool wei_tr = memory_desc_matches_one_of_tag(
            *pd()->weights_md(), hwio, dhwio, io);

    float alpha = 1.0, beta = 0.0;
    extended_sgemm(wei_tr ? "T" : "N", "N", &IC, &MB, &OC, &alpha, weights,
            wei_tr ? &OC : &IC, diff_dst, &OC, &beta, diff_src, &IC);
}

template <impl::data_type_t data_type>
void gemm_inner_product_bwd_weights_t<data_type>::execute_backward_weights(
        const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_BIAS);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    diff_dst += diff_dst_d.offset0();

    const int MB = pd()->MB();
    const int OC = pd()->OC();
    const int IC = pd()->IC_total_padded();

    bool wei_tr = memory_desc_matches_one_of_tag(
            *pd()->diff_weights_md(), hwio, dhwio, io);

    float alpha = 1.0, beta = 0.0;
    if (wei_tr)
        extended_sgemm("N", "T", &OC, &IC, &MB, &alpha, diff_dst, &OC, src, &IC,
                &beta, diff_weights, &OC);
    else
        extended_sgemm("N", "T", &IC, &OC, &MB, &alpha, src, &IC, diff_dst, &OC,
                &beta, diff_weights, &IC);

    if (diff_bias) {
        diff_bias += diff_bias_d.offset0();
        constexpr int blksize = 8;
        const int OC_blocks = OC / blksize;
        const int rem_OC = OC % blksize;
        parallel(0, [&](const int ithr, const int nthr) {
            int oc_st{0}, oc_e{0};
            balance211(OC_blocks, nthr, ithr, oc_st, oc_e);
            oc_st = oc_st * blksize;
            oc_e = oc_e * blksize;

            PRAGMA_OMP_SIMD()
            for (int oc = oc_st; oc < oc_e; ++oc) {
                diff_bias[oc] = diff_dst[oc];
            }

            for (int mb = 1; mb < MB; ++mb) {
                PRAGMA_OMP_SIMD()
                for (int oc = oc_st; oc < oc_e; ++oc) {
                    diff_bias[oc] += diff_dst[mb * OC + oc];
                }
            }

            if (rem_OC != 0 && ithr == nthr-1) {
                for (int oc = OC_blocks * blksize; oc < OC; oc++)
                    diff_bias[oc] = diff_dst[oc];
                for (int mb = 1; mb < MB; ++mb) {
                    for (int oc = OC_blocks * blksize; oc < OC; oc++) {
                        diff_bias[oc] += diff_dst[mb * OC + oc];
                    }
                }
            }
        });
    }
}

template struct gemm_inner_product_fwd_t<data_type::f32>;
template struct gemm_inner_product_bwd_data_t<data_type::f32>;
template struct gemm_inner_product_bwd_weights_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
