/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#include "c_types_map.hpp"
#include "jit_sse42_convolution.hpp"
#include "mkldnn_thread.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

#define src_blk_off(f, n, c, h, w) \
    (pd()->ndims() == 3) \
    ? (f).blk_off(n, c, w) \
    : (f).blk_off(n, c, h, w)

#define wht_blk_off_(f, g, ...) \
    pd()->with_groups() \
    ? (f).blk_off(g, __VA_ARGS__) \
    : (f).blk_off(__VA_ARGS__)
#define wht_blk_off(f, g, oc, ic, kh, kw) \
        pd()->ndims() == 3 \
        ? wht_blk_off_(f, g, oc, ic, kw) \
        : wht_blk_off_(f, g, oc, ic, kh, kw)

void jit_sse42_convolution_fwd_t::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const data_t *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const auto &jcp = kernel_->jcp;

    int ocb_work = div_up(jcp.nb_oc, jcp.nb_oc_blocking);
    const size_t work_amount = jcp.mb * jcp.ngroups * ocb_work * jcp.oh;

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{ 0 }, end{ 0 };
        balance211(work_amount, nthr, ithr, start, end);

        int icbb = 0;
        while (icbb < jcp.nb_ic) {
            int icb_step = jcp.nb_ic_blocking;
            int icb_step_rem = jcp.nb_ic - icbb;
            if (icb_step_rem < jcp.nb_ic_blocking_max)
                icb_step = icb_step_rem;

            size_t n{0}, g{0}, ocbb{0}, oh{0};
            nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                             oh, jcp.oh);
            for (size_t iwork = start; iwork < end; ++iwork) {
                int ocb = ocbb * jcp.nb_oc_blocking;
                int ocb_num = jcp.nb_oc_blocking;

                for (int icb = icbb; icb < icbb + icb_step; ++icb) {
                    auto par_conv = jit_conv_call_s();

                    const int ij = oh * jcp.stride_h;
                    const int i_t_overflow = nstl::max(0, jcp.t_pad - ij);
                    const int i_b_overflow = nstl::max(jcp.ih, ij
                        + (jcp.kh-1) * (jcp.dilate_h+1) - jcp.t_pad+1) - jcp.ih;

                    const size_t _oc = g * jcp.nb_oc + ocb;
                    const size_t _ic = g * jcp.nb_ic + icb;

                    const int ih = nstl::max(ij - jcp.t_pad
                        + div_up(i_t_overflow,
                                 (jcp.dilate_h+1)) * (jcp.dilate_h + 1), 0);
                    par_conv.src = &src[src_blk_off(src_d, n,
                        jcp.ic == 3 ? 0 : _ic, ih, 0)];

                    par_conv.dst = &dst[src_blk_off(dst_d, n, _oc, oh, 0)];

                    const int wh = div_up(i_t_overflow, (jcp.dilate_h + 1));
                    par_conv.filt = &weights[wht_blk_off(weights_d, g, ocb,
                        jcp.ic == 3 ? 0 : icb, wh, 0)];

                    if (icb == 0) {
                        if (bias)
                            par_conv.bias =
                                    &bias[bias_d.blk_off(_oc * jcp.oc_block)];
                        par_conv.flags |= FLAG_IC_FIRST;
                    }

                    if (jcp.with_eltwise && icb + 1 == jcp.nb_ic) {
                        par_conv.flags |= FLAG_IC_LAST;
                    }

                    par_conv.oc_blocks =
                            nstl::min(ocb + ocb_num, jcp.nb_oc) - ocb;

                    par_conv.kw_padding = 0;
                    const int kh_padding = jcp.kh
                        - div_up(i_t_overflow, (jcp.dilate_h + 1))
                        - div_up(i_b_overflow, (jcp.dilate_h + 1));
                    par_conv.kh_padding = nstl::max(0, kh_padding);
                    kernel_->jit_ker(&par_conv);
                }
                nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ocbb, ocb_work,
                                 oh, jcp.oh);
            }
            icbb += icb_step;
        }
    });

    if (pd()->wants_zero_pad_dst())
        ctx.memory(MKLDNN_ARG_DST)->zero_pad();
}

}
}
}
