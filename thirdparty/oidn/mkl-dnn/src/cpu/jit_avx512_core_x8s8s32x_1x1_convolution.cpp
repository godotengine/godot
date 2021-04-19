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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

#include "jit_avx512_core_x8s8s32x_1x1_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

namespace {
template <typename T, typename U>
void balance2D(U nthr, U ithr, T ny, T &ny_start, T &ny_end,
    T nx, T &nx_start, T &nx_end, T nx_divider)
{
    const T grp_size = utils::div_up(nthr, nx_divider);
    const T grp_count = utils::div_up(nthr, grp_size);

    T grp = ithr / grp_size;
    T grp_ithr = ithr % grp_size;
    T grp_nthr = grp_size;
    T first_grps = nthr % grp_count;
    if (first_grps > 0 && grp >= first_grps) {
        ithr -= first_grps * grp_size;
        grp_nthr--;
        grp = ithr / grp_nthr + first_grps;
        grp_ithr = ithr % grp_nthr;
    }
    balance211(nx, grp_count, grp, nx_start, nx_end);
    balance211(ny, grp_nthr, grp_ithr, ny_start, ny_end);
}
}

/* convolution forward */
template <data_type_t src_type, data_type_t dst_type>
void jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<src_type, dst_type>::
execute_forward(const exec_ctx_t &ctx) const
{
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    auto scratchpad = this->scratchpad(ctx);

    if (pd()->jcp_.signed_input && pd()->jcp_.ver != ver_vnni) {
        auto local_scales = scratchpad.template get<float>(
                key_conv_adjusted_scales);
        auto scales = pd()->attr()->output_scales_.scales_;
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, scales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = scales[c] * factor;
        }
    }

    parallel(kernel_->jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src, weights, bias, dst, scratchpad);
    });
}

template <data_type_t src_type, data_type_t dst_type>
void jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<src_type, dst_type>
::execute_forward_thr(const int ithr, const int nthr, const src_data_t *src,
        const wei_data_t *weights, const char *bias, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));

    const size_t bia_dt_size = pd()->with_bias()
        ? types::data_type_size(pd()->desc()->bias_desc.data_type) : 0;

    const auto &jcp = kernel_->jcp;
    auto rtus_space = scratchpad.get<src_data_t>(key_conv_rtus_space);
    auto local_scales = scratchpad.get<float>(key_conv_adjusted_scales);

    const int work_amount = jcp.mb * jcp.ngroups * jcp.nb_bcast;

    const int stride_h = pd()->desc()->strides[0];
    const int stride_w = pd()->desc()->strides[1];
    const int pad_t = pd()->desc()->padding[0][0];
    const int pad_l = pd()->desc()->padding[0][1];

    const auto &oscales = pd()->attr()->output_scales_;

    int offset = jcp.ngroups * (jcp.oc / jcp.oc_block) * (jcp.ic / jcp.ic_block)
        * jcp.oc_block * jcp.ic_block;
    wei_data_t *w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input)
        ? reinterpret_cast<int32_t *>(w + offset) : 0;

    auto step = [](int default_step, int remaining, int tail_step) {
        assert(default_step <= tail_step);
        return remaining < tail_step ? remaining : default_step;
    };

    auto p = jit_1x1_conv_call_s();

    auto rp = rtus_driver_t<avx512_common>::call_params_t();
    const int nb_oc = jcp.nb_load;
    const int os_block = jcp.bcast_block;

    int bcast_start{0}, bcast_end{0}, ocb_start{0}, ocb_end{0};
    balance2D(nthr, ithr, work_amount, bcast_start, bcast_end,
        jcp.nb_load / jcp.nb_load_chunk, ocb_start, ocb_end,
        jcp.load_grp_count);
    if (jcp.nb_load_chunk > 1) {
        ocb_start *= jcp.nb_load_chunk;
        ocb_end *= jcp.nb_load_chunk;
    }

    auto init_bcast = [&](int iwork, int &n, int &g, int &bcast_step,
            int &oh, int &ow, int &ih, int &iw)
    {
        int osb{0};
        nd_iterator_init(iwork, n, jcp.mb, g, jcp.ngroups, osb,
            jcp.nb_bcast);
        bcast_step = step(jcp.nb_bcast_blocking, jcp.nb_bcast - osb,
                jcp.nb_bcast_blocking_max);
        bcast_step = nstl::min(bcast_step, bcast_end - iwork);

        const int os = osb * os_block;
        oh = os / jcp.ow;
        ow = os % jcp.ow;

        ih = nstl::max(oh * stride_h - pad_t, 0);
        iw = nstl::max(ow * stride_w - pad_l, 0);
        rp.iw_start = iw;

        p.bcast_dim = this_block_size(os, jcp.os,
            bcast_step * os_block);
        rp.os = p.bcast_dim;
    };

    auto init_load = [&](int ocb, int &load_step)
    {
        load_step = step(jcp.nb_load_blocking, ocb_end - ocb,
            jcp.nb_load_blocking_max);
        p.load_dim = this_block_size(ocb * jcp.oc_block,
            ocb_end * jcp.oc_block, load_step * jcp.oc_block);

        if (ocb + load_step >= nb_oc)
            p.first_last_flag |= FLAG_OC_LAST;
        else
            p.first_last_flag &= ~FLAG_OC_LAST;

    };

    auto init_reduce = [&]()
    {
        p.reduce_dim = this_block_size(0, jcp.ic, jcp.ic);
        rp.icb = p.reduce_dim / jcp.reduce_block;
    };

    auto inner_ker = [&](int ocb, int n, int g, int oh, int ow,
            int ih, int iw)
    {
        const int icb = 0; // Start from the first IC block
        const int _ocb = g * nb_oc + ocb;
        const int _icb = g;

        const size_t dst_off = dst_d.blk_off(n, _ocb * jcp.oc_block, oh, ow);

        p.output_data = &dst[dst_off];
        p.load_data = &weights[pd()->with_groups()
            ? weights_d.blk_off(g, ocb, icb)
            : weights_d.blk_off(ocb, icb)];
        p.bias_data = &bias[_ocb * jcp.oc_block * bia_dt_size];
        p.compensation = (jcp.signed_input)
            ? &compensation[_ocb * jcp.oc_block] : 0;
        p.scales = (jcp.signed_input && jcp.ver != ver_vnni)
            ? &local_scales[jcp.is_oc_scale * _ocb * jcp.oc_block]
            : &oscales.scales_[jcp.is_oc_scale * _ocb * jcp.oc_block];
        if (pd()->rtus_.reduce_src_) {
            rp.ws = rtus_space + ithr * pd()->rtus_.space_per_thread_
                + _icb * jcp.is * jcp.ic_block;
            if (ocb == ocb_start) {
                rp.src = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);
                rtus_driver_->ker_(&rp);
            }
            p.bcast_data = rp.ws;
        } else
            p.bcast_data = src + src_d.blk_off(n, _icb * jcp.ic_block, ih, iw);

        kernel_->jit_ker(&p);
    };

    if (jcp.loop_order == loop_rlb) {
        init_reduce();
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_lbr) {
        int ocb = ocb_start;
        while (ocb < ocb_end) {
            int load_step;
            init_load(ocb, load_step);
            int iwork = bcast_start;
            while (iwork < bcast_end) {
                int n, g, bcast_step, oh, ow, ih, iw;
                init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                iwork += bcast_step;
            }
            ocb += load_step;
        }
    } else if (jcp.loop_order == loop_rbl) {
        init_reduce();
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else if (jcp.loop_order == loop_blr) {
        int iwork = bcast_start;
        while (iwork < bcast_end) {
            int n, g, bcast_step, oh, ow, ih, iw;
            init_bcast(iwork, n, g, bcast_step, oh, ow, ih, iw);
            int ocb = ocb_start;
            while (ocb < ocb_end) {
                int load_step;
                init_load(ocb, load_step);
                init_reduce();
                inner_ker(ocb, n, g, oh, ow, ih, iw);
                ocb += load_step;
            }
            iwork += bcast_step;
        }
    } else {
        assert(!"unsupported loop order");
    }
}

using namespace data_type;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, u8>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, u8>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s8>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s8>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, s32>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, s32>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<u8, f32>;
template struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<s8, f32>;

}
}
}
