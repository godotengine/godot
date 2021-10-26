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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_x8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;

using namespace nstl;

using jit_conv_ker_t = void (*)(jit_conv_call_s *);

#define wht_blk_off(d, g, ...) \
        (pd()->with_groups() \
         ? (d).blk_off((g), __VA_ARGS__) \
         : (d).blk_off(__VA_ARGS__))

template <data_type_t src_type, data_type_t dst_type>
void jit_avx512_core_x8s8s32x_convolution_fwd_t<src_type,
        dst_type>::execute_forward_1d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
        ? types::data_type_size(pd()->desc()->bias_desc.data_type) : 0;

    const auto &jcp = pd()->jcp_;
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
    assert(jcp.nb_ch % jcp.nb_ch_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = scratchpad(ctx).template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }

    size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input)
                                ? reinterpret_cast<int32_t *>(&w[offset]) : 0;
    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking;
    int nb_groups = jcp.nb_ch / jcp.nb_ch_blocking;
    int group_block = jcp.ch_block;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.nb_ow;

    parallel(0, [&](const int ithr, const int nthr) {

        int start{ 0 }, end{ 0 };
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();

        int n{ 0 }, gg{ 0 }, occ{ 0 }, owb{ 0 };
        switch (jcp.loop_order) {
        case loop_cwgn:
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, gg,
                    nb_groups, n, jcp.mb);
            break;
        case loop_gncw:
            nd_iterator_init(start, gg, nb_groups, n, jcp.mb, occ, oc_chunks,
                    owb, jcp.nb_ow);
            break;
        case loop_ngcw:
            nd_iterator_init(start, n, jcp.mb, gg, nb_groups, occ, oc_chunks,
                    owb, jcp.nb_ow);
            break;
        case loop_nwcg:
            nd_iterator_init(start, n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks,
                    gg, nb_groups);
            break;
        default: assert(!"unsupported loop order");
        }
        while (start < end) {
            int ocb = occ * jcp.nb_oc_blocking;
            int gb = gg * jcp.nb_ch_blocking;
            int g = gb * group_block;
            int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;
            int g_ic = g * jcp.nb_ic * jcp.ic_block;
            int ow_s = owb * jcp.ow_block;
            int iw_s = ow_s * jcp.stride_w;

            p.bias = bias ? bias + (bias_d.blk_off(g_oc) * bia_dt_size) : 0;
            p.compensation = (jcp.signed_input) ? compensation + g_oc : 0;
            p.dst = dst + dst_d.blk_off(n, g_oc, ow_s);
            p.src = src + src_d.blk_off(n, g_ic, iw_s);
            p.filt = weights + wht_blk_off(weights_d, gb, ocb, 0);
            p.scales = &oscales[jcp.is_oc_scale * g_oc];
            p.oc_blocks = jcp.is_depthwise ? gb : ocb;
            p.kh_padding = jcp.kh;
            p.t_overflow = 0;
            p.b_overflow = 0;
            p.owb = owb;

            kernel_->jit_ker(&p);

            ++start;
            switch (jcp.loop_order) {
            case loop_cwgn:
                nd_iterator_step(occ, oc_chunks, owb, jcp.nb_ow, gg, nb_groups,
                        n, jcp.mb);
                break;
            case loop_gncw:
                nd_iterator_step(gg, nb_groups, n, jcp.mb, occ, oc_chunks, owb,
                        jcp.nb_ow);
                break;
            case loop_ngcw:
                nd_iterator_step(n, jcp.mb, gg, nb_groups, occ, oc_chunks, owb,
                        jcp.nb_ow);
                break;
            case loop_nwcg:
                nd_iterator_step(n, jcp.mb, owb, jcp.nb_ow, occ, oc_chunks, gg,
                        nb_groups);
                break;
            default: assert(!"unsupported loop order");
            }
        }
    });
}

template <data_type_t src_type, data_type_t dst_type>
void jit_avx512_core_x8s8s32x_convolution_fwd_t<src_type,
        dst_type>::execute_forward_2d(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
        ? types::data_type_size(pd()->desc()->bias_desc.data_type) : 0;

    const auto &jcp = pd()->jcp_;
    assert(jcp.ch_block == 1);
    assert(jcp.nb_ch_blocking == 1);
    assert(jcp.nb_oc % jcp.nb_oc_blocking == 0);
    assert(jcp.nb_ch % jcp.nb_ch_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = scratchpad(ctx).template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }

    size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input)
                                ? reinterpret_cast<int32_t *>(&w[offset]) : 0;
    int oc_chunks = jcp.nb_oc / jcp.nb_oc_blocking_thr_chunk;
    int nb_groups = jcp.nb_ch;
    int work_amount = jcp.mb * nb_groups * oc_chunks * jcp.oh * jcp.nb_ow;

    parallel(0, [&](const int ithr, const int nthr) {

        int start{0}, end{0};
        balance211(work_amount, nthr, ithr, start, end);

        auto p = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t dst_h_stride = dst_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int n{ 0 }, g{ 0 }, occ{ 0 }, oh_s{ 0 }, owb{ 0 };
        switch (jcp.loop_order) {
        case loop_cwgn:
            nd_iterator_init(start, occ, oc_chunks, owb, jcp.nb_ow, g,
                    nb_groups, n, jcp.mb, oh_s, jcp.oh);
            break;
        case loop_ngcw:
            nd_iterator_init(start, n, jcp.mb, g, nb_groups, occ, oc_chunks,
                    owb, jcp.nb_ow, oh_s, jcp.oh);
            break;
        case loop_nhwcg:
            nd_iterator_init(start, n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow,
                    occ, oc_chunks, g, nb_groups);
            break;
        default: assert(!"unsupported loop order");
        }
        while (start < end) {
            for (int occ1 = 0; occ1 < jcp.nb_oc_blocking_thr_chunk;
                occ1 += jcp.nb_oc_blocking) {
                int ocb = occ * jcp.nb_oc_blocking_thr_chunk + occ1;
                int g_oc = (g * jcp.nb_oc + ocb) * jcp.oc_block;

                int g_ic = g * jcp.nb_ic * jcp.ic_block;

                int work_rem = end - start;
                int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
                int oh_e = oh_s + work_rem > jcp.oh ? jcp.oh : oh_s + work_rem;
                if (jcp.loop_order == loop_nhwcg)
                    oh_e = oh_s + 1; // step instead
                int ow_s = owb * jcp.ow_block;
                int iw_s = ow_s * jcp.stride_w;

                auto bias_w = bias
                    ? bias + (bias_d.blk_off(g_oc) * bia_dt_size)
                    : 0;
                int32_t *compensation_w = (jcp.signed_input)
                                          ? compensation + g_oc : 0;

                auto dst_w = dst + dst_d.blk_off(n, g_oc, oh_s, ow_s);
                auto src_w = src + src_d.blk_off(n, g_ic, ih_s, iw_s);
                auto wht_w = weights + wht_blk_off(weights_d, g, ocb, 0);

                auto scales = &oscales[jcp.is_oc_scale * g_oc];

                for (int oj = oh_s, ij = ih_s; oj < oh_e;
                    ++oj, ij += jcp.stride_h) {
                    int dilate_h = jcp.dilate_h + 1;
                    int i_t_overflow = nstl::min(jcp.kh,
                                                div_up(max(0, -ij), dilate_h));
                    int i_b_overflow = nstl::min(jcp.kh, div_up(
                            max(0, ij - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                            dilate_h));
                    int kh_padding = nstl::max(0,
                        jcp.kh - i_t_overflow - i_b_overflow);

                    size_t wei_stride = (!jcp.signed_input)
                                            ? i_t_overflow * wht_h_stride : 0;
                    p.src = src_w + i_t_overflow * dilate_h * src_h_stride;
                    p.dst = dst_w;
                    p.filt = wht_w + wei_stride;
                    p.bias = bias_w;
                    p.compensation = compensation_w;
                    p.oc_blocks = ocb;
                    p.kh_padding = kh_padding;
                    p.scales = scales;
                    p.t_overflow = i_t_overflow;
                    p.b_overflow = i_b_overflow;
                    p.owb = owb;

                    kernel_->jit_ker(&p);
                    src_w += src_h_stride * jcp.stride_h;
                    dst_w += dst_h_stride;
                }
            }
            switch (jcp.loop_order) {
            case loop_cwgn:
                nd_iterator_jump(start, end, occ, oc_chunks, owb, jcp.nb_ow, g,
                        nb_groups, n, jcp.mb, oh_s, jcp.oh);
                break;
            case loop_ngcw:
                nd_iterator_jump(start, end, n, jcp.mb, g, nb_groups, occ,
                        oc_chunks, owb, jcp.nb_ow, oh_s, jcp.oh);
                break;
            case loop_nhwcg:
                ++start;
                nd_iterator_step(n, jcp.mb, oh_s, jcp.oh, owb, jcp.nb_ow, occ,
                        oc_chunks, g, nb_groups);
                break;
            default: assert(!"unsupported loop order");
            }
        }
    });
}

template <data_type_t src_type, data_type_t dst_type>
void jit_avx512_core_x8s8s32x_convolution_fwd_t<src_type,
        dst_type>::execute_forward_2d_dw(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const size_t bia_dt_size = pd()->with_bias()
            ? types::data_type_size(pd()->desc()->bias_desc.data_type) : 0;

    const auto &jcp = pd()->jcp_;
    assert(jcp.ic_block == 1);
    assert(jcp.oc_block == 1);
    assert(jcp.nb_ic == 1);
    assert(jcp.nb_oc == 1);
    assert(jcp.nb_oc_blocking == 1);
    assert(jcp.nb_ch % jcp.nb_ch_blocking == 0);

    const float *oscales = pd()->attr()->output_scales_.scales_;
    if (jcp.signed_input && jcp.ver != ver_vnni) {
        auto local_scales = scratchpad(ctx).template get<float>(
                key_conv_adjusted_scales);
        size_t count = pd()->attr()->output_scales_.count_;
        float factor = 1.f / pd()->jcp_.wei_adj_scale;
        if (count == 1) {
            utils::array_set(local_scales, oscales[0] * factor, 16);
        } else {
            for (size_t c = 0; c < count; c++)
                local_scales[c] = oscales[c] * factor;
        }
        oscales = local_scales;
    }

    size_t offset = weights_d.size() - weights_d.additional_buffer_size();
    auto w = const_cast<wei_data_t *>(weights);
    int32_t* compensation = (jcp.signed_input)
                                ? reinterpret_cast<int32_t *>(&w[offset]) : 0;
    int nb_groups = jcp.nb_ch / jcp.nb_ch_blocking;
    int group_block = jcp.ch_block;

    parallel_nd(jcp.mb, jcp.oh, jcp.nb_ow, nb_groups,
            [&](int n, int oh_s, int owb, int gg) {

        auto p = jit_conv_call_s();

        size_t src_h_stride = src_d.blk_off(0, 0, 1);
        size_t wht_h_stride = wht_blk_off(weights_d, 0, 0, 0, 1);

        int gb = gg * jcp.nb_ch_blocking;
        int g = gb * group_block;

        int ih_s = -jcp.t_pad + oh_s * jcp.stride_h;
        int ow_s = owb * jcp.ow_block;
        int iw_s = ow_s * jcp.stride_w;

        auto bias_w = bias ? bias + (bias_d.blk_off(g) * bia_dt_size) : 0;
        int32_t *compensation_w = jcp.signed_input ? compensation + g : 0;

        auto dst_w = dst + dst_d.blk_off(n, g, oh_s, ow_s);
        auto src_w = src + src_d.blk_off(n, g, ih_s, iw_s);
        auto wht_w = weights + wht_blk_off(weights_d, gb, 0);

        auto scales = &oscales[jcp.is_oc_scale * g];

        int dilate_h = jcp.dilate_h + 1;
        int i_t_overflow = nstl::min(jcp.kh, div_up(max(0, -ih_s), dilate_h));
        int i_b_overflow = nstl::min(jcp.kh,
                div_up(max(0, ih_s - jcp.ih + (jcp.kh - 1) * dilate_h + 1),
                                             dilate_h));
        int kh_padding = nstl::max(0, jcp.kh - i_t_overflow - i_b_overflow);

        size_t wei_stride = jcp.signed_input ? 0 : i_t_overflow * wht_h_stride;
        p.src = src_w + i_t_overflow * dilate_h * src_h_stride;
        p.dst = dst_w;
        p.filt = wht_w + wei_stride;
        p.bias = bias_w;
        p.compensation = compensation_w;
        p.oc_blocks = gb;
        p.kh_padding = kh_padding;
        p.scales = scales;
        p.t_overflow = i_t_overflow;
        p.b_overflow = i_b_overflow;
        p.owb = owb;

        kernel_->jit_ker(&p);
    });
}

template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::s8, data_type::u8>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::u8, data_type::u8>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::s8, data_type::s8>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::u8, data_type::s8>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::s8, data_type::s32>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::u8, data_type::s32>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::s8, data_type::f32>;
template struct jit_avx512_core_x8s8s32x_convolution_fwd_t<
                                                data_type::u8, data_type::f32>;
}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
