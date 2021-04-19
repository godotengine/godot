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
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "mkldnn_traits.hpp"
#include "type_helpers.hpp"

#include "ref_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using math::saturate;
using math::get_bias;

template <data_type_t src_type, data_type_t wei_type,
         data_type_t dst_type, data_type_t acc_type>
void ref_convolution_fwd_t<src_type, wei_type, dst_type, acc_type>::
execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD();
    const int KDH = pd()->KDH();
    const int KDW = pd()->KDW();

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool with_relu = 0; // TODO: change if support post_ops
    const float nslope = 0.f;

    const int ndims = pd()->desc()->src_desc.ndims;

    auto ker = [=](int g, int mb, int oc, int od, int oh,
            int ow) {
        acc_data_t d = 0;
        for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            const int id = od * KSD - padFront + kd * (1 + KDD);
            const int ih = oh * KSH - padT + kh * (1 + KDH);
            const int iw = ow * KSW - padL + kw * (1 + KDW);

            if (id < 0 || id >= ID) continue;
            if (ih < 0 || ih >= IH) continue;
            if (iw < 0 || iw >= IW) continue;

            if (ndims == 5)
                d += (acc_data_t)src[src_d.off(mb, g*IC + ic, id, ih, iw)]
                    * (with_groups
                    ? weights[weights_d.off(g, oc, ic, kd, kh, kw)]
                    : weights[weights_d.off(oc, ic, kd, kh, kw)]);
            else if (ndims == 4)
                d += (acc_data_t)src[src_d.off(mb, g*IC + ic, ih, iw)]
                    * (with_groups
                    ? weights[weights_d.off(g, oc, ic, kh, kw)]
                    : weights[weights_d.off(oc, ic, kh, kw)]);
            else if (ndims == 3)
                d += (acc_data_t)src[src_d.off(mb, g*IC + ic, iw)]
                    * (with_groups
                    ? weights[weights_d.off(g, oc, ic, kw)]
                    : weights[weights_d.off(oc, ic, kw)]);
           else
               assert(false);

        }
        return d;
    };

    parallel_nd(G, MB, OC, OD, OH, OW,
        [&](int g, int mb, int oc, int od, int oh, int ow) {
        float a = bias
            ? get_bias(bias, bias_d.off(g * OC + oc),
                    pd()->desc()->bias_desc.data_type)
            : 0;
        a += ker(g, mb, oc, od, oh, ow);
        if (with_relu && a < 0)
            a = a * nslope;
        if (ndims == 5)
            dst[dst_d.off(mb, g*OC + oc, od, oh, ow)] = saturate<dst_data_t>(a);
        else if (ndims == 4)
            dst[dst_d.off(mb, g*OC + oc, oh, ow)] = saturate<dst_data_t>(a);
        else if (ndims == 3)
            dst[dst_d.off(mb, g*OC + oc, ow)] = saturate<dst_data_t>(a);
        else
            assert(false);
   });
}

template <data_type_t diff_src_type, data_type_t wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_data_t<diff_src_type, wei_type, diff_dst_type,
     acc_type>::execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, MKLDNN_ARG_DIFF_DST);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto diff_src = CTX_OUT_MEM(diff_src_data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_src_d(pd()->diff_src_md());
    const memory_desc_wrapper weights_d(pd()->weights_md(0));
    const memory_desc_wrapper bias_d(pd()->weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD();
    const int KDH = pd()->KDH();
    const int KDW = pd()->KDW();

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->diff_src_desc.ndims;

    auto ker = [=](int g, int mb, int ic, int id, int ih,
            int iw) {
        acc_data_t d = 0;
        for (int oc = 0; oc < OC; ++oc)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            if (iw + padL < kw * (1 + KDW)
                || ih + padT < kh * (1 + KDH)
                || id + padFront < kd * (1 + KDD))
                continue;
            int ow = iw - kw * (1 + KDW) + padL;
            int oh = ih - kh * (1 + KDH) + padT;
            int od = id - kd * (1 + KDD) + padFront;
            if (ow % KSW != 0 || oh % KSH != 0 || od % KSD != 0)
                continue;

            ow /= KSW;
            oh /= KSH;
            od /= KSD;

            if (od < OD && oh < OH && ow < OW) {
                if (ndims == 5)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, od, oh, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kd, kh, kw)]
                        : weights[weights_d.off(oc, ic, kd, kh, kw)]);
                else if (ndims == 4)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, oh, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kh, kw)]
                        : weights[weights_d.off(oc, ic, kh, kw)]);
                else if (ndims == 3)
                    d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC
                        + oc, ow)] * (with_groups
                        ? weights[weights_d.off(g, oc, ic, kw)]
                        : weights[weights_d.off(oc, ic, kw)]);
                else
                    assert(false);
            }
        }
        return d;
    };

    parallel_nd(G, MB, IC, ID, IH, IW,
        [&](int g, int mb, int ic, int id, int ih, int iw) {
        auto ds_idx = (ndims == 5)
            ? diff_src_d.off(mb, g*IC + ic, id, ih, iw)
            : (ndims == 4)
            ? diff_src_d.off(mb, g*IC + ic, ih, iw)
            : diff_src_d.off(mb, g*IC + ic, iw);
        float a = bias
            ? get_bias(bias, bias_d.off(g * IC + ic),
                    pd()->desc()->bias_desc.data_type)
            : 0;
        a += ker(g, mb, ic, id, ih, iw);
        diff_src[ds_idx] = saturate<diff_src_data_t>(a);
    });
}

template <data_type_t src_type, data_type_t diff_wei_type,
         data_type_t diff_dst_type, data_type_t acc_type>
void ref_convolution_bwd_weights_t<src_type, diff_wei_type, diff_dst_type,
     acc_type>::execute_backward_weights(const exec_ctx_t &ctx) const {
    auto diff_dst = CTX_IN_MEM(const diff_dst_data_t *, MKLDNN_ARG_DIFF_DST);
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto diff_weights = CTX_OUT_MEM(diff_wei_data_t *, MKLDNN_ARG_DIFF_WEIGHTS);
    auto diff_bias = CTX_OUT_MEM(diff_wei_data_t *, MKLDNN_ARG_DIFF_BIAS);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper diff_dst_d(pd()->diff_dst_md());
    const memory_desc_wrapper diff_weights_d(pd()->diff_weights_md(0));
    const memory_desc_wrapper diff_bias_d(pd()->diff_weights_md(1));

    const bool with_groups = pd()->with_groups();

    const int G = pd()->G();
    const int MB = pd()->MB();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();

    const int OC = pd()->OC() / G;
    const int IC = pd()->IC() / G;
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();

    const int KSD = pd()->KSD();
    const int KSH = pd()->KSH();
    const int KSW = pd()->KSW();

    const int KDD = pd()->KDD();
    const int KDH = pd()->KDH();
    const int KDW = pd()->KDW();

    const int padFront = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const int ndims = pd()->desc()->src_desc.ndims;

auto ker = [=](acc_data_t &d, int g, int oc, int ic, int kd, int kh, int kw) {
        for (int mb = 0; mb < MB; ++mb)
        for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ow*KSW + kw * (1 + KDW) < padL
                || oh*KSH + kh * (1 + KDH) < padT
                || od*KSD + kd * (1 + KDD) < padFront
                || ow*KSW + kw * (1 + KDW) >= IW + padL
                || oh*KSH + kh * (1 + KDH) >= IH + padT
                || od*KSD + kd * (1 + KDD) >= ID + padFront)
                continue;

            int id = od*KSD - padFront + kd * (1 + KDD);
            int ih = oh*KSH - padT + kh * (1 + KDH);
            int iw = ow*KSW - padL + kw * (1 + KDW);
            if (ndims == 5)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, od,
                    oh, ow)] * src[src_d.off(mb, g*IC + ic, id, ih, iw)];
            else if (ndims == 4)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, oh, ow)]
                    * src[src_d.off(mb, g*IC + ic, ih, iw)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, ow)]
                    * src[src_d.off(mb, g*IC + ic, iw)];
            else
                assert(false);
        }
    };

    auto ker_bias = [=](acc_data_t &d, int g, int oc) {
        for (int mb = 0; mb < MB; ++mb)
        for (int od = 0; od < OD; ++od)
        for (int oh = 0; oh < OH; ++oh)
        for (int ow = 0; ow < OW; ++ow) {
            if (ndims == 5)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, od, oh,
                     ow)];
            else if (ndims == 4)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, oh,
                     ow)];
            else if (ndims == 3)
                d += (acc_data_t)diff_dst[diff_dst_d.off(mb, g*OC + oc, ow)];
            else
                assert(false);
        }
    };

    parallel_nd(G, OC, [&](int g, int oc) {
        if (diff_bias) {
            // XXX: loss of precision when bias is a float...
            acc_data_t db = 0;
            ker_bias(db, g, oc);
            diff_bias[diff_bias_d.off(g*OC+oc)]
                = saturate<diff_wei_data_t>(db);
        }

        for (int ic = 0; ic < IC; ++ic)
        for (int kd = 0; kd < KD; ++kd)
        for (int kh = 0; kh < KH; ++kh)
        for (int kw = 0; kw < KW; ++kw) {
            acc_data_t dw = 0;
            ker(dw, g, oc, ic, kd, kh, kw);

            if (ndims == 5) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kd, kh, kw)
                    : diff_weights_d.off(oc, ic, kd, kh, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else if (ndims == 4) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kh, kw)
                    : diff_weights_d.off(oc, ic, kh, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else if (ndims == 3) {
                auto idx = with_groups
                    ? diff_weights_d.off(g, oc, ic, kw)
                    : diff_weights_d.off(oc, ic, kw);
                    diff_weights[idx] = saturate<diff_wei_data_t>(dw);
            } else {
                 assert(false);
            }
        }
    });
}

using namespace data_type;

template struct ref_convolution_fwd_t<f32>;

template struct ref_convolution_fwd_t<u8, s8, f32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s32, s32>;
template struct ref_convolution_fwd_t<u8, s8, s8, s32>;
template struct ref_convolution_fwd_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_data_t<f32, f32, f32, f32>;

template struct ref_convolution_bwd_data_t<f32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s32, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<s8, s8, u8, s32>;
template struct ref_convolution_bwd_data_t<u8, s8, u8, s32>;

template struct ref_convolution_bwd_weights_t<f32, f32, f32, f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
