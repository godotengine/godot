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

#include <assert.h>
#include <math.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "nstl.hpp"

#include "nchw_pooling.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
void nchw_pooling_fwd_t<data_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    using namespace alg_kind;

    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(unsigned char *, MKLDNN_ARG_WORKSPACE);

    const memory_desc_wrapper ws_d(pd()->workspace_md());
    const data_type_t ws_dt = ws ? ws_d.data_type() : data_type::undef;

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    auto alg = pd()->desc()->alg_kind;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto set_ws = [=](int mb, int c, int od, int oh, int ow, int value) {
        if (ws) {
            assert(ws_dt == data_type::u8 || ws_dt == data_type::s32);
            size_t ws_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            if (ws_dt == data_type::u8) {
                assert(0 <= value && value <= 255);
                ws[ws_offset] = value;
            } else
                reinterpret_cast<int *>(ws)[ws_offset] = value;
        }
    };

    auto ker_max = [=](data_t *d, int mb, int c, int od, int oh, int ow) {
        for (int kd = 0; kd < KD; ++kd) {
            for (int kh = 0; kh < KH; ++kh) {
                for (int kw = 0; kw < KW; ++kw) {
                    const int id = od * SD - padF + kd;
                    const int ih = oh * SH - padT + kh;
                    const int iw = ow * SW - padL + kw;

                    if (id < 0 || id >= ID) continue;
                    if (ih < 0 || ih >= IH) continue;
                    if (iw < 0 || iw >= IW) continue;

                    auto src_offset
                        = (size_t)IW * IH * ID * C * mb
                        + (size_t)IW * IH * ID * c
                        + (size_t)IW * IH * id
                        + (size_t)IW * ih
                        + (size_t)iw;
                    auto s = src[src_offset];
                    if (s > d[0]) {
                        d[0] = s;
                        set_ws(mb, c, od, oh, ow, kd*KH*KW + kh*KW + kw);
                    }
                }
            }
        }
    };

    auto ker_avg = [=](data_t *d, int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        auto num_summands = (alg == pooling_avg_include_padding) ? KD*KW*KH
            : (id_end - id_start)*(ih_end - ih_start)*(iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    auto src_offset
                        = (size_t)IW * IH * ID * C * mb
                        + (size_t)IW * IH * ID * c
                        + (size_t)IW * IH * id
                        + (size_t)IW * ih
                        + (size_t)iw;
                    d[0] += src[src_offset];
                }
            }
        }

        d[0] = math::out_round<data_t>((float)d[0] / num_summands);
    };


    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {
            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            data_t *d = &dst[dst_offset];
            d[0] = nstl::numeric_limits<data_t>::lowest();
            set_ws(mb, c, od, oh, ow, 0);
            ker_max(d, mb, c, od, oh, ow);
        });
    } else {
        parallel_nd(MB, C, OD, OH, OW,
            [&](int mb, int c, int od, int oh, int ow) {
            size_t dst_offset
                = (size_t)OW * OH * OD * C * mb
                + (size_t)OW * OH * OD * c
                + (size_t)OW * OH * od
                + (size_t)OW * oh
                + (size_t)ow;
            data_t *d = &dst[dst_offset];
            d[0] = 0;
            ker_avg(d, mb, c, od, oh, ow);
        });
    }
}

template <impl::data_type_t data_type>
void nchw_pooling_bwd_t<data_type>::execute_backward(
        const exec_ctx_t &ctx) const {
    using namespace alg_kind;

    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const unsigned char *, MKLDNN_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const memory_desc_wrapper ws_d(pd()->workspace_md());

    const int MB = pd()->MB();
    const int C = pd()->C();
    const int OD = pd()->OD();
    const int OH = pd()->OH();
    const int OW = pd()->OW();
    const int ID = pd()->ID();
    const int IH = pd()->IH();
    const int IW = pd()->IW();
    const int KD = pd()->KD();
    const int KH = pd()->KH();
    const int KW = pd()->KW();
    const int SD = pd()->KSD();
    const int SH = pd()->KSH();
    const int SW = pd()->KSW();
    const int padF = pd()->padFront();
    const int padT = pd()->padT();
    const int padL = pd()->padL();

    const bool is_3d = pd()->desc()->diff_src_desc.ndims == 5;

    auto alg = pd()->desc()->alg_kind;

    auto apply_offset = [=](int index, int offset) {
        return (index > offset) ? index - offset : 0;
    };

    auto ker_zero = [=](int mb, int c) {
        size_t diff_src_offset = (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW;
        for (int id = 0; id < ID; ++id) {
            for (int ih = 0; ih < IH; ++ih) {
                for (int iw = 0; iw < IW; ++iw) {
                    diff_src[diff_src_offset++] = 0;
                }
            }
        }
    };

    auto ker_max = [=](const data_t *d, int mb, int c, int od, int oh, int ow) {
        auto b_c = ws_d.blocking_desc().inner_nblks == 0
            ? 1 : ws_d.blocking_desc().inner_blks[0];
        auto ws_offset = is_3d
            ? ws_d.blk_off(mb, c / b_c, od, oh, ow) + c % b_c
            : ws_d.blk_off(mb, c / b_c, oh, ow) + c % b_c;

        const int index = ws_d.data_type() == data_type::u8
            ? (int)ws[ws_offset] : ((const int *)ws)[ws_offset];
        const int kw = index % KW;
        const int kh = (index / KW) % KH;
        const int kd = (index / KW) / KH;

        const int id = od * SD - padF + kd;
        const int ih = oh * SH - padT + kh;
        const int iw = ow * SW - padL + kw;

        // If padding area could fit the kernel,
        // then input displacement would be out of bounds.
        // No need to back propagate there as padding is
        // virtual in pooling_max case.
        if (id < 0 || id >= ID)
            return;
        if (ih < 0 || ih >= IH)
            return;
        if (iw < 0 || iw >= IW)
            return;

        size_t diff_src_offset =
            (size_t)mb*C*ID*IH*IW + (size_t)c*ID*IH*IW + (size_t)id*IH*IW
            + (size_t)ih*IW + (size_t)iw;
        diff_src[diff_src_offset] += d[0];
    };

    auto ker_avg = [=](const data_t *d, int mb, int c, int od, int oh, int ow) {
        auto id_start = apply_offset(od*SD, padF);
        auto ih_start = apply_offset(oh*SH, padT);
        auto iw_start = apply_offset(ow*SW, padL);
        auto id_end = nstl::min(od*SD - padF + KD, ID);
        auto ih_end = nstl::min(oh*SH - padT + KH, IH);
        auto iw_end = nstl::min(ow*SW - padL + KW, IW);

        size_t num_summands = (alg == pooling_avg_include_padding)
            ? (size_t)KW*KH*KD
            : (size_t)(id_end - id_start)*(ih_end - ih_start)
                *(iw_end - iw_start);

        for (int id = id_start; id < id_end; ++id) {
            for (int ih = ih_start; ih < ih_end; ++ih) {
                for (int iw = iw_start; iw < iw_end; ++iw) {
                    size_t diff_src_offset = (size_t)mb*C*ID*IH*IW
                        + (size_t)c*ID*IH*IW + (size_t)id*IH*IW
                        + (size_t)ih*IW + (size_t)iw;
                    diff_src[diff_src_offset] += d[0] / num_summands;
                }
            }
        }
    };

    if (pd()->desc()->alg_kind == pooling_max) {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW
                + (size_t)c*OD*OH*OW;
            ker_zero(mb, c);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = &diff_dst[diff_dst_offset++];
                        ker_max(d, mb, c, od, oh, ow);
                    }
                }
            }
        });
    } else {
        parallel_nd(MB, C, [&](int mb, int c) {
            size_t diff_dst_offset = (size_t)mb*C*OD*OH*OW
                + (size_t)c*OD*OH*OW;
            ker_zero(mb, c);
            for (int od = 0; od < OD; ++od) {
                for (int oh = 0; oh < OH; ++oh) {
                    for (int ow = 0; ow < OW; ++ow) {
                        const data_t *d = &diff_dst[diff_dst_offset++];
                        ker_avg(d, mb, c, od, oh, ow);
                    }
                }
            }
        });
    }
}

template struct nchw_pooling_fwd_t<data_type::f32>;
template struct nchw_pooling_bwd_t<data_type::f32>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
