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

#include "mkldnn_types.h"

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "cpu_isa_traits.hpp"

#include "gemm_convolution_utils.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;
using namespace prop_kind;
using namespace data_type;

namespace jit_gemm_convolution_utils {

void im2col_3d(const jit_gemm_conv_conf_t &jcp, const float *im, float *col,
        int od)
{
    const size_t OHW = jcp.oh * jcp.ow;
    const size_t im_step = jcp.ih * jcp.iw * jcp.id;
    const size_t col_step = jcp.ks * OHW;

    parallel_nd(jcp.ic, [&](int ic) {
        const float *__restrict im_loc = im + ic * im_step;
        float *__restrict col_loc = col + ic * col_step;
        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            float *__restrict col_ = col_loc + kd * jcp.kh * jcp.kw * OHW;
            if (id < 0 || id >= jcp.id) {
                int ih_ = -jcp.t_pad;
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    int ih = ih_;
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        if (ih < 0 || ih >= jcp.ih) {
                            ih += jcp.stride_h;
                            continue;
                        }
                        int iw_ = -jcp.l_pad;
                        for (int kw = 0; kw < jcp.kw; ++kw) {
                            int iw = iw_;
                            for (int ow = 0; ow < jcp.ow; ++ow) {
                                if (iw < 0 || iw >= jcp.iw) {
                                    iw += jcp.stride_w;
                                    continue;
                                }

                                const size_t col_idx = kw * OHW + oh * jcp.ow
                                    + ow;

                                col_[col_idx] = 0;
                                iw += jcp.stride_w;
                            }
                            iw_ += (1 + jcp.dilate_w);
                        }
                        ih += jcp.stride_h;
                    }
                    ih_ += (1 + jcp.dilate_h);
                    col_ += jcp.kw * OHW;
                }
            } else {
                const float *__restrict im_ = im_loc + id * jcp.ih * jcp.iw;
                int ih_ = -jcp.t_pad;
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    int ih = ih_;
                    for (int oh = 0; oh < jcp.oh; ++oh) {
                        if (ih < 0 || ih >= jcp.ih) {
                            ih += jcp.stride_h;
                            continue;
                        }
                        int iw_ = -jcp.l_pad;
                        for (int kw = 0; kw < jcp.kw; ++kw) {
                            int iw = iw_;
                            for (int ow = 0; ow < jcp.ow; ++ow) {
                                if (iw < 0 || iw >= jcp.iw) {
                                    iw += jcp.stride_w;
                                    continue;
                                }

                                const size_t col_idx = kw * OHW + oh * jcp.ow
                                    + ow;
                                const size_t im_idx = ih * jcp.iw + iw;

                                col_[col_idx] = im_[im_idx];
                                iw += jcp.stride_w;
                            }
                            iw_ += (1 + jcp.dilate_w);
                        }
                        ih += jcp.stride_h;
                    }
                    ih_ += (1 + jcp.dilate_h);
                    col_ += jcp.kw * OHW;
                }
            }
            id += (1 + jcp.dilate_d);
        }
    });
}

/* col[ic][kh][kw][oh][ow] <-- im2col(im[ic][ih][iw]) */
void im2col(const jit_gemm_conv_conf_t &jcp, const float *__restrict im,
       float *__restrict col, int hs, int hb, int ws, int wb) {
    const size_t im_step = jcp.is;
    const size_t col_step = jcp.ks * hb * wb;
    if (jcp.stride_w == 1) {
        // Generated code is more optimized for stride_w == 1
        // because innermost loop is by width
        auto ker = [&](int ic, int kh, int kw, int oh) {
            const float *__restrict im_ = im + ic * im_step;
            float *__restrict col_
                = col + ic * col_step + ((kh * jcp.kw + kw) * hb + oh) * wb;

            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) {
                for (int ow = 0; ow < wb; ++ow)
                    col_[ow] = 0.f;
            } else {
                for (int ow = 0; ow < wb; ++ow) {
                    const int iw = ow + ws - jcp.l_pad + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw)
                        col_[ow] = 0.f;
                    else {
                        const size_t im_idx = ih * jcp.iw + iw;
                        col_[ow] = im_[im_idx];
                    }
                }
            }
        };

        if (jcp.outer_threading) {
            for (int ic = 0; ic < jcp.ic; ic++)
                for (int kh = 0; kh < jcp.kh; kh++)
                    for (int kw = 0; kw < jcp.kw; kw++)
                        for (int oh = 0; oh < hb; oh++)
                            ker(ic, kh, kw, oh);
        }
        else {
            parallel_nd(jcp.ic, jcp.kh, jcp.kw, hb, ker);
        }
    } else if (jcp.ic == 1) {
        parallel_nd(jcp.kh, hb, [&](int kh, int oh) {
            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                    + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih)
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    for (int ow = 0; ow < wb; ++ow) {
                        const size_t col_idx
                                = ((kh * jcp.kw + kw) * hb + oh) * wb + ow;
                        col[col_idx] = 0;
                    }
                }
            else
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    for (int ow = 0; ow < wb; ++ow) {
                        const int iw = (ow + ws) * jcp.stride_w - jcp.l_pad
                                + kw * (1 + jcp.dilate_w);
                        const size_t col_idx
                                = ((kh * jcp.kw + kw) * hb + oh) * wb + ow;
                        const size_t im_idx = ih * jcp.iw + iw;
                        if (iw < 0 || iw >= jcp.iw)
                            col[col_idx] = 0;
                        else
                            col[col_idx] = im[im_idx];
                    }
                }
        });
    } else {

        parallel_nd(jcp.ic, jcp.kh, jcp.kw, hb,
            [&](int ic, int kh, int kw, int oh) {
            const float *__restrict im_ = im + ic * im_step;
            float *__restrict col_ = col + ic * col_step
                + ((kh * jcp.kw + kw) * hb + oh) * wb;

            const int ih = (oh + hs) * jcp.stride_h - jcp.t_pad
                + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) {
                for (int ow = 0; ow < wb; ++ow)
                    col_[ow] = 0.f;
            } else {
                for (int ow = 0; ow < wb; ++ow) {
                    const int iw = (ow + ws) * jcp.stride_w - jcp.l_pad
                        + kw * (1 + jcp.dilate_w);
                    const size_t im_idx = ih * jcp.iw + iw;
                    if (iw < 0 || iw >= jcp.iw)
                        col_[ow] = 0.f;
                    else
                        col_[ow] = im_[im_idx];
                }
            }
        });
    }
}

inline int limit(int low, int upper, int value) {
    return nstl::max(low, nstl::min(upper, value));
}

/* col[kh][kw][ic][oh][ow] <-- im2col_u8(im[ih][iw][ic]) */
template <typename T>
void im2col_u8(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        T *__restrict imtr, uint8_t *__restrict col, int hs, int hb, int ws,
        int wb) {
    uint8_t shift = jcp.signed_input ? 128 : 0;
    const int dh = 1 + jcp.dilate_h;
    const int dw = 1 + jcp.dilate_w;
    const int sh = jcp.stride_h;
    const int sw = jcp.stride_w;
    const int im_iw_stride = jcp.ic * jcp.ngroups;
    const int im_ih_stride = jcp.iw * im_iw_stride;
    const int tp = jcp.t_pad;
    const int lp = jcp.l_pad;

    if (jcp.outer_threading && sh == 1 && sw == 1 && dh == 1 && dw == 1) {
        /* im[ih][iw][ic] --> imtr[ic][ih][iw] --> col[kh][kw][ic][oh][ow] */
        const int hp = hs - tp;
        const int wp = ws - lp;
        const int ih_start = limit(0, jcp.ih, hp);
        const int ih_end = limit(0, jcp.ih, hp + hb + jcp.kh);
        const int iw_start = limit(0, jcp.iw, wp);
        const int iw_end = limit(0, jcp.iw, wp + wb + jcp.kw);

        const int ihb = ih_end - ih_start;
        const int iwb = iw_end - iw_start;

        const int imtr_ic_stride = ihb * iwb;
        const ptrdiff_t imtr_idx_shift = ih_start * iwb + iw_start;
        for (int ic = 0; ic < jcp.ic; ic++) {
            const ptrdiff_t imtr_idx_ic = ic * imtr_ic_stride - imtr_idx_shift;
            for (int ih = ih_start; ih < ih_end; ih++) {
                const ptrdiff_t im_idx_ih = ic + ih * im_ih_stride;
                const ptrdiff_t imtr_idx_ih = imtr_idx_ic + ih * iwb;
                for (int iw = iw_start; iw < iw_end; iw++)
                    imtr[imtr_idx_ih + iw] = im[im_idx_ih + iw * im_iw_stride];
            }
        }

        const int col_ic_str = hb * wb;
        const int col_kw_stride = jcp.ic * col_ic_str;
        const int col_kh_stride = jcp.kw * col_kw_stride;

        const int oh_init = ih_start - hp;
        const int ow_init = iw_start - wp;
        for (int kh = 0; kh < jcp.kh; kh++) {
            const ptrdiff_t col_idx_kh = kh * col_kh_stride;
            const int oh_kh = oh_init - kh;
            const int oh_start = limit(0, hb, oh_kh);
            const int oh_end = limit(0, hb, oh_kh + ihb);
            for (int kw = 0; kw < jcp.kw; kw++) {
                const ptrdiff_t col_idx_kw
                        = col_idx_kh + kw * jcp.ic * col_ic_str;
                const int ow_kw = ow_init - kw;
                const int imtr_shift = oh_kh * iwb + ow_kw;
                const int ow_start = limit(0, wb, ow_kw);
                const int ow_end = limit(0, wb, ow_kw + iwb);
                for (int ic = 0; ic < jcp.ic; ic++) {
                    const ptrdiff_t col_idx_ic = col_idx_kw + ic * col_ic_str;
                    const int imtr_idx_ic = ic * imtr_ic_stride - imtr_shift;
                    for (int oh = 0; oh < oh_start; oh++) {
                        const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                        for (int ow = 0; ow < wb; ++ow)
                            col[col_idx_oh + ow] = shift;
                    }
                    for (int oh = oh_start; oh < oh_end; oh++) {
                        const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                        const ptrdiff_t imtr_idx_oh = imtr_idx_ic + oh * iwb;
                        for (int ow = 0; ow < ow_start; ++ow)
                            col[col_idx_oh + ow] = shift;
                        for (int ow = ow_start; ow < ow_end; ++ow)
                            col[col_idx_oh + ow]
                                    = imtr[imtr_idx_oh + ow] + shift;
                        for (int ow = ow_end; ow < wb; ++ow)
                            col[col_idx_oh + ow] = shift;
                    }
                    for (int oh = oh_end; oh < hb; oh++) {
                        const ptrdiff_t col_idx_oh = col_idx_ic + oh * wb;
                        for (int ow = 0; ow < wb; ++ow)
                            col[col_idx_oh + ow] = shift;
                    }
                }
            }
        }
    } else {
        parallel_nd(jcp.kh, jcp.kw, jcp.ic, hb,
            [&](int kh, int kw, int ic, int oh) {
                const int hp = tp - kh * dh;
                const int ih = (oh + hs) * sh - hp;
                const ptrdiff_t col_idx_base
                        = (((kh * jcp.kw + kw) * jcp.ic + ic) * hb + oh) * wb;
                if (ih < 0 || ih >= jcp.ih)
                    for (int ow = 0; ow < wb; ow++)
                        col[col_idx_base + ow] = shift;
                else {
                    const int wp = lp - kw * dw;
                    const int ow_start = limit(0, wb, div_up(wp, sw) - ws);
                    const int ow_end
                            = limit(0, wb, div_up(jcp.iw + wp, sw) - ws);
                    for (int ow = 0; ow < ow_start; ow++)
                        col[col_idx_base + ow] = shift;
                    const int iw_base = ws * sw - wp;
                    const ptrdiff_t im_idx_base = ih * im_ih_stride + ic;
                    for (int ow = ow_start; ow < ow_end; ow++) {
                        const int iw = iw_base + ow * sw;
                        const ptrdiff_t im_idx
                                = im_idx_base + iw * im_iw_stride;
                        col[col_idx_base + ow] = im[im_idx] + shift;
                    }
                    for (int ow = ow_end; ow < wb; ow++)
                        col[col_idx_base + ow] = shift;
                }
            });
    }
}

template void im2col_u8<int8_t>(const jit_gemm_conv_conf_t &jcp,
        const int8_t *__restrict im, int8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb);
template void im2col_u8<uint8_t>(const jit_gemm_conv_conf_t &jcp,
        const uint8_t *__restrict im, uint8_t *__restrict imtr,
        uint8_t *__restrict col, int hs, int hb, int ws, int wb);

/* im[ih][iw][ic] <-- col2im_s32(col[oh][ow][kh][kw][ic]) */
void col2im_s32(const jit_gemm_conv_conf_t &jcp, const int32_t *__restrict col,
        int32_t *__restrict im)
{
    parallel(0, [&](const int ithr, const int nthr) {
        int h_nthr = nstl::min(jcp.ih, nthr);
        int w_nthr = nstl::min(jcp.iw, nthr / h_nthr);
        int h_ithr = 1, h_s = 0, h_e = 0, w_ithr = 1, w_s = 0, w_e = 0;
        if (ithr < h_nthr * w_nthr) {
            h_ithr = ithr / w_nthr;
            w_ithr = ithr % w_nthr;
            balance211(jcp.ih, h_nthr, h_ithr, h_s, h_e);
            balance211(jcp.iw, w_nthr, w_ithr, w_s, w_e);
        } else {
            h_ithr = w_ithr = -ithr;
            h_s = h_e = w_s = w_e = -1;
        }

        for (int ih = h_s; ih < h_e; ++ih) {
            for (int iw = w_s; iw < w_e; ++iw) {
                PRAGMA_OMP_SIMD()
                for (int ic = 0; ic < jcp.ic; ++ic) {
                    im[(ih * jcp.iw + iw) * jcp.ic + ic] = 0;
                }
            }
        }

        // TODO: reduce region: [0.. oh] --> [h_s * sh .. h_e * sh]
        for (int oh = 0; oh < jcp.oh; ++oh) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                for (int kh = 0; kh < jcp.kh; ++kh) {
                    const int ih = oh * jcp.stride_h
                        - jcp.t_pad + kh * (1 + jcp.dilate_h);
                    if (ih < h_s || ih >= h_e) continue;

                    for (int kw = 0; kw < jcp.kw; ++kw) {
                        const int iw = ow * jcp.stride_w
                            - jcp.l_pad + kw * (1 + jcp.dilate_w);
                        if (iw < w_s || iw >= w_e) continue;

                        const size_t col_idx = (((oh * jcp.ow + ow) * jcp.kh
                                + kh) * jcp.kw + kw) * jcp.ic;
                        const size_t im_idx
                            = (ih * jcp.iw + iw) * jcp.ic;
                        PRAGMA_OMP_SIMD()
                        for (int ic = 0; ic < jcp.ic; ++ic) {
                            im[im_idx + ic] += col[col_idx + ic];
                        }
                    }
                }
            }
        }
    });
}

void col2im_3d(const jit_gemm_conv_conf_t &jcp, const float *col, float *im,
        int od)
{
    parallel_nd(jcp.ic, [&](int ic) {
        const float *__restrict col_ = col + (size_t)ic * jcp.ks * jcp.os;
        float *__restrict im_ic = im + (size_t)ic * jcp.ih * jcp.iw * jcp.id;

        int id = od * jcp.stride_d - jcp.f_pad;
        for (int kd = 0; kd < jcp.kd; ++kd) {
            if (id < 0 || id >= jcp.id) {
                col_ += jcp.kh * jcp.kw * jcp.os;
                id += (1 + jcp.dilate_d);
                continue;
            }

            float *__restrict im_ = im_ic + id * jcp.ih * jcp.iw;

            for (int oh = 0; oh < jcp.oh; ++oh) {
            for (int kh = 0; kh < jcp.kh; ++kh) {
                const int ih = oh * jcp.stride_h - jcp.t_pad
                    + kh * (1 + jcp.dilate_h);
                if (ih < 0 || ih >= jcp.ih) continue;

                for (int ow = 0; ow < jcp.ow; ++ow) {
                for (int kw = 0; kw < jcp.kw; ++kw) {
                    const int iw = ow * jcp.stride_w - jcp.l_pad
                        + kw * (1 + jcp.dilate_w);
                    if (iw < 0 || iw >= jcp.iw) continue;

                    const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                    const size_t im_idx = ih*jcp.iw + iw;
                    im_[im_idx] += col_[col_idx];
                }}
            }}

            col_ += jcp.kh * jcp.kw * jcp.os;
            id += (1 + jcp.dilate_d);
        }
    });
}

void col2im(const jit_gemm_conv_conf_t &jcp, const float *col, float *im) {
    const size_t col_step = jcp.ks * jcp.os;
    const size_t im_step = jcp.ih * jcp.iw;
    const int iS = jcp.ih * jcp.iw;

    parallel_nd(jcp.ic, [&](int ic) {
        float *__restrict im_ = im + ic * im_step;
        const float *__restrict col_ = col + ic * col_step;
        PRAGMA_OMP_SIMD()
        for (int is = 0; is < iS; ++is) im_[is] = 0.;

        for (int kh = 0; kh < jcp.kh; ++kh) {
        for (int oh = 0; oh < jcp.oh; ++oh) {
            const int ih =
                    oh * jcp.stride_h - jcp.t_pad + kh * (1 + jcp.dilate_h);
            if (ih < 0 || ih >= jcp.ih) continue;

            for (int kw = 0; kw < jcp.kw; ++kw) {
            for (int ow = 0; ow < jcp.ow; ++ow) {
                const int iw =
                        ow * jcp.stride_w - jcp.l_pad + kw * (1 + jcp.dilate_w);
                if (iw < 0 || iw >= jcp.iw) continue;

                const size_t col_idx = ((kh*jcp.kw + kw)*jcp.oh+oh)*jcp.ow+ow;
                const size_t im_idx = ih*jcp.iw + iw;
                im_[im_idx] += col_[col_idx];
            }
            }
        }
        }
    });
}

status_t init_conf(jit_gemm_conv_conf_t &jcp,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, int max_threads) {
    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    const int ndims = src_d.ndims();
    const int is_1d = ndims == 3;
    const int is_3d = ndims == 5;

    jcp.prop_kind = cd.prop_kind;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];

    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.id = is_3d ? src_d.dims()[2] : 1;
    jcp.ih = is_1d ? 1 : src_d.dims()[ndims - 2];
    jcp.iw = src_d.dims()[ndims - 1];
    jcp.od = is_3d ? dst_d.dims()[2] : 1;
    jcp.oh = is_1d ? 1 : dst_d.dims()[ndims - 2];
    jcp.ow = dst_d.dims()[ndims - 1];

    jcp.kd = is_3d ? weights_d.dims()[with_groups + 2] : 1;
    jcp.kh = is_1d ? 1 : weights_d.dims()[with_groups + ndims - 2];
    jcp.kw = weights_d.dims()[with_groups + ndims - 1];

    jcp.f_pad = is_3d ? cd.padding[0][0] : 0;
    jcp.t_pad = is_1d ? 0 : cd.padding[0][ndims - 4];
    jcp.l_pad = cd.padding[0][ndims - 3];

    jcp.stride_d = is_3d ? cd.strides[0] : 1;
    jcp.stride_h = is_1d ? 1 : cd.strides[ndims - 4];
    jcp.stride_w = cd.strides[ndims - 3];

    jcp.dilate_d = is_3d ? cd.dilates[0] : 0;
    jcp.dilate_h = is_1d ? 0 : cd.dilates[ndims - 4];
    jcp.dilate_w = cd.dilates[ndims - 3];

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef
        || cd.diff_bias_desc.format_kind != format_kind::undef;

    jcp.is = jcp.ih * jcp.iw;
    jcp.os = jcp.oh * jcp.ow;
    jcp.ks = jcp.kh * jcp.kw * jcp.kd;

    jcp.signed_input = src_d.data_type() == data_type::s8;

    jcp.im2col_sz = !everyone_is(true,
            jcp.ow == jcp.iw, jcp.oh == jcp.ih, jcp.od == jcp.id,
            jcp.stride_w == 1, jcp.stride_h == 1, jcp.stride_d == 1,
            jcp.ks == 1, !jcp.signed_input)
        ? (ptrdiff_t)jcp.ic * jcp.ks * jcp.os : 0;

    jcp.outer_threading = false;

    bool is_int8_conv = utils::one_of(src_d.data_type(), s32, s8, u8)
        && weights_d.data_type() == s8;

    const int vlen = mayiuse(avx512_common)
        ? cpu_isa_traits<avx512_common>::vlen
        : mayiuse(avx)
            ? cpu_isa_traits<avx>::vlen
            : mayiuse(sse42) ? cpu_isa_traits<sse42>::vlen : 4;
    const int simd_w = vlen / (is_int8_conv ? 1 : 4);

    const bool is_bwd_d = jcp.prop_kind == backward_data;
    const bool is_bwd_w = jcp.prop_kind == backward_weights;
    const bool is_fwd = !is_bwd_d && !is_bwd_w;
    jcp.oh_block = is_fwd ? jcp.oh : jcp.ih;
    jcp.ow_block = is_fwd ? jcp.ow : jcp.iw;

    using namespace memory_tracking::names;
    bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;

    // TODO: maybe mitigate blocking restriction
    const int wei_size = jcp.oc * jcp.ic * jcp.kh * jcp.kw;
    const int L2 = get_cache_size(2, true)
          / (is_int8_conv ? sizeof(int8_t) : sizeof(float));
    bool is_blocking_applicable = true
            && is_fwd && jcp.im2col_sz
            && jcp.id == 1 && jcp.od == 1
            && jcp.dilate_h == 0 && jcp.dilate_w == 0
            && !is_depthwise
            && wei_size < L2/2;
    if (is_blocking_applicable) {
        // looking for oh and ow blocking
        int h_block{ jcp.oh_block }, w_block{ jcp.ow_block };
        const int ic = jcp.ic;
        const int oc = jcp.oc;
        const int iw = jcp.iw;
        const int ow = jcp.ow;
        const int oh = jcp.oh;
        const int os = oh * ow;

        // 1. cache requirement
        int row_size = ic * ow * jcp.ks + 2 * (ic * iw + oc * ow);
        if (is_int8_conv) {
            // Heuristic rule: gemm needed a lot of memory for internal usage
            row_size *= 5;
            // memory for accumulators
            row_size += oc * ow * sizeof(uint32_t);
            // memory for transposition
            row_size += ic * iw;
        }

        h_block = nstl::max(1, nstl::min(oh, div_up(L2, row_size)));
        if (h_block == 1) {
            int col_size = ic * jcp.ks + 2 * (ic + oc);
            if (is_int8_conv) {
                col_size *= 5;
                col_size += oc * sizeof(uint32_t);
                col_size += ic;
            }
            w_block = nstl::max(1, nstl::min(ow, div_up(L2, col_size)));
        }

        // 2. threading requirement
        if (h_block != oh)
            h_block = nstl::max(1, rnd_dn(h_block, 4));
        if (w_block != ow)
            w_block = nstl::max(1, rnd_dn(w_block, simd_w));

        float thr_eff = 0.f;
        float thr_eff_treshold = 0.9f;
        if (w_block == ow) {
            do {
                int nb_h = div_up(oh, h_block);
                size_t work = jcp.ngroups * jcp.mb * jcp.od * nb_h;
                float disb = (float)oh / rnd_up(oh, h_block);
                thr_eff = (float)work / rnd_up(work, max_threads);
                thr_eff = (thr_eff + disb) / 2.f;
                if (thr_eff >= thr_eff_treshold)
                    break;
                h_block = rnd_dn(h_block - 4, 4);
            } while (h_block > 0);
        }
        if (thr_eff < thr_eff_treshold) // we didn't find suitable h_block
        {
            h_block = 1;
            int nb_h = oh;
            do {
                int nb_w = div_up(ow, w_block);
                size_t work_amount = jcp.ngroups * jcp.mb * nb_h * nb_w;
                float disb = (float)ow / rnd_up(ow, w_block);
                thr_eff = (float)work_amount / rnd_up(work_amount, max_threads);
                thr_eff = (thr_eff + disb) / 2.f;
                if (thr_eff > thr_eff_treshold)
                    break;
                w_block = rnd_dn(w_block - simd_w, simd_w);
            } while (w_block > 0);
        }
        h_block = nstl::max(1, h_block);
        w_block = nstl::max(1, w_block);
        const size_t inner_work = div_up(os, simd_w) * div_up(oc, simd_w);
        const float inner_thr_eff
                = (float)inner_work / rnd_up(inner_work, max_threads);
        if (thr_eff >= inner_thr_eff / 2 && h_block > 0 && w_block > 0) {
            jcp.oh_block = h_block;
            jcp.ow_block = w_block;
            jcp.outer_threading = true;
        }
        // updating jcp.im2col_sz
        if (jcp.oh_block != 1)
            jcp.ow_block = ow;
        jcp.im2col_sz = (ptrdiff_t)ic * jcp.ks * jcp.oh_block * jcp.ow_block;
    }
    //  For threading selection in bwd_d we do:
    //  1. Rough estimation of efficiency for inner and outer threading.
    //  2. Gemm size estimation in assumption that it does not work
    //  so effectively for small sizes.
    //  64K - this is heuristic gemm size per thread threshold.
    const int gemm_thrld = 64 * 1024;

    if (is_int8_conv) {
        if (is_fwd) {
            if (!jcp.outer_threading) {
                bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;
                const size_t outer_work = jcp.ngroups * jcp.mb;
                const float outer_thr_eff
                    = (float)outer_work / rnd_up(outer_work, max_threads);
                const size_t inner_work
                    = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
                const float inner_thr_eff
                    = (float)inner_work / rnd_up(inner_work, max_threads);
                jcp.outer_threading = (is_depthwise
                 || (jcp.is / max_threads < 64 && jcp.mb != 1))
                 && (outer_thr_eff / inner_thr_eff >= 1.f
                     || (jcp.os * jcp.ic * jcp.oc) / max_threads < gemm_thrld);
            }
            jcp.nthr = jcp.outer_threading ? max_threads : 1;
            scratchpad.book(key_conv_gemm_col,
                sizeof(int8_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                sizeof(int32_t) * jcp.nthr * jcp.oh_block * jcp.ow_block * jcp.oc);
            scratchpad.book(key_conv_gemm_imtr,
                sizeof(int8_t) * jcp.nthr * jcp.is * jcp.ic);
        } else if (is_bwd_d) {
            bool is_depthwise = jcp.ic == 1 && jcp.oc == 1 && jcp.ngroups != 1;
            const size_t outer_work = jcp.ngroups * jcp.mb;
            const float outer_thr_eff
                    = (float)outer_work / rnd_up(outer_work, max_threads);
            const size_t inner_work
                    = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
            const float inner_thr_eff
                    = (float)inner_work / rnd_up(inner_work, max_threads);
            jcp.outer_threading = (is_depthwise
                 || (jcp.is / max_threads < 64 && jcp.mb != 1))
                 && (outer_thr_eff / inner_thr_eff >= 1.f
                     || (jcp.is * jcp.ic * jcp.oc) / max_threads < gemm_thrld);

            jcp.nthr = jcp.outer_threading ? max_threads : 1;
            scratchpad.book(key_conv_gemm_col,
                sizeof(int32_t) * jcp.nthr * jcp.im2col_sz);
            scratchpad.book(key_conv_int_dat_in_acc_dt,
                sizeof(int32_t) * jcp.nthr * jcp.is * jcp.ic);
        } else if (is_bwd_w) {
            assert(!"unimplemented prop_kind");
            return status::unimplemented;
        }
    } else {
        if (is_fwd) {
            if (!jcp.outer_threading) {
                const size_t outer_work_amount = jcp.ngroups * jcp.mb * jcp.od;
                const float outer_thr_eff = (float)outer_work_amount
                        / rnd_up(outer_work_amount, max_threads);
                const size_t inner_work_amount
                        = div_up(jcp.os, simd_w) * div_up(jcp.oc, simd_w);
                const float inner_thr_eff = (float)inner_work_amount
                        / rnd_up(inner_work_amount, max_threads);
                jcp.outer_threading = jcp.os / max_threads < 512
                    && IMPLICATION(jcp.od == 1, jcp.mb != 1 || jcp.ngroups > 2)
                    && (outer_thr_eff / inner_thr_eff >= 1.f
                      || (jcp.os * jcp.ic * jcp.oc) / max_threads < gemm_thrld);
            }
        } else if (is_bwd_d) {
            const size_t outer_work_amount = jcp.ngroups * jcp.mb;
            const float outer_thr_eff = (float)outer_work_amount
                / rnd_up(outer_work_amount, max_threads);
            const size_t inner_work
                = div_up(jcp.is, simd_w) * div_up(jcp.ic, simd_w);
            const float inner_thr_eff = (float)inner_work
                / rnd_up(inner_work, max_threads);
            jcp.outer_threading = (jcp.os / max_threads < 512 || jcp.ks < 64)
                && (jcp.mb != 1 || jcp.ngroups > 2)
                && (outer_thr_eff / inner_thr_eff >= 1.f
                    || (jcp.is * jcp.ic * jcp.oc) / max_threads < gemm_thrld);
        } else if (is_bwd_w)
            jcp.outer_threading = jcp.os / max_threads < 256
                && (jcp.mb != 1 || jcp.ngroups > 2);

        jcp.nthr = jcp.outer_threading ? max_threads : 1;
        scratchpad.book(key_conv_gemm_col,
                sizeof(float) * jcp.nthr * jcp.im2col_sz);

        if (is_bwd_w) {
            jcp.need_wei_reduction = mkldnn_thr_syncable()
                ? jcp.mb != 1 && jcp.nthr != 1 : false;
            scratchpad.book(key_conv_wei_reduction,
                    sizeof(float) * jcp.nthr * jcp.ngroups * weights_d.size());
        }
    }

    return status::success;
}

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb, int &ithr_g,
        int &nthr_g, int &ithr_mb, int &nthr_mb) {
    nthr_g = nstl::min(ngroups, nthr);
    nthr_mb = nstl::min(mb, nthr / nthr_g);
    if (ithr / nthr_mb >= ngroups) {
        ithr_g = ithr_mb = -1;
    } else {
        ithr_g = ithr / nthr_mb;
        ithr_mb = ithr % nthr_mb;
    }
}

void bwd_weights_reduction_par(int ithr, int nthr,
        const jit_gemm_conv_conf_t &jcp, const float *weights_reduce_ws,
        float *weights) {
    const size_t weights_g_size = jcp.ic * jcp.oc * jcp.ks;

    size_t weights_start{0}, weights_end{0};
    balance211(weights_g_size, nthr, ithr, weights_start, weights_end);

    for (int i = 0; i < nthr; ++i) {
        const float *ws_i = weights_reduce_ws + i * weights_g_size;
        for (size_t s = weights_start; s < weights_end; ++s)
            weights[s] = (i == 0 ? 0 : weights[s]) + ws_i[s];
    }
}

};

}
}
}
