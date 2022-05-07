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

#ifndef CPU_SIMPLE_REORDER_HPP
#define CPU_SIMPLE_REORDER_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "tag_traits.hpp"
#include "cpu_reorder_pd.hpp"
#include "cpu_primitive.hpp"

#include "simple_q10n.hpp"
#include "cpu_isa_traits.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::format_tag;
using namespace mkldnn::impl::data_type;

using bd = block_dim_t;
using ib = inner_blk_t;

using namespace mkldnn::impl::utils;
using math::saturate;

template<impl::data_type_t type>
using data_t = typename prec_traits<type>::type;

template<impl::data_type_t type_i, impl::data_type_t type_o>
using _qz_a1b0 = qz_a1b0<data_t<type_i>, data_t<type_o>>;

template<impl::data_type_t type_i, impl::data_type_t type_o>
using _qz = qz<data_t<type_i>, data_t<type_o>>;

namespace fmt_order {
    const bool keep = true;
    const bool reverse = false;
    const bool any = keep;
}

namespace spec {
struct direct_copy {};
struct direct_copy_except_dim_0 {};
struct reference {};
struct conv_s8s8 {};
}

#define SIMPLE_REORDER_TEMPL_DECL \
    impl::data_type_t type_i, impl::format_tag_t tag_i, \
    impl::data_type_t type_o, impl::format_tag_t tag_o, bool order_keep
#define SIMPLE_REORDER_TEMPL_CALL \
    type_i, tag_i, type_o, tag_o, order_keep

#define DECLARE_COMMON_PARAMS() \
        const memory_desc_wrapper &input_d = pd->src_md(); \
        const memory_desc_wrapper &output_d = pd->dst_md(); \
        const float alpha = pd->alpha(); MAYBE_UNUSED(alpha); \
        const float beta = pd->beta(); MAYBE_UNUSED(beta);

/* specific reorders: common template */
template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_impl {};

namespace {
inline bool simple_fmt_check(bool order_keep, impl::format_tag_t tag_i,
        impl::format_tag_t tag_o, const memory_desc_wrapper &input_d,
        const memory_desc_wrapper &output_d) {
    return input_d.matches_tag(order_keep ? tag_i : tag_o)
        && output_d.matches_tag(order_keep ? tag_o : tag_i);
}
inline bool simple_attr_check(const primitive_attr_t *attr, bool many_scales_support) {
    if (many_scales_support)
        return true;
    return IMPLICATION(attr, attr->output_scales_.mask_ == 0);
}
}

/* specific reorders: implementation */
template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
typename utils::enable_if<tag_i == any && (false
    || tag_o == hwio
    || tag_o == hwigo)
    , spec::conv_s8s8>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr)
    {
        const size_t D_mask = utils::array_product(input_d.dims(),
                                math::ilog2q(attr->output_scales_.mask_ + 1));
        const int oc = (input_d.dims()[tag_o == hwigo + 0]);
        const int g = (tag_o == hwigo) ? (input_d.dims()[0]) : 1;

        return output_d.matches_tag(tag_o)
            && (output_d.extra().flags & memory_extra_flags::compensation_conv_s8s8)
            && (input_d.data_type() == f32 || input_d.data_type() == s8)
            && output_d.data_type() == s8
            && (D_mask == 1 || D_mask == (size_t)g * oc);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups = tag_o == hwigo;

        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int IC = dims[w_groups + 1];
        const int H = dims[w_groups + 2];
        const int W = dims[w_groups + 3];

        const float *scales = pd->attr()->output_scales_.scales_;
        const size_t D_mask = utils::array_product(input_d.dims(),
                math::ilog2q(pd->attr()->output_scales_.mask_ + 1));

        assert(output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8);
        float adj_scale =
            (output_d.extra().flags & memory_extra_flags::scale_adjust)
            ? output_d.extra().scale_adjust : 1.f;

        size_t offset = G * pdims[w_groups + 0] * pdims[w_groups + 1] * H * W;
        int32_t *cp = reinterpret_cast<int32_t *>(output + offset);

        parallel_nd(G, OC, [&](int g, int oc) {
            cp[g * OC + oc] = 0;
            for (int ic = 0; ic < IC; ic++)
            for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++) {
                auto i = input[input_d.blk_off<!w_groups>(g, oc, ic, h, w)];
                auto &o = output[output_d.blk_off<!w_groups>(g, oc, ic, h, w)];
                const float s = scales[(D_mask == 1) ? 0 : g * OC + oc];

                o = qz_b0<data_t<type_i>, data_t<type_o>>()(
                    i, s * adj_scale);
                cp[g * OC + oc] -= (int32_t)o;
            }
            cp [g * OC + oc] *= 128;
        });
        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
          (tag_i == goiw && tag_o == gOIw4i16o4i)
       || (tag_i == oiw && tag_o == OIw4i16o4i)
       || (tag_i == goihw && tag_o == gOIhw4i16o4i)
       || (tag_i == oihw && tag_o == OIhw4i16o4i)
       || (tag_i == goihw && tag_o == gOIhw2i8o4i)
       || (tag_i == goihw && tag_o == gOIhw4o4i)
    , spec::conv_s8s8>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr)
    {
        const size_t D_mask = utils::array_product(input_d.dims(),
                                math::ilog2q(attr->output_scales_.mask_ + 1));
        const bool w_groups = !utils::one_of(tag_o, OIw4i16o4i, OIhw4i16o4i);
        const int oc = (input_d.dims()[w_groups ? 1 : 0]);
        const int g = w_groups ? input_d.dims()[0] : 1;

        return input_d.matches_tag(tag_i)
            && output_d.matches_tag(tag_o)
            && (output_d.extra().flags & memory_extra_flags::compensation_conv_s8s8)
            && (input_d.data_type() == f32 || input_d.data_type() == s8)
            && output_d.data_type() == s8
            && (D_mask == 1 || D_mask == (size_t)g * oc);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        static constexpr bool w_groups =
            !utils::one_of(tag_o, OIw4i16o4i, OIhw4i16o4i);
        constexpr int is_1d =
            utils::one_of(tag_o, gOIw4i16o4i, OIw4i16o4i);
        constexpr int blksize = tag_traits<tag_o>::inner_blks == ib::_4b4c
            ? 4
            : tag_traits<tag_o>::inner_blks == ib::_2c8b4c
            ? 8
            : 16;

        const auto &_g_oihw_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.padded_dims()
            : input_d.padded_dims();

        const int G = w_groups ? dims[0] : 1;
        const int OC = dims[w_groups + 0];
        const int NB_OC = pdims[w_groups + 0] / blksize;
        const int IC = dims[w_groups + 1];
        const int NB_IC = pdims[w_groups + 1] / blksize;
        const int H = is_1d ? 1 : dims[w_groups + 2];
        const int W = dims[w_groups + 3 - is_1d];

        const float *scales = pd->attr()->output_scales_.scales_;
        const size_t D_mask = utils::array_product(input_d.dims(),
                            math::ilog2q(pd->attr()->output_scales_.mask_ + 1));

        assert(output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8);
        float adj_scale =
            (output_d.extra().flags & memory_extra_flags::scale_adjust)
            ? output_d.extra().scale_adjust : 1.f;

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
            int32_t *c, const float *s, const int oc_block, const int ic_block) {
#           define index AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>

            for (int ic = 0; ic < ic_block; ++ic) {
            for (int oc = 0; oc < oc_block; ++oc) {
                const auto _g_oihw_off =
                    oc * _g_oihw_d.blocking_desc().strides[w_groups + 0]
                  + ic * _g_oihw_d.blocking_desc().strides[w_groups + 1];
                out[index(oc, ic)]
                    = qz_b0<data_t<type_i>, data_t<type_o>>()(
                            inp[_g_oihw_off], s[oc] * adj_scale);
                c[oc] -= (128 * (int32_t)(out[index(oc, ic)]));
            }
            }
#           undef index
        };

        constexpr int i_mult = blksize;
        constexpr int o_mult = 1;

        size_t offset = G * pdims[w_groups+0] * pdims[w_groups+1] * H * W;
        int32_t *cp = reinterpret_cast<int32_t *>(output + offset);
        parallel_nd(G * NB_OC * blksize, [&](int i) {
            cp[i] = 0;
        });

#       define wei_blk_off(md, g, o, i, h, w) \
        (is_1d ? (md).blk_off<!w_groups>(g, o, i, w) \
         : (md).blk_off<!w_groups>(g, o, i, h, w))

        parallel_nd(G, NB_OC, [&](int g, int O) {
            for (int I = 0; I < NB_IC; I++)
                for (int h = 0; h < H; h++)
                for (int w = 0; w < W; w++) {
                    auto i = &input[wei_blk_off(
                            input_d, g, i_mult * O, i_mult * I, h, w)];
                    auto o = &output[wei_blk_off(
                            output_d, g, o_mult * O, o_mult * I, h, w)];
                    const int oc_block = nstl::min(blksize, OC - O * blksize);
                    const int ic_block = nstl::min(blksize, IC - I * blksize);

                    int _offset = (g * NB_OC + O) * blksize;
                    ker(i, o, (order_keep) ? &cp[_offset] : nullptr,
                            &scales[(D_mask == 1) ? 0 : _offset],
                                        oc_block, ic_block);
                }
        });

#       undef wei_blk_off

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<false
    ||(tag_i == goiw && tag_o == Goiw16g)
    ||(tag_i == goihw && tag_o == Goihw16g)
    , spec::conv_s8s8>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        const size_t D_mask = utils::array_product(input_d.dims(),
                            math::ilog2q(attr->output_scales_.mask_ + 1));
        const int oc = input_d.dims()[1];
        const int g = input_d.dims()[0];

        return true
            && order_keep
            && input_d.matches_tag(tag_i)
            && output_d.matches_tag(tag_o)
            && (output_d.extra().flags & memory_extra_flags::compensation_conv_s8s8)
            && (input_d.data_type() == f32 || input_d.data_type() == s8)
            && output_d.data_type() == s8
            && (D_mask == 1 || D_mask == (size_t)g * oc);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
            const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr bool is_1d = tag_i == goiw;
        constexpr int blksize = 16;

        const auto &dims = input_d.dims();
        const auto &pdims = output_d.padded_dims();
        const int G = dims[0];
        const int Gp = pdims[0];
        const int OC = dims[1];
        const int IC = dims[2];
        const int H = is_1d ? 1 : dims[3];
        const int W = dims[4 - is_1d];

        const size_t D_mask = utils::array_product(input_d.dims(),
                            math::ilog2q(pd->attr()->output_scales_.mask_ + 1));
        const float *scales = pd->attr()->output_scales_.scales_;

        assert(output_d.extra().flags
                & memory_extra_flags::compensation_conv_s8s8);
        float adj_scale =
            (output_d.extra().flags & memory_extra_flags::scale_adjust)
            ? output_d.extra().scale_adjust : 1.f;

        auto ker = [&](const data_t<type_i> *inp, data_t<type_o> *out,
                int32_t *cp, const float *s, const int g_block) {
            PRAGMA_OMP_SIMD()
            for (int g = 0; g < g_block; g++) {
                const auto i_off = g * input_d.blocking_desc().strides[0];
                out[g] = qz_b0<data_t<type_i>, data_t<type_o>>()(
                        inp[i_off], s[g * OC] * adj_scale);
                cp[g * OC] -= 128 * (int32_t)(out[g]);
            }
        };

        size_t cp_offset = output_d.size() - output_d.additional_buffer_size();
        int32_t *cp = reinterpret_cast<int32_t *>(output + cp_offset);
        parallel_nd((Gp/blksize) * OC, [&](int ib) {
            PRAGMA_OMP_SIMD()
            for (int i = 0; i < blksize; i++)
                cp[ib * blksize + i] = 0;
        });

#       define wei_blk_off(md, g, o, i, h, w) \
        (is_1d ? (md).blk_off(g, o, i, w) : (md).blk_off(g, o, i, h, w))

        parallel_nd(Gp/blksize, OC, [&](int gb, int O) {
                for (int I = 0; I < IC; I++) {
                    for (int h = 0; h < H; h++)
                    for (int w = 0; w < W; w++)
                    {
                        const int g_block = nstl::min(G - gb * blksize, blksize);
                        const auto inp = &input[wei_blk_off(
                                input_d, gb * blksize, O, I, h, w)];
                        const auto out = &output[wei_blk_off(
                                output_d, gb, O, I, h, w)];
                        int offset = gb * blksize + O;
                        ker(inp, out, &cp[offset],
                            &scales[(D_mask == 1) ? 0 : offset], g_block);
                   }
               }
        });

#       undef wei_blk_off

        return success;
    }
};

/* reorders with tail support */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
typename utils::enable_if<false
    || (tag_i == nCdhw8c && tag_o == nCdhw16c)
    || (tag_i == nChw8c && tag_o == nChw16c)
    || (tag_i == nCw8c && tag_o == nCw16c)
    >::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr)
    {
        return simple_fmt_check(order_keep, tag_i, tag_o, input_d, output_d)
            && simple_attr_check(attr, false);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        constexpr int is_1d = tag_i == nCw8c;
        constexpr int is_3d = tag_i == nCdhw8c;
        constexpr int blksize_16 = 16;
        constexpr int blksize_8 = 8;
        constexpr int ic_mult = order_keep ? 2 : 1;
        constexpr int oc_mult = order_keep ? 1 : 2;

        const auto &dims = input_d.dims();
        const auto &pdims = order_keep ? output_d.padded_dims()
                                       : input_d.padded_dims();

        const int C = dims[1];
        const int D = is_3d ? dims[2] : 1;
        const int H = is_1d ? 1 : dims[2 + is_3d];
        const int W = dims[3 + is_3d - is_1d];

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
            const int block_16) {
            const int nb = (block_16 - 1) / blksize_8 + 1;
            if (alpha == 1.0 && beta == 0.0) {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off = order_keep ? b : b * blksize_8;
                    const ptrdiff_t o_off = order_keep ? b * blksize_8 : b;
                    const int block_8 = nstl::min(blksize_8,
                                                  block_16 - b * blksize_8);
                    for (int c = 0; c < block_8; ++c) {
                        o[o_off + c] = _qz_a1b0<type_i, type_o>()(
                                i[i_off + c]);
                    }
                }
            } else {
                for (int b = 0; b < nb; ++b) {
                    const ptrdiff_t i_off = order_keep ? b : b * blksize_8;
                    const ptrdiff_t o_off = order_keep ? b * blksize_8 : b;
                    const int block_8 = nstl::min(blksize_8,
                                                  block_16 - b * blksize_8);
                    for (int c = 0; c < block_8; ++c) {
                        o[o_off + c] = _qz<type_i, type_o>()(i[i_off + c],
                                o[o_off + c], alpha, beta);
                    }
                }
            }
        };

#       define data_blk_off(md, n, c, d, h, w) \
        ( is_1d ? (md).blk_off(n, c, w) \
          : is_3d ? (md).blk_off(n, c, d, h, w) : (md).blk_off(n, c, h, w))

        parallel_nd(dims[0], pdims[1] / blksize_16, D, H, W,
            [&](int n, int nb_c, int d, int h, int w) {
            auto i = &input[data_blk_off(input_d, n, ic_mult * nb_c, d, h, w)];
            auto o = &output[data_blk_off(output_d, n, oc_mult * nb_c, d, h, w)];
            const int block_16 = nstl::min(blksize_16, C - nb_c * blksize_16);
            ker(i, o, block_16);
        });

#       undef data_blk_off

        return success;
    }
};

#define PLAIN_TO_BLOCKED_IS_APPLICABLE() \
    static bool is_applicable(const memory_desc_wrapper &input_d, \
        const memory_desc_wrapper &output_d, const primitive_attr_t *attr) { \
        return simple_attr_check(attr, false) && (order_keep \
                ? output_d.matches_tag(tag_o) && input_d.is_plain() \
                : input_d.matches_tag(tag_o) && output_d.is_plain()); \
    }

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
typename utils::enable_if<tag_i == any
    && (tag_traits<tag_o>::block_dims == bd::_A
            || tag_traits<tag_o>::block_dims == bd::_B)
    && tag_traits<tag_o>::ndims >= 3
    && tag_traits<tag_o>::ndims <= 6
    >::type>
{
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &block_d = order_keep ? output_d : input_d;
        const auto &dims = input_d.dims();
        const auto &pdims = block_d.padded_dims();

        constexpr int ndims = tag_traits<tag_o>::ndims;
        constexpr int blk_idx = tag_traits<tag_o>::block_dims == bd::_A ? 0 : 1;

        const dim_t H0 = dims[0];
        const dim_t H1 = dims[1];
        const dim_t M0 = ndims >= 6 ? dims[ndims - 4] : 1;
        const dim_t M1 = ndims >= 5 ? dims[ndims - 3] : 1;
        const dim_t M2 = ndims >= 4 ? dims[ndims - 2] : 1;
        const dim_t L = dims[ndims - 1];
        const dim_t l_blk_stride = block_d.blocking_desc().strides[ndims - 1];

        constexpr int blksize = false ? 0
            : utils::one_of(tag_traits<tag_o>::inner_blks, ib::_4a, ib::_4b) ? 4
            : utils::one_of(tag_traits<tag_o>::inner_blks, ib::_8a, ib::_8b) ? 8
            : 16;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o, int block) {
            if (alpha == 1.0 && beta == 0.0) {
                for (int l = 0; l < L; ++l)
                for (int blk = 0; blk < block; ++blk) {
                    const dim_t flat_off = 0
                        + blk * flat_d.blocking_desc().strides[blk_idx]
                        + l * flat_d.blocking_desc().strides[ndims - 1];
                    if (order_keep) {
                        o[l * l_blk_stride + blk] = _qz_a1b0<type_i, type_o>()(
                                i[flat_off]);
                    } else {
                        o[flat_off] = _qz_a1b0<type_i, type_o>()(
                                i[l * l_blk_stride + blk]);
                    }
                }
            } else {
                for (int l = 0; l < L; ++l)
                for (int blk = 0; blk < block; ++blk) {
                    const dim_t flat_off = 0
                        + blk * flat_d.blocking_desc().strides[blk_idx]
                        + l * flat_d.blocking_desc().strides[ndims - 1];
                    if (order_keep) {
                        o[l * l_blk_stride + blk] = _qz<type_i, type_o>()(
                                i[flat_off], o[l * blksize + blk],
                                alpha, beta);
                    } else {
                        o[flat_off] = _qz<type_i, type_o>()(
                                i[l * l_blk_stride + blk], o[flat_off],
                                alpha, beta);
                    }
                }
            }
        };

#       define off(md, h0, h1, m0, m1, m2) \
        (ndims >= 6 ? (md).blk_off(h0, h1, m0, m1, m2) \
         : ndims >= 5 ? (md).blk_off(h0, h1, m1, m2) \
         : ndims >= 4 ? (md).blk_off(h0, h1, m2) \
         : /* ndims >= 3 ? */ (md).blk_off(h0, h1))

        constexpr int i_mult = order_keep ? blksize : 1;
        constexpr int o_mult = order_keep ? 1 : blksize;

        if (blk_idx == 0) {
            const dim_t BH0 = pdims[0] / blksize;
            parallel_nd(BH0, H1, M0, M1, M2,
                [&](dim_t bh0, dim_t h1, dim_t m0, dim_t m1, dim_t m2) {
                auto i = &input[off(input_d, bh0 * i_mult, h1, m0, m1, m2)];
                auto o = &output[off(output_d, bh0 * o_mult, h1, m0, m1, m2)];
                const int block = nstl::min<int>(blksize, H0 - bh0 * blksize);
                ker(i, o, block);
            });
        } else if (blk_idx == 1) {
            const dim_t BH1 = pdims[1] / blksize;
            parallel_nd(H0, BH1, M0, M1, M2,
                [&](dim_t h0, dim_t bh1, dim_t m0, dim_t m1, dim_t m2) {
                auto i = &input[off(input_d, h0, bh1 * i_mult, m0, m1, m2)];
                auto o = &output[off(output_d, h0, bh1 * o_mult, m0, m1, m2)];
                const int block = nstl::min<int>(blksize, H1 - bh1 * blksize);
                ker(i, o, block);
            });
        } else {
            assert(!"unimplemented");
        }

#       undef off

        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
typename utils::enable_if<tag_i == any
    && (tag_traits<tag_o>::block_dims == bd::_AB
            || tag_traits<tag_o>::block_dims == bd::_BC)
    && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_AB,
            tag_traits<tag_o>::ndims >= 3 && tag_traits<tag_o>::ndims <= 5)
    && IMPLICATION(tag_traits<tag_o>::block_dims == bd::_BC,
            tag_traits<tag_o>::ndims >= 4 && tag_traits<tag_o>::ndims <= 6)
    >::type>
{
    PLAIN_TO_BLOCKED_IS_APPLICABLE();

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const auto &flat_d = order_keep ? input_d : output_d;
        const auto &dims = input_d.dims();
        const auto &pdims = order_keep
            ? output_d.padded_dims()
            : input_d.padded_dims();

        constexpr int ndims = tag_traits<tag_o>::ndims;

        static constexpr bool with_g = tag_traits<tag_o>::block_dims == bd::_BC;
        const dim_t G = with_g ? dims[0] : 1;

        const dim_t H0 = dims[0 + with_g];
        const dim_t H1 = dims[1 + with_g];

        const dim_t M0 = ndims >= 5 + with_g ? dims[ndims - 3] : 1;
        const dim_t M1 = ndims >= 4 + with_g ? dims[ndims - 2] : 1;
        const dim_t M2 = ndims >= 3 + with_g ? dims[ndims - 1] : 1;

        constexpr int blksize_0 = false ? 0
            : utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_4b4a, ib::_4b4c, ib::_4c4b)
            ? 4
            : utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_8a8b, ib::_8b8a, ib::_8b8c, ib::_8c8b, ib::_2c8b4c)
            ? 8
            : utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_16a16b, ib::_16a4b, ib::_16b16a, ib::_16b4c,
                    ib::_16b16c, ib::_16c16b, ib::_8a16b2a, ib::_4b16a4b,
                    ib::_8b16a2b, ib::_8b16c2b, ib::_4c16b4c, ib::_8c16b2c)
            ? 16 : INT_MIN;

        constexpr int blksize_1 = utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_8a8b, ib::_8b8a, ib::_8b8c, ib::_8c8b, ib::_2c8b4c)
            ? 8
            : utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_16a16b, ib::_16b16a, ib::_16b16c, ib::_16c16b,
                    ib::_8a16b2a, ib::_4b16a4b, ib::_8b16a2b, ib::_8b16c2b,
                    ib::_4c16b4c, ib::_8c16b2c)
            ? 16
            : utils::one_of(tag_traits<tag_o>::inner_blks,
                    ib::_4b4a, ib::_4b4c, ib::_4c4b,
                    ib::_16a4b, ib::_16b4c)
            ? 4
            : INT_MIN;

        const dim_t NB_H0 = pdims[0 + with_g] / blksize_0;
        const dim_t NB_H1 = pdims[1 + with_g] / blksize_1;

        auto ker = [&](const data_t<type_i> *i, data_t<type_o> *o,
            const int block_h0, const int block_h1) {
#           define blk_off AB_or_BC_blk_off<tag_traits<tag_o>::inner_blks>

            if (alpha == 1.0 && beta == 0.0) {
                for (int h0 = 0; h0 < block_h0; ++h0)
                for (int h1 = 0; h1 < block_h1; ++h1) {
                    const dim_t flat_off = 0
                        + h0 * flat_d.blocking_desc().strides[with_g + 0]
                        + h1 * flat_d.blocking_desc().strides[with_g + 1];
                    if (order_keep) {
                        o[blk_off(h0, h1)] = _qz_a1b0<type_i, type_o>()(
                                i[flat_off]);
                    } else {
                        o[flat_off] = _qz_a1b0<type_i, type_o>()(
                                i[blk_off(h0, h1)]);
                    }
                }
            } else {
                for (int h0 = 0; h0 < block_h0; ++h0)
                for (int h1 = 0; h1 < block_h1; ++h1) {
                    const dim_t flat_off = 0
                        + h0 * flat_d.blocking_desc().strides[with_g + 0]
                        + h1 * flat_d.blocking_desc().strides[with_g + 1];
                    if (order_keep) {
                        o[blk_off(h0, h1)] = _qz<type_i, type_o>()(i[flat_off],
                                o[blk_off(h0, h1)], alpha, beta);
                    } else {
                        o[flat_off] = _qz<type_i, type_o>()(i[blk_off(h0, h1)],
                                o[flat_off], alpha, beta);
                    }
                }
            }

#           undef blk_off
        };

        constexpr int i_mult_0 = order_keep ? blksize_0 : 1;
        constexpr int o_mult_0 = order_keep ? 1 : blksize_0;

        constexpr int i_mult_1 = order_keep ? blksize_1 : 1;
        constexpr int o_mult_1 = order_keep ? 1 : blksize_1;

#       define off(md, g, h0, h1, m0, m1, m2) \
        (ndims >= 5 + with_g ? (md).blk_off<!with_g>(g, h0, h1, m0, m1, m2) \
         : ndims >= 4 + with_g ? (md).blk_off<!with_g>(g, h0, h1, m1, m2) \
         : /* ndims >= 3 + with_g ? */ (md).blk_off<!with_g>(g, h0, h1, m2))

        parallel_nd(G, NB_H0, NB_H1, M0, M1, M2,
            [&](dim_t g, dim_t nb_h0, dim_t nb_h1, dim_t m0, dim_t m1, dim_t m2) {
            auto i = &input[off(input_d,
                    g, i_mult_0 * nb_h0, i_mult_1 * nb_h1, m0, m1, m2)];
            auto o = &output[off(output_d,
                    g, o_mult_0 * nb_h0, o_mult_1 * nb_h1, m0, m1, m2)];
            const int block_h0 = nstl::min<int>(blksize_0, H0 - nb_h0 * blksize_0);
            const int block_h1 = nstl::min<int>(blksize_1, H1 - nb_h1 * blksize_1);
            ker(i, o, block_h0, block_h1);
        });

#       undef off

        return success;
    }
};

/* generic and direct-copy reorders */

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        tag_i == any && tag_o == any && order_keep == fmt_order::any,
    spec::direct_copy>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 0)
            && input_d.is_dense() && output_d.is_dense()
            && simple_attr_check(attr, false);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        assert(input_d.is_dense());

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const size_t nelems = input_d.nelems();

        constexpr int block_size = 16;
        const auto num_blocks = nelems / block_size;
        const auto rem_elems = nelems % block_size;

        parallel(0, [&](const int ithr, const int nthr) {
            size_t start{0}, end{0};
            balance211(num_blocks, nthr, ithr, start, end);
            start = start * block_size;
            end = end * block_size;

            if (alpha == 1.0 && beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1b0<data_t<type_i>, data_t<type_o>>()
                                (input[e]);
                }
            } else if (alpha == 1.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_a1<data_t<type_i>, data_t<type_o>>()
                                (input[e], output[e], beta);
                }
            } else if (beta == 0.0) {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz_b0<data_t<type_i>, data_t<type_o>>()
                                (input[e], alpha);
                }
            } else {
                PRAGMA_OMP_SIMD()
                for (size_t e = start; e < end; ++e) {
                    output[e] = qz<data_t<type_i>, data_t<type_o>>()
                                (input[e], output[e], alpha, beta);
                }
            }

            if (rem_elems != 0 && ithr == nthr - 1){
                if (alpha == 1.0 && beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1b0<data_t<type_i>,
                            data_t<type_o>>()(input[e]);
                    }
                } else if (alpha == 1.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_a1<data_t<type_i>,
                            data_t<type_o>>()(input[e], output[e], beta);
                    }
                } else if (beta == 0.0) {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz_b0<data_t<type_i>,
                            data_t<type_o>>()(input[e], alpha);
                    }
                } else {
                    PRAGMA_OMP_SIMD()
                    for (size_t e = nelems - rem_elems; e < nelems; ++e) {
                        output[e] = qz<data_t<type_i>, data_t<type_o>>()
                                    (input[e], output[e], alpha, beta);
                   }
               }
            }
        });
        return success;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        tag_i == any && tag_o == any && order_keep == fmt_order::any,
    spec::direct_copy_except_dim_0>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        auto is_dense_no_0 = [](const memory_desc_wrapper &data_d) {
            return nelems_no_dim_0(data_d) == _size_no_dim_0(data_d);
        };
        /* FIXME: is the formula correct? */
        return input_d.similar_to(output_d, true, false, 1)
            && is_dense_no_0(input_d) && is_dense_no_0(output_d)
            && simple_attr_check(attr, false);
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        input += input_d.blk_off(0);
        output += output_d.blk_off(0);

        const int N = input_d.dims()[0];
        const dim_t is = input_d.blocking_desc().strides[0];
        const dim_t os = output_d.blocking_desc().strides[0];
        const dim_t nelems_no_d0 = nelems_no_dim_0(input_d);
        const dim_t work_amount = N * nelems_no_d0;

        if (alpha == 1.0 && beta == 0.0) {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n{0}, dim1_s{0};
                dim_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e = dim1_s + work_rem > nelems_no_d0
                        ? nelems_no_d0 : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e) {
                        output[os * n + e] = _qz_a1b0<type_i, type_o>()(
                                input[is * n + e]);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        } else {
            parallel(0, [&](const int ithr, const int nthr) {
                dim_t n{0}, dim1_s{0};
                dim_t start{0}, end{0};
                balance211(work_amount, nthr, ithr, start, end);
                nd_iterator_init(start, n, N, dim1_s, nelems_no_d0);
                while(start < end) {
                    dim_t work_rem = end - start;
                    dim_t dim1_e =
                        dim1_s + work_rem > nelems_no_d0 ? nelems_no_d0
                        : dim1_s + work_rem;
                    PRAGMA_OMP_SIMD()
                    for (dim_t e = dim1_s; e < dim1_e; ++e){
                        output[os * n + e] = _qz<type_i, type_o>()(
                                input[is * n + e], output[os * n + e], alpha,
                                beta);
                    }
                    nd_iterator_jump(start, end, n, N, dim1_s, nelems_no_d0);
                }
            });
        }

        return success;
    }

private:
    static dim_t nelems_no_dim_0(const memory_desc_wrapper &data_d) {
        const int ndims = data_d.ndims();
        if (ndims <= 1) return 1;
        return utils::array_product(data_d.dims() + 1, data_d.ndims() - 1);
    }

    static dim_t _size_no_dim_0(const memory_desc_wrapper &data_d) {
        dims_t blocks;
        data_d.compute_blocks(blocks);

        const auto &blk = data_d.blocking_desc();

        dim_t blk_size = 1;
        for (int iblk = 0; iblk < blk.inner_nblks; ++iblk)
            blk_size *= blk.inner_blks[iblk];

        dim_t max_size = blk_size;
        for (int d = 1; d < data_d.ndims(); ++d) {
            max_size = nstl::max(max_size,
                    data_d.padded_dims()[d] / blocks[d] * blk.strides[d]);
        }

        return max_size;
    }
};

template <SIMPLE_REORDER_TEMPL_DECL>
struct simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL,
    typename utils::enable_if<
        tag_i == any && tag_o == any && order_keep == fmt_order::any,
    spec::reference>::type>
{
    static bool is_applicable(const memory_desc_wrapper &input_d,
            const memory_desc_wrapper &output_d, const primitive_attr_t *attr) {
        /* supported smask: 0x0...011..10...0,
         * i.e. 1 should be contiguous */
        int smask = attr ? attr->output_scales_.mask_ : 0;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1);
        for (; smask > 0 && smask & 0x1; smask >>= 1);
        return true
            && input_d.is_blocking_desc()
            && output_d.is_blocking_desc()
            && !output_d.is_additional_buffer()
            && !input_d.is_additional_buffer()
            && smask == 0;
    }

    static status_t execute(const cpu_reorder_pd_t *pd,
        const data_t<type_i> *input, data_t<type_o> *output) {
        DECLARE_COMMON_PARAMS();

        const size_t nelems = input_d.nelems();

        int ndims_start = 0, ndims_mask = 0;
        int smask = pd->attr()->output_scales_.mask_;
        for (; smask > 0 && !(smask & 0x1); smask >>= 1) ++ndims_start;
        for (; smask > 0 && smask & 0x1; smask >>= 1) ++ndims_mask;
        assert(smask == 0);

        const ptrdiff_t D_start
            = utils::array_product(input_d.dims(), ndims_start);
        const ptrdiff_t D_mask
            = utils::array_product(input_d.dims() + ndims_start, ndims_mask);
        const ptrdiff_t D_rest = nelems / D_start / D_mask;

        const float *scales = pd->attr()->output_scales_.scales_;

        parallel_nd(D_start, D_mask, D_rest,
            [&](ptrdiff_t ds, ptrdiff_t dm, ptrdiff_t dr) {
            const float scale = scales[dm];

            const size_t e = (ds * D_mask + dm) * D_rest + dr;
            const auto &i = input[input_d.off_l(e)];
            auto &o = output[output_d.off_l(e)];

            o = _qz<type_i, type_o>()(i, o, scale, beta);
        });

        return success;
    }
};


/* high level class declaration */

template <SIMPLE_REORDER_TEMPL_DECL, typename spec = void>
struct simple_reorder_t: public cpu_primitive_t {
    struct pd_t: public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("simple:any", simple_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
            bool args_ok = true
                && src_md->data_type == type_i
                && dst_md->data_type == type_o
                && simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::
                is_applicable(src_md, dst_md, attr);
            if (!args_ok)
                return status::invalid_arguments;

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init() != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

    simple_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const data_t<type_i> *, MKLDNN_ARG_FROM);
        auto output = CTX_OUT_MEM(data_t<type_o> *, MKLDNN_ARG_TO);
        simple_reorder_impl<SIMPLE_REORDER_TEMPL_CALL, spec>::execute(
                pd(), input, output);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

#undef SIMPLE_REORDER_TEMPL_DECL
#undef SIMPLE_REORDER_TEMPL_CALL

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
