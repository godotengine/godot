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

#ifndef CPU_WINO_REORDER_HPP
#define CPU_WINO_REORDER_HPP

#include "mkldnn_thread.hpp"

#include "simple_q10n.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t type_i, data_type_t type_o>
struct wino_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("wino_reorder", wino_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true
                && id.data_type() == type_i
                && od.data_type() == type_o
                && id.matches_tag(utils::pick(id.ndims() - 4,
                            format_tag::oihw, format_tag::goihw))
                && od.format_kind() == format_kind::wino
                && utils::one_of(od.wino_desc().wino_format,
                        mkldnn_wino_wei_aaOIoi, mkldnn_wino_wei_aaOio,
                        mkldnn_wino_wei_aaOBiOo, mkldnn_wino_wei_OBaaIBOIio);
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return status::out_of_memory;
            if (_pd->init() != status::success) {
                delete _pd;
                return status::unimplemented;
            }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            auto &o = memory_desc_wrapper(dst_md()).wino_desc();
            size_t transform_space_size = (size_t)o.r * o.alpha * o.oc_block;
            size_t plain_size = (size_t)o.alpha * o.alpha * o.oc * o.ic;

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_reorder_wino_transform_space,
                    sizeof(in_data_t) * transform_space_size);
            scratchpad.book(key_reorder_wino_plain,
                    sizeof(out_data_t) * plain_size);
        }
    };

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;
    const int unsign_val_in_wino_domain_ = 5;

    wino_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {
        const memory_desc_wrapper src_d(pd()->src_md());
        const memory_desc_wrapper dst_d(pd()->dst_md());

        r_ = dst_d.wino_desc().r;
        w_alpha_ = dst_d.wino_desc().alpha;
        wino_format_ = dst_d.wino_desc().wino_format;

        const auto &in_dims = src_d.dims();
        int groups;
        int groups_offset;
        if (src_d.ndims() == 5) {
            groups = in_dims[0];
            groups_offset = 1;
        } else {
            groups = 1;
            groups_offset = 0;
        }
        assert(groups == 1); // groups are not supported now
        MAYBE_UNUSED(groups);

        or_oc_ = in_dims[0 + groups_offset];
        or_ic_ = in_dims[1 + groups_offset];
        kh_ = in_dims[2 + groups_offset];
        kw_ = in_dims[3 + groups_offset];

        oc_ = dst_d.wino_desc().oc;
        ic_ = dst_d.wino_desc().ic;
        oc_block_ = dst_d.wino_desc().oc_block;
        ic_block_ = dst_d.wino_desc().ic_block;
        assert(oc_ % oc_block_ == 0 && ic_ % ic_block_ == 0);
        nb_oc_ = oc_ / oc_block_;
        nb_ic_ = ic_ / ic_block_;
        ic2_block_ = 1;
        if (wino_format_ == mkldnn_wino_wei_OBaaIBOIio)
            ic2_block_ = dst_d.wino_desc().ic2_block;
        oc2_block_ = dst_d.wino_desc().oc2_block;
        assert(nb_ic_ % ic2_block_ == 0 && nb_oc_ % oc2_block_ == 0);

        adj_scale_ = dst_d.wino_desc().adj_scale;

        size_wino_wei_ = w_alpha_ * w_alpha_ * oc_ * ic_;
        size_wspace_ = r_ * w_alpha_ * oc_block_;
    }

    void transform(out_data_t *__restrict tmp_wei,
            const in_data_t *__restrict input,
            in_data_t *__restrict wspace) const {
        const memory_desc_wrapper src_d(pd()->src_md());

        const int smask = pd()->attr()->output_scales_.mask_;
        const int ndims_mask = math::ilog2q(smask + 1);
        const size_t D_mask = utils::array_product(src_d.dims(), ndims_mask);
        const float *__restrict scales = pd()->attr()->output_scales_.scales_;
        assert(D_mask == 1 || D_mask == (size_t)oc_);

        /* transform weights to winograd domain */
        const float G_2x2_3x3[4][3] = { { 1.0, 0.0, 0.0 }, { 0.5, 0.5, 0.5 },
            { 0.5, -0.5, 0.5 }, { 0.0, 0.0, 1.0 } };

        const float G_4x4_3x3[6][3] = { { 1.13777777777778f, 0.f, 0.f },
            { -0.688403361344538f, -0.430252100840336f, -0.26890756302521f },
            { -0.688403361344538f, 0.430252100840336f, -0.26890756302521f },
            { 0.119514472455649f, 0.179271708683473f, 0.26890756302521f },
            { 0.119514472455649f, -0.179271708683473f, 0.26890756302521f },
            { 0.f, 0.f, 1.f } };

        float *__restrict g;
        if (utils::one_of(wino_format_, mkldnn_wino_wei_aaOIoi,
                    mkldnn_wino_wei_aaOio, mkldnn_wino_wei_aaOBiOo))
            g = (float *)G_2x2_3x3;
        else if (wino_format_ == mkldnn_wino_wei_OBaaIBOIio)
            g = (float *)G_4x4_3x3;
        else {
            assert("Unknown winograd weights target layout");
            return;
        }

        int Z = oc_ * ic_;
        assert(r_ == kh_ && r_ == kw_);

        for (int iic = 0; iic < ic_; iic++) {
        for (int ob = 0; ob < nb_oc_; ob++) {
            const in_data_t *__restrict _inp
                    = input + (ob * oc_block_ * or_ic_ + iic) * kh_ * kw_;
            out_data_t *__restrict _out
                    = tmp_wei + (iic * nb_oc_ + ob) * oc_block_;

            for_nd(0, 1, size_wspace_, [&](int i) { wspace[i] = 0.f; });

            for_nd(0, 1, r_, w_alpha_, oc_block_,
                [&](int ih, int j, int ioc) {
                for (int iw = 0; iw < r_; ++iw) {
                    int inp_oc = ob * oc_block_ + ioc;
                    int inp_ic = iic;
                    in_data_t inp_v = (inp_ic < or_ic_ && inp_oc < or_oc_)
                        ? _inp[ioc * or_ic_ * kh_ * kw_ + ih * kw_ + iw]
                        : 0.f;
                    wspace[(ih * w_alpha_ + j) * oc_block_ + ioc]
                            += inp_v * g[j * r_ + iw];
                }
            });

            for_nd(0, 1, w_alpha_, w_alpha_, oc_block_,
                [&](int i, int j, int ioc) {
                float t = 0;
                for (int k = 0; k < r_; ++k)
                    t += g[i * r_ + k]
                            * wspace[(k * w_alpha_ + j) * oc_block_ + ioc];
                if (type_o == data_type::s8) {
                    const float scale = (D_mask == 1)
                        ? scales[0]
                        : scales[ob * oc_block_ + ioc];
                    _out[(i * w_alpha_ + j) * Z + ioc]
                            = qz_b0<in_data_t, out_data_t>()(
                                    (in_data_t)t, scale * adj_scale_);
                } else {
                    _out[(i * w_alpha_ + j) * Z + ioc] = (out_data_t)t;
                }
            });
        }}
    }

    void reorder_to_aaOIoi(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        int32_t *__restrict dst_bias = nullptr;
        if (type_o == data_type::s8) {
            const auto bias_shift = sizeof(out_data_t) * size_wino_wei_;
            const size_t bias_size = w_alpha_ * w_alpha_ * oc_;

            dst_bias = (int32_t *)(output + bias_shift);
            utils::array_set((int32_t *)dst_bias, 0, bias_size);
        }
        int index = 0;
        for (int u_h = 0; u_h < w_alpha_; u_h++) {
        for (int u_w = 0; u_w < w_alpha_; u_w++) {
            for_nd(0, 1, nb_oc_, oc_block_, [&](int ob, int o) {
                int u_h_shift = u_h * w_alpha_ * ic_ * oc_;
                int u_w_shift = u_w * ic_ * oc_;
                int u_h_shift_b = u_h * w_alpha_ * oc_;
                int u_w_shift_b = u_w * oc_;
                int oc_block_shift = ob * oc_block_ * ic_ + o * ic_block_;
                for (int ib = 0; ib < nb_ic_; ib++) {
                for (int i = 0; i < ic_block_; i++) {
                    int _i = ib * ic_block_;
                    int _o = ob * oc_block_;
                    int ic_shift = (_i + i) * oc_;
                    int oc_shift = (_o + o);
                    int ic_block_shift = ib * oc_block_ * ic_block_ + i;
                    int src_offset =
                            u_h_shift + u_w_shift + ic_shift + oc_shift;
                    int dst_offset = u_h_shift + u_w_shift + oc_block_shift
                            + ic_block_shift;

                    output[dst_offset] = tmp_wei[src_offset];
                    if (type_o == data_type::s8) {
                        int bias_offset = u_h_shift_b + u_w_shift_b + oc_shift;
                        if (index != unsign_val_in_wino_domain_)
                            dst_bias[bias_offset]
                                    -= (128 * (int32_t)output[dst_offset]);
                        else
                            dst_bias[bias_offset] = 0;
                    }
                }}
            });
            index++;
        }}
    }

    void reorder_to_aaOio(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        for_nd(0, 1, w_alpha_, w_alpha_, nb_oc_,
            [&](int u_h, int u_w, int ob) {
            for (int ib = 0; ib < nb_ic_; ib++) {
            for (int i = 0; i < ic_block_; i++) {
            for (int o = 0; o < oc_block_; o++) {
                int src_offset = u_h * w_alpha_ * ic_ * oc_ + u_w * ic_ * oc_
                    + (ib * ic_block_ + i) * oc_ + (ob * oc_block_ + o);

                int dst_offset
                    = u_h * w_alpha_ * nb_oc_ * nb_ic_ * ic_block_ * oc_block_
                    + u_w * nb_oc_ * nb_ic_ * ic_block_ * oc_block_
                    + ob * nb_ic_ * ic_block_ * oc_block_
                    + ib * ic_block_ * oc_block_ + i * oc_block_ + o;
                output[dst_offset] = tmp_wei[src_offset];
            }}}
        });
    }

    void reorder_to_aaOBiOo(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        int oc_chunks = nb_oc_ / oc2_block_;

        for_nd(0, 1, w_alpha_, w_alpha_, oc_chunks,
            [&](int u_h, int u_w, int occ) {
            for (int ib = 0; ib < nb_ic_; ib++) {
                out_data_t *__restrict wei_ptr = output
                    + (((u_h * w_alpha_ + u_w) * oc_chunks + occ) * nb_ic_ + ib)
                    * oc2_block_ * ic_block_ * oc_block_;
                int wei_offset = 0;
                for (int i = 0; i < ic_block_; i++) {
                for (int ob2 = 0; ob2 < oc2_block_; ob2++) {
                    for (int o = 0; o < oc_block_; o++) {
                        int icp = ib * ic_block_ + i;
                        int ocp =
                            occ * oc2_block_ * oc_block_ + ob2 * oc_block_ + o;

                        int src_offset = u_h * w_alpha_ * ic_ * oc_
                            + u_w * ic_ * oc_ + icp * oc_ + ocp;
                        wei_ptr[wei_offset + o] = tmp_wei[src_offset];
                    }
                    wei_offset += oc_block_;
                }}
            }
        });
    }

    void reorder_to_OBaaIBOIio(out_data_t *__restrict output,
            const out_data_t *__restrict tmp_wei) const {
        int ic_chunks = nb_ic_ / ic2_block_;
        int oc_chunks = nb_oc_ / oc2_block_;

        for_nd(0, 1, oc_chunks, w_alpha_, w_alpha_,
            [&](int occ, int u_h, int u_w) {
            for (int icc = 0; icc < ic_chunks; icc++) {
            for (int ob = 0; ob < oc2_block_; ob++) {
                int ocp = (occ * oc2_block_ + ob) * oc_block_;
                for (int ib = 0; ib < ic2_block_; ib++) {
                for (int i = 0; i < ic_block_; i++) {
                    int icp = (icc * ic2_block_ + ib) * ic_block_ + i;

                    int src_offset = u_h * w_alpha_ * ic_ * oc_
                        + u_w * ic_ * oc_ + icp * oc_ + ocp;
                    int wei_offset
                        = ((((((occ * w_alpha_ + u_h) * w_alpha_ + u_w)
                            * ic_chunks + icc) * oc2_block_ + ob) * ic2_block_
                            + ib) * ic_block_ + i) * oc_block_;
                    for (int o = 0; o < oc_block_; o++)
                        output[wei_offset + o] = tmp_wei[src_offset + o];
                }}
            }}
        });
    }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, MKLDNN_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, MKLDNN_ARG_TO);

        auto wspace = (in_data_t *__restrict)scratchpad(ctx).template get<void>(
                memory_tracking::names::key_reorder_wino_transform_space);
        auto tmp_wei = (out_data_t *__restrict)scratchpad(ctx).template get<void>(
                memory_tracking::names::key_reorder_wino_plain);

        transform(tmp_wei, input, wspace);

        /* reorder to winograd domain */
        switch (wino_format_) {
        case mkldnn_wino_wei_aaOIoi:
            reorder_to_aaOIoi(output, tmp_wei); break;
        case mkldnn_wino_wei_aaOio:
            reorder_to_aaOio(output, tmp_wei); break;
        case mkldnn_wino_wei_aaOBiOo:
            reorder_to_aaOBiOo(output, tmp_wei); break;
        case mkldnn_wino_wei_OBaaIBOIio:
            reorder_to_OBaaIBOIio(output, tmp_wei); break;
        default: assert("Unknown wino format"); break;
        }

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    int r_, w_alpha_;
    int ic_, oc_, or_ic_, or_oc_, kh_, kw_;
    int oc_block_, ic_block_, oc2_block_, ic2_block_;
    float adj_scale_;
    int nb_oc_, nb_ic_;
    mkldnn_wino_memory_format_t wino_format_;
    int size_wino_wei_;
    int size_wspace_;
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif
