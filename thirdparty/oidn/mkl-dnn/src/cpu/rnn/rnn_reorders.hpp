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

#ifndef CPU_RNN_REORDERS_HPP
#define CPU_RNN_REORDERS_HPP

#include <assert.h>

#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "simple_q10n.hpp"
#include "cpu_reorder_pd.hpp"
#include "../gemm/os_blas.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t type_i, data_type_t type_o>
struct rnn_data_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_data_reorder", rnn_data_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true
                    && id.data_type() == type_i
                    && od.data_type() == type_o
                    && id.matches_one_of_tag(format_tag::tnc, format_tag::ldsnc)
                    && od == id;
            if (!args_ok) return status::invalid_arguments;

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }
    };

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    rnn_data_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto input = CTX_IN_MEM(const in_data_t *, MKLDNN_ARG_FROM);
        auto output = CTX_OUT_MEM(out_data_t *, MKLDNN_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const size_t nelems = input_d.nelems();
        const float scale = pd()->attr()->rnn_data_qparams_.scale_;
        const float shift = pd()->attr()->rnn_data_qparams_.shift_;

        parallel_nd(nelems, [&](size_t i) {
            float in = (float)input[input_d.off_l(i)] * scale + shift;
            output[output_d.off_l(i)] = qz_a1b0<float, out_data_t>()(in);
        });

        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <data_type_t type_i, data_type_t type_o>
struct rnn_weights_reorder_t : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
#if !USE_MKL_PACKED_GEMM
            return status::unimplemented;
#endif
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true
                    && id.data_type() == type_i
                    && od.data_type() == type_o
                    && od.format_kind() == format_kind::rnn_packed
                    && od.rnn_packed_desc().format == mkldnn_ldigo_p
                    && od.rnn_packed_desc().n_parts == 1
                    && attr != nullptr;
            if (!args_ok) return status::invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi);
            if (itag == format_tag::undef) return status::invalid_arguments;

            const int mask = attr->rnn_weights_qparams_.mask_;
            if (!utils::one_of(mask, 0, 3)) return status::unimplemented;

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return out_of_memory;
            _pd->itag_ = itag;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        status_t init() {
            status_t status = cpu_reorder_pd_t::init();
            if (status != status::success) return status;

            init_scratchpad();

            return status::success;
        }

        format_tag_t itag_ = mkldnn_format_tag_undef;

    private:
        void init_scratchpad() {
            const memory_desc_wrapper id(src_md());
            const size_t nelems = id.nelems();
            const auto &dims = id.dims();

            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            size_t quantization_size = sizeof(int8_t) * nelems;
            size_t reduction_size = itag_ == ldigo
                    ? sizeof(int32_t) * mkldnn_get_max_threads() * dims[0]
                            * dims[1] * dims[3] * dims[4]
                    : 0;
            scratchpad.book(
                    key_reorder_rnn_weights_quantization, quantization_size);
            scratchpad.book(key_reorder_rnn_weights_reduction, reduction_size);
        }
    };

private:
    typedef typename prec_traits<type_i>::type in_data_t;
    typedef typename prec_traits<type_o>::type out_data_t;

    rnn_weights_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
#if USE_MKL_PACKED_GEMM
        auto input = CTX_IN_MEM(const in_data_t *, MKLDNN_ARG_FROM);
        auto output = CTX_OUT_MEM(char *, MKLDNN_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const auto &dims = input_d.dims();

        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        const bool is_igo = pd()->itag_ == format_tag::ldigo;

        /* Quantize input & compute compensation */
        auto quantized = (int8_t * __restrict)scratchpad(ctx).template get<void>(
                memory_tracking::names::key_reorder_rnn_weights_quantization);
        auto reduction = (int32_t * __restrict)scratchpad(ctx).template get<void>(
                memory_tracking::names::key_reorder_rnn_weights_reduction);
        float *comp = reinterpret_cast<float *>(
                output + output_d.rnn_packed_desc().offset_compensation);
        const float *scales = pd()->attr()->rnn_weights_qparams_.scales_;
        const int mask = pd()->attr()->rnn_weights_qparams_.mask_;

        if (is_igo) {
            int nthr = mkldnn_get_max_threads();
            int LD_nthr = nstl::min(L * D, nthr);
            int I_nthr = nstl::min(I, nthr / LD_nthr);
            parallel(nthr, [&](const int ithr, const int nthr) {
                int LD_ithr = -1, LD_s = -1, LD_e = -1;
                int I_ithr = -1, I_s = -1, I_e = -1;
                if (ithr < LD_nthr * I_nthr) {
                    LD_ithr = ithr % LD_nthr;
                    I_ithr = ithr / LD_nthr;
                    balance211(L * D, LD_nthr, LD_ithr, LD_s, LD_e);
                    balance211(I, I_nthr, I_ithr, I_s, I_e);
                }
                int32_t *comp_ithr = reduction + I_ithr * L * D * G * O;
                for (int ld = LD_s; ld < LD_e; ld++) {
                    for (int go = 0; go < G * O; go++)
                        comp_ithr[ld * G * O + go] = 0;
                    for (int i = I_s; i < I_e; i++) {
                        PRAGMA_OMP_SIMD()
                        for (int go = 0; go < G * O; go++) {
                            const float s = scales[(mask == 0) ? 0 : go];
                            int8_t q = qz_b0<in_data_t, out_data_t>()(
                                    input[ld * I * G * O + i * G * O + go], s);
                            quantized[ld * I * G * O + i * G * O + go]
                                    = (int32_t)q;
                            comp_ithr[ld * G * O + go] += (int32_t)q;
                        }
                    }
                }
            });
            parallel_nd(L * D * G * O,
                    [&](int s) { comp[s] = saturate<float>(reduction[s]); });
            for (int i = 1; i < I_nthr; i++) {
                parallel_nd(L * D * G * O, [&](int s) {
                    comp[s] += saturate<float>(
                            reduction[i * L * D * G * O + s]);
                });
            }
        } else {
            parallel_nd(L * D, G * O, [&](int ld, int go) {
                int32_t compensation = 0;
                const float s = scales[(mask == 0) ? 0 : go];
                PRAGMA_OMP_SIMD()
                for (int i = 0; i < I; i++) {
                    int8_t q = qz_b0<in_data_t, out_data_t>()(
                            input[ld * G * O * I + go * I + i], s);
                    compensation += (int32_t)q;
                    quantized[ld * G * O * I + go * I + i] = q;
                }
                comp[ld * G * O + go] = saturate<float>(compensation);
            });
        }

        /* Pack */
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };
        int n_parts = output_d.rnn_packed_desc().n_parts;
        const size_t *size_packed_cell
                = output_d.rnn_packed_desc().part_pack_size;
        const int *parts = output_d.rnn_packed_desc().parts;
        const int n = output_d.rnn_packed_desc().n;
        char *to_pack = output;
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = parts[p] * O;
                    int k_p = I;
                    cblas_gemm_s8u8s32_pack(CblasColMajor, CblasAMatrix,
                            is_igo ? CblasNoTrans : CblasTrans, m_p, n, k_p,
                            &quantized[is_igo ? off_igo(l, d, 0, g, 0) :
                                                off_goi(l, d, g, 0, 0)],
                            is_igo ? G * O : I, to_pack);
                    to_pack += size_packed_cell[p];
                }
            }
        }
#endif
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <>
struct rnn_weights_reorder_t<data_type::f32, data_type::f32>
        : public cpu_primitive_t {
    struct pd_t : public cpu_reorder_pd_t {
        using cpu_reorder_pd_t::cpu_reorder_pd_t;

        DECLARE_COMMON_PD_T("rnn_weights_reorder", rnn_weights_reorder_t);

        static status_t create(reorder_pd_t **reorder_pd,
                engine_t *engine, const primitive_attr_t *attr,
                engine_t *src_engine, const memory_desc_t *src_md,
                engine_t *dst_engine, const memory_desc_t *dst_md) {
#if !USE_MKL_PACKED_GEMM
            return status::unimplemented;
#endif
            const memory_desc_wrapper id(src_md), od(dst_md);
            bool args_ok = true
                    && id.data_type() == data_type::f32
                    && od.data_type() == data_type::f32
                    && od.format_kind() == format_kind::rnn_packed
                    && utils::one_of(od.rnn_packed_desc().format,
                        mkldnn_ldigo_p, mkldnn_ldgoi_p)
                    && attr->has_default_values();
            if (!args_ok) return status::invalid_arguments;

            format_tag_t itag = id.matches_one_of_tag(
                    format_tag::ldigo, format_tag::ldgoi);
            if (itag == format_tag::undef) return status::invalid_arguments;

            const int mask = attr->rnn_weights_qparams_.mask_;
            if (!utils::one_of(mask, 0, 3)) return status::unimplemented;

            auto _pd = new pd_t(engine, attr, src_engine, src_md, dst_engine,
                    dst_md);
            if (_pd == nullptr) return out_of_memory;
            if (_pd->init() != success) { delete _pd; return unimplemented; }
            _pd->itag_ = itag;
            return safe_ptr_assign<reorder_pd_t>(*reorder_pd, _pd);
        }

        format_tag_t itag_;
    };

private:
    rnn_weights_reorder_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
#if USE_MKL_PACKED_GEMM
        auto input = CTX_IN_MEM(const float *, MKLDNN_ARG_FROM);
        auto output = CTX_OUT_MEM(float *, MKLDNN_ARG_TO);
        const memory_desc_wrapper &input_d = pd()->src_md();
        const memory_desc_wrapper &output_d = pd()->dst_md();
        const auto &dims = input_d.dims();
        const rnn_packed_desc_t &rnn_pdata = output_d.rnn_packed_desc();
        const int L = dims[0];
        const int D = dims[1];
        const int I = dims[2];
        const int G = dims[3];
        const int O = dims[4];

        /* Pack */
        bool cross_case = false
            || (pd()->itag_ == format_tag::ldigo
                    && rnn_pdata.format == mkldnn_ldgoi_p)
            || (pd()->itag_ == format_tag::ldgoi
                    && rnn_pdata.format == mkldnn_ldigo_p);
        auto trans = cross_case ? CblasTrans : CblasNoTrans;
        int n_parts = rnn_pdata.n_parts;
        const size_t *size_packed_cell = rnn_pdata.part_pack_size;
        const int *parts = rnn_pdata.parts;
        const int n = rnn_pdata.n;

        const bool is_igo = pd()->itag_ == format_tag::ldigo;
        auto off_igo = [&](int l, int d, int i, int g, int o) {
            return l * D * I * G * O + d * I * G * O + i * G * O + g * O + o;
        };
        auto off_goi = [&](int l, int d, int i, int g, int o) {
            return l * D * G * O * I + d * G * O * I + g * O * I + o * I + i;
        };
        for (int l = 0; l < L; l++) {
            for (int d = 0; d < D; d++) {
                for (int p = 0; p < n_parts; p++) {
                    int g = (p > 0) ? parts[p - 1] : 0;
                    int m_p = is_igo ? parts[p] * O : I;
                    int k_p = is_igo ? I : parts[p] * O;
                    int ld = is_igo ? G * O : I;
                    cblas_sgemm_pack(CblasColMajor, CblasAMatrix, trans, m_p, n,
                            k_p, 1.0f, &input[is_igo ? off_igo(l, d, 0, g, 0) :
                                                       off_goi(l, d, 0, g, 0)],
                            ld, output);
                    output += size_packed_cell[p] / sizeof(float);
                }
            }
        }
#endif
        return status::success;
    }

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

} // namespace cpu
} // namespace impl
} // namespace mkldnn

#endif
