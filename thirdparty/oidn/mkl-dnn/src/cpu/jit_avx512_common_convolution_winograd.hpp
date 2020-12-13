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

#ifndef CPU_JIT_AVX512_COMMON_CONVOLUTION_WINOGRAD_HPP
#define CPU_JIT_AVX512_COMMON_CONVOLUTION_WINOGRAD_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace winograd_avx512_common {
inline void init_scratchpad(memory_tracking::registrar_t &scratchpad,
        const jit_conv_winograd_conf_t &jcp) {
    using namespace memory_tracking::names;

    size_t U_sz = (size_t)alpha * alpha * jcp.ic * jcp.oc;
    size_t V_sz = (size_t)alpha * alpha * jcp.mb * jcp.ic
        * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding);
    size_t M_sz = (size_t)alpha * alpha * jcp.mb * jcp.oc
        * (jcp.itiles * jcp.jtiles + jcp.tile_4fma_padding);

    scratchpad.book(key_wino_U, sizeof(float) * U_sz, PAGE_2M);
    scratchpad.book(key_wino_V, sizeof(float) * V_sz, PAGE_2M);
    scratchpad.book(key_wino_M, sizeof(float) * M_sz, PAGE_2M);

    if (jcp.sched_policy == WSCHED_WEI_S_D_G_W) {
        const int nthr = mkldnn_get_max_threads();

        size_t tr_src_sz = jcp.ver != ver_4fma ? 0 : (size_t)nthr
            * alpha * alpha * jcp.tile_4fma * jcp.ic_simd_block;
        scratchpad.book(key_conv_tr_src, sizeof(float) * tr_src_sz, PAGE_2M);

        size_t br_sz = jcp.with_bias ? nthr * jcp.oc : 0;
        scratchpad.book(key_conv_bia_reduction, sizeof(float) * br_sz, PAGE_2M);

        size_t padded_bias_sz =
            jcp.with_bias && jcp.oc_without_padding != jcp.oc ? jcp.oc : 0;
        scratchpad.book(key_conv_padded_bias, sizeof(float) * padded_bias_sz);
    }
}
}

template <bool is_fwd>
struct _jit_avx512_common_convolution_winograd_t {
    _jit_avx512_common_convolution_winograd_t(
            const jit_conv_winograd_conf_t &jcp, const primitive_attr_t *attr)
        : kernel_(nullptr), attr_(attr) {
        kernel_ = new _jit_avx512_common_conv_winograd_data_kernel_f32(jcp);
    }

    ~_jit_avx512_common_convolution_winograd_t() { delete kernel_; }

    protected:
        void _execute_data_W_S_G_D(float *inp_ptr, float *out_ptr,
                float *wei_ptr, float *bias_ptr,
                const memory_tracking::grantor_t &scratchpad) const;
        _jit_avx512_common_conv_winograd_data_kernel_f32 *kernel_;
        const primitive_attr_t *attr_;
};

struct jit_avx512_common_convolution_winograd_fwd_t
     : _jit_avx512_common_convolution_winograd_t<true>
     , public cpu_primitive_t
    {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_common, ""),
                jit_avx512_common_convolution_winograd_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && utils::one_of(desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_winograd)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_common_conv_winograd_fwd_kernel_f32::
                init_conf(jcp_, *desc(), *src_md(), *weights_md(), *dst_md(),
                        *attr());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_common::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_tag = with_groups() ? gOIhw16i16o : OIhw16i16o;
            return set_default_formats_common(nChw16c, wei_tag, nChw16c);
        }
    };

    jit_avx512_common_convolution_winograd_fwd_t(const pd_t *apd)
        : _jit_avx512_common_convolution_winograd_t<true>(apd->jcp_, apd->attr())
        , cpu_primitive_t(apd, true) {}

    ~jit_avx512_common_convolution_winograd_fwd_t(){};

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override
    {
        auto src = CTX_IN_MEM(const float *, MKLDNN_ARG_SRC);
        auto weights = CTX_IN_MEM(const float *, MKLDNN_ARG_WEIGHTS);
        auto bias = CTX_IN_MEM(const float *, MKLDNN_ARG_BIAS);
        auto dst = CTX_OUT_MEM(float *, MKLDNN_ARG_DST);
        this->_execute_data_W_S_G_D((float *)src, dst, (float *)weights,
                (float *)bias, this->scratchpad(ctx));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct jit_avx512_common_convolution_winograd_bwd_data_t
        : _jit_avx512_common_convolution_winograd_t<false>,
        public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_common, ""),
                jit_avx512_common_convolution_winograd_bwd_data_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::undef, data_type::f32, data_type::f32)
                && utils::one_of(desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_winograd)
                && !has_zero_dim_memory()
                && set_default_formats()
                && mkldnn_thr_syncable();
            if (!ok) return status::unimplemented;

            status_t status =
                jit_avx512_common_conv_winograd_bwd_data_kernel_f32::init_conf(
                        jcp_, *desc(), *diff_src_md(), *weights_md(),
                        *diff_dst_md());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_common::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_tag = with_groups() ? gOIhw16i16o : OIhw16i16o;
            return set_default_formats_common(nChw16c, wei_tag, nChw16c);
        }
    };

    jit_avx512_common_convolution_winograd_bwd_data_t(const pd_t *apd)
        : _jit_avx512_common_convolution_winograd_t<false>(apd->jcp_, apd->attr())
        , cpu_primitive_t(apd, true) {}

    ~jit_avx512_common_convolution_winograd_bwd_data_t(){};

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        auto diff_dst = CTX_IN_MEM(const float *, MKLDNN_ARG_DIFF_DST);
        auto weights = CTX_IN_MEM(const float *, MKLDNN_ARG_WEIGHTS);
        auto diff_src = CTX_OUT_MEM(float *, MKLDNN_ARG_DIFF_SRC);
        this->_execute_data_W_S_G_D((float *)diff_dst, diff_src,
                (float *)weights, nullptr, this->scratchpad(ctx));
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct jit_avx512_common_convolution_winograd_bwd_weights_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr,
                    hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_wino:", avx512_common, ""),
                jit_avx512_common_convolution_winograd_bwd_weights_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && utils::one_of(desc()->alg_kind,
                        alg_kind::convolution_auto,
                        alg_kind::convolution_winograd)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats()
                && mkldnn_thr_syncable();
            if (!ok) return status::unimplemented;

            status_t status =
                jit_avx512_common_conv_winograd_bwd_weights_kernel_f32::
                init_conf(jcp_, *desc(), *src_md(), *diff_dst_md(),
                        *diff_weights_md());
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            auto scratchpad = scratchpad_registry().registrar();
            winograd_avx512_common::init_scratchpad(scratchpad, jcp_);

            return status;
        }

        jit_conv_winograd_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            auto wei_tag = with_groups() ? gOIhw16i16o : OIhw16i16o;
            return set_default_formats_common(nChw16c, wei_tag, nChw16c);
        }
    };

    jit_avx512_common_convolution_winograd_bwd_weights_t(const pd_t *apd)
        : cpu_primitive_t(apd, true), kernel_(nullptr)
    {
        kernel_ = new jit_avx512_common_conv_winograd_bwd_weights_kernel_f32(
                pd()->jcp_);
    }

    ~jit_avx512_common_convolution_winograd_bwd_weights_t()
    { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override
    {
        _execute_backward_weights_S_D_G_W(ctx, scratchpad(ctx));
        return status::success;
    }

private:
    void _execute_backward_weights_S_D_G_W(const exec_ctx_t &ctx,
            const memory_tracking::grantor_t &scratchpad) const;
    void _maybe_execute_diff_bias_copy(float *diff_bias,
            const memory_tracking::grantor_t &scratchpad) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    jit_avx512_common_conv_winograd_bwd_weights_kernel_f32 *kernel_;
};

void trans_W_4x4_3x3(float Fw_[6][6][16][16], float F[3][3][16][16]);
void trans_O_4x4_3x3(float Mw[6][6][16], float O[4][4][16]);
void trans_W_3x3_4x4(float Fw[6][6][16], float F[4][6][16]);
void trans_O_3x3_4x4(float Mw[6][6][16][16], float M[3][3][16][16]);
void trans_I_4x4_3x3(float Iw[6][6][16], float I[6][6][16]);
void trans_W_3x3_4x4_wu(float Fw[6][6][16], float F[4][6][16]);
void trans_O_3x3_4x4_wu(float Mw[6][6][16][16], float M[3][3][16][16]);

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
