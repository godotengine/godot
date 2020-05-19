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

#ifndef CPU_JIT_AVX512_CORE_U8S8S32X_WINO_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_U8S8S32X_WINO_CONVOLUTION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"

#include "jit_primitive_conf.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t;
struct jit_avx512_core_u8s8s32x_wino_conv_src_trans_t;
struct jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t;

template <data_type_t dst_data_type>
struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            :  cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8_wino:", avx512_core, ""),
                jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>);

        status_t init() {
            bool ok = true
                && is_fwd()
                && utils::one_of(desc()->alg_kind,
                           alg_kind::convolution_auto,
                           alg_kind::convolution_winograd)
                && expect_data_types(data_type::u8, data_type::s8,
                        data_type::undef, dst_data_type, data_type::s32)
                && IMPLICATION(with_bias(), utils::one_of(
                            desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                && !has_zero_dim_memory()
                && set_default_formats();

            if (!ok) return status::unimplemented;

            status_t status = jit_conf();
            if (status != status::success) return status;
            set_default_alg_kind(alg_kind::convolution_winograd);

            init_scratchpad();

            return status;
        }

        jit_conv_conf_2x3_wino_t jcp_;

    protected:
        status_t jit_conf();
        void init_scratchpad();

        bool set_default_formats() {
            using namespace format_tag;
            return set_default_formats_common(nhwc, any, nhwc);
        }
    };

    typedef typename prec_traits<data_type::u8>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;
    typedef typename prec_traits<dst_data_type>::type dst_data_t;

    jit_avx512_core_u8s8s32x_wino_convolution_fwd_t(const pd_t *apd);
    ~jit_avx512_core_u8s8s32x_wino_convolution_fwd_t();

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    const float *adjust_oscales(const memory_tracking::grantor_t &scratchpad)
        const;
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_small_mb(const src_data_t *src, const wei_data_t *wei,
            const char *bia, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_mbN(const src_data_t *src, const wei_data_t *wei,
            const char *bia, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t *kernel_;
    jit_avx512_core_u8s8s32x_wino_conv_src_trans_t *src_trans_;
    jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t *dst_trans_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
