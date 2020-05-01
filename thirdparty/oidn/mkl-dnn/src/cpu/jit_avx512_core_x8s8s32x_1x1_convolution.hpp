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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_1X1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"

#include "jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"
#include "jit_uni_1x1_conv_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_int8_1x1:", avx512_core, ""),
                jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t<
                src_type, dst_type>);

        status_t init() {
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(src_type, data_type::s8, data_type::undef,
                        dst_type, data_type::s32)
                && IMPLICATION(with_bias(), utils::one_of(
                            desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                && !has_zero_dim_memory()
                && set_default_formats_common(dat_tag(), format_tag::any,
                        dat_tag())
                && set_or_check_wei_format();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md());

            status_t status = jit_avx512_core_x8s8s32x_1x1_conv_kernel::
                init_conf(jcp_, *conv_d, *src_d, *weights_md(), *dst_md(),
                        *weights_md(1), *attr(), mkldnn_get_max_threads(),
                        rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
                    scratchpad, jcp_, *attr());

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        format_tag_t dat_tag() const { return format_tag::nhwc; }

        bool set_or_check_wei_format() {
            using namespace format_tag;

            const bool is_src_s8 = src_md_.data_type == data_type::s8;
            format_tag_t wei_tag = with_groups() ? gOIhw4i16o4i : OIhw4i16o4i;

            memory_desc_t want_wei_md = weights_md_;
            memory_desc_init_by_tag(want_wei_md, wei_tag);
            if (is_src_s8) {
                want_wei_md.extra.flags = 0
                    | memory_extra_flags::compensation_conv_s8s8
                    | memory_extra_flags::scale_adjust;
                want_wei_md.extra.compensation_mask = (1 << 0)
                    + (with_groups() ? (1 << 1) : 0);
                want_wei_md.extra.scale_adjust =
                    mayiuse(avx512_core_vnni) ? 1.f : 0.5f;
            }

            if (weights_md_.format_kind == format_kind::any) {
                weights_md_ = want_wei_md;
                return true;
            }

            return weights_md_ == want_wei_md;
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx512_core_x8s8s32x_1x1_conv_kernel(pd()->jcp_,
                    *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

  private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const char *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_core_x8s8s32x_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

}
}
}

#endif
