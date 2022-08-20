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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP

#include "c_types_map.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "primitive_iterator.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_deconvolution_pd.hpp"
#include "cpu_primitive.hpp"

#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_x8s8s32x_1x1_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t
        : public cpu_primitive_t {
    struct pd_t : public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr) {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
        {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(),
                jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t<src_type, dst_type>);

        status_t init_convolution() {
            convolution_desc_t cd;
            status_t status;

            auto dd = desc();
            status = conv_desc_init(&cd, prop_kind::forward_training,
                    alg_kind::convolution_direct, &(dd->src_desc),
                    &(dd->weights_desc), &(dd->bias_desc), &(dd->dst_desc),
                    dd->strides, dd->dilates, dd->padding[0], dd->padding[1],
                    dd->padding_kind);

            if (status == status::success) {
                status = mkldnn_primitive_desc::create<conv_pd_t>(
                        &conv_pd_, (op_desc_t *)&cd, &attr_, engine_, nullptr);
            }

            if (status == status::success)
                status = set_default_params();

            return status;
        };

        status_t init() {
            bool ok = true
                && is_fwd()
                && desc()->alg_kind == alg_kind::deconvolution_direct
                && !has_zero_dim_memory()
                && desc()->src_desc.data_type == src_type
                && desc()->dst_desc.data_type == dst_type
                && desc()->weights_desc.data_type == data_type::s8
                && IMPLICATION(with_bias(), utils::one_of(
                            desc()->bias_desc.data_type, data_type::f32,
                            data_type::s32, data_type::s8, data_type::u8))
                && desc()->accum_data_type == data_type::s32;
            if (!ok) return status::unimplemented;

            CHECK(init_convolution());

            return status::success;
        }

        virtual void init_scratchpad_md() override {
            const auto conv_1x1_pd = static_cast<conv_pd_t *>(conv_pd_);
            scratchpad_md_ = *conv_1x1_pd->scratchpad_md();
        }

    protected:
        status_t set_default_params() {
            auto conv_1x1_pd_ = static_cast<conv_pd_t *>(conv_pd_);
            src_md_ = *conv_1x1_pd_->src_md();
            dst_md_ = *conv_1x1_pd_->dst_md();
            weights_md_ = *conv_1x1_pd_->weights_md();
            if (with_bias())
                bias_md_ = *conv_1x1_pd_->weights_md(1);
            return status::success;
        }

        using conv_pd_t = typename jit_avx512_core_x8s8s32x_1x1_convolution_fwd_t
            <src_type, dst_type>::pd_t;
        friend jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t;
        primitive_desc_t *conv_pd_;
    };

    jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }

    ~jit_avx512_core_x8s8s32x_1x1_deconvolution_fwd_t()
    { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        return conv_p_->execute(ctx);
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

}
}
}

#endif /* CPU_JIT_AVX512_CORE_X8S8S32X_1X1_DECONVOLUTION_HPP */
