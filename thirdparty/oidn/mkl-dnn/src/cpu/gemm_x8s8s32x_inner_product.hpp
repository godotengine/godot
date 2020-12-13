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

#ifndef GEMM_X8S8S32X_INNER_PRODUCT_HPP
#define GEMM_X8S8S32X_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "gemm/gemm.hpp"
#include "jit_generator.hpp"

#include "cpu_inner_product_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type, impl::data_type_t dst_type>
struct gemm_x8s8s32x_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T(src_type == data_type::u8
                ? IGEMM_S8U8S32_IMPL_STR
                : IGEMM_S8S8S32_IMPL_STR,
                gemm_x8s8s32x_inner_product_fwd_t);

        status_t init() {
            using namespace data_type;

            bool ok = true
                && set_default_params() == status::success
                && is_fwd()
                && !has_zero_dim_memory()
                && src_md()->data_type == src_type
                && dst_md()->data_type == dst_type
                && weights_md()->data_type == s8
                && IMPLICATION(with_bias(), utils::one_of(
                            weights_md(1)->data_type, f32, s32, s8, u8))
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_,
                        attr()->post_ops_.entry_[0].is_relu(true, false))
                && dense_gemm_consitency_check(src_md(), weights_md(),
                        dst_md());
            if (!ok) return status::unimplemented;

            dst_is_acc_ = utils::one_of(dst_type, s32, f32);

            init_scratchpad();

            return status::success;
        }

        bool dst_is_acc_;

    protected:
        status_t set_default_params() {
            using namespace format_tag;
            if (src_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(src_md_,
                            utils::pick(ndims() - 2, nc, nwc, nhwc, ndhwc)));
            }
            if (dst_md_.format_kind == format_kind::any)
                CHECK(memory_desc_init_by_tag(dst_md_, nc));
            if (weights_md_.format_kind == format_kind::any) {
                CHECK(memory_desc_init_by_tag(weights_md_,
                            utils::pick(ndims() - 2, io, wio, hwio, dhwio)));
            }
            return inner_product_fwd_pd_t::set_default_params();
        }

    private:
        void init_scratchpad() {
            if (!dst_is_acc_) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(
                        memory_tracking::names::key_iprod_int_dat_in_acc_dt,
                        sizeof(acc_data_t) * MB() * OC());
            }
        }
    };

    gemm_x8s8s32x_inner_product_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd, true)
    { pp_kernel_ = new pp_kernel_t(apd, pd()->dst_is_acc_); }
    ~gemm_x8s8s32x_inner_product_fwd_t() { delete pp_kernel_; }

    typedef typename prec_traits<dst_type>::type data_t;

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<data_type::s8>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;
    typedef typename prec_traits<data_type::s32>::type acc_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    // XXX: this is throwaway code that will become unnecessary when we have a
    // sufficiently advanced igemm jit generator that supports quantization,
    // relu, and whatnot
    class pp_kernel_t: jit_generator {
    public:
        DECLARE_CPU_JIT_AUX_FUNCTIONS(
                gemm_x8s8s32x_inner_product_fwd_t::pp_kernel);
        pp_kernel_t(const pd_t *pd, bool dst_is_acc);

        void operator()(dst_data_t *dst, const acc_data_t *acc,
                const char *bias, const float *scales, float nslope,
                size_t start, size_t end);
    private:
        void generate();

        struct ker_args {
            dst_data_t *dst;
            const acc_data_t *acc;
            const char *bias;
            const float *scales;
            float nslope;
            size_t len;
            size_t oc_offset;
        };
        void (*ker_)(const ker_args *args);

        size_t OC_;
        data_type_t bias_data_type_;
        size_t bias_data_type_size_;
        size_t scale_idx_mult_;
        bool do_bias_;
        bool do_relu_;
    };

    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    pp_kernel_t *pp_kernel_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
