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

#ifndef CPU_GEMM_INNER_PRODUCT_HPP
#define CPU_GEMM_INNER_PRODUCT_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "gemm/gemm.hpp"

#include "cpu_inner_product_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct gemm_inner_product_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_fwd_pd_t {
        using cpu_inner_product_fwd_pd_t::cpu_inner_product_fwd_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_fwd_t);

        status_t init() {
            using namespace utils;

            bool ok = true
                && set_default_params() == status::success
                && is_fwd()
                && !has_zero_dim_memory()
                && everyone_is(data_type,
                        src_md()->data_type,
                        weights_md()->data_type,
                        dst_md()->data_type,
                        with_bias() ? weights_md(1)->data_type : data_type)
                && attr()->output_scales_.has_default_values()
                && attr()->post_ops_.len_ <= 1
                && IMPLICATION(attr()->post_ops_.len_ == 1,
                        attr()->post_ops_.entry_[0].is_relu(true, false))
                && dense_gemm_consitency_check(src_md(), weights_md(),
                        dst_md());
            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_fwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_data_pd_t {
        using cpu_inner_product_bwd_data_pd_t::cpu_inner_product_bwd_data_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_data_t);

        status_t init() {
            bool ok = true
                && set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_data
                && !has_zero_dim_memory()
                && utils::everyone_is(data_type,
                        diff_src_md()->data_type,
                        weights_md()->data_type,
                        diff_dst_md()->data_type)
                && attr()->has_default_values()
                && dense_gemm_consitency_check(diff_src_md(), weights_md(),
                        diff_dst_md());
            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_data_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

template <impl::data_type_t data_type>
struct gemm_inner_product_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_inner_product_bwd_weights_pd_t {
        using cpu_inner_product_bwd_weights_pd_t::cpu_inner_product_bwd_weights_pd_t;

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_inner_product_bwd_weights_t);

        status_t init() {
            bool ok = true
                && set_default_params() == status::success
                && desc()->prop_kind == prop_kind::backward_weights
                && !has_zero_dim_memory()
                && utils::everyone_is(data_type,
                        src_md()->data_type,
                        diff_weights_md()->data_type,
                        diff_dst_md()->data_type,
                        with_bias() ? diff_weights_md(1)->data_type : data_type)
                && attr()->has_default_values()
                && dense_gemm_consitency_check(src_md(), diff_weights_md(),
                        diff_dst_md());

            return ok ? status::success : status::unimplemented;
        }
    };

    gemm_inner_product_bwd_weights_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
