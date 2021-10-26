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

#ifndef CPU_JIT_UNI_I8I8_POOLING_HPP
#define CPU_JIT_UNI_I8I8_POOLING_HPP

#include "c_types_map.hpp"

#include "cpu_pooling_pd.hpp"
#include "cpu_primitive.hpp"

#include "cpu_isa_traits.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t;

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", isa, ""),
                jit_uni_i8i8_pooling_fwd_t<isa>);

        status_t init() {
            bool ok = true
                && mayiuse(isa)
                && ndims() == 4
                && set_default_params() == status::success
                && desc()->prop_kind == prop_kind::forward_inference
                && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                        alg_kind::pooling_avg_include_padding,
                        alg_kind::pooling_avg_exclude_padding)
                && utils::one_of(src_md()->data_type, data_type::s32,
                        data_type::s8, data_type::u8)
                && src_md()->data_type == dst_md()->data_type
                && attr()->has_default_values()
                && memory_desc_matches_tag(*src_md(), format_tag::nhwc)
                && memory_desc_matches_tag(*dst_md(), format_tag::nhwc);
            if (!ok) return status::unimplemented;

            return jit_conf();
        }

        jit_pool_conf_t jpp_;

    protected:
        status_t jit_conf();
    };

    jit_uni_i8i8_pooling_fwd_t(const pd_t *apd);
    ~jit_uni_i8i8_pooling_fwd_t();

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_uni_i8i8_pooling_fwd_ker_t<isa> *ker_;
};

}
}
}

#endif
