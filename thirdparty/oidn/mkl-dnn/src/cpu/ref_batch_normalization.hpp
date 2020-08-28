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

#ifndef CPU_REF_BATCH_NORMALIZATION_HPP
#define CPU_REF_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_batch_normalization_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_batch_normalization_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_fwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_batch_normalization_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && src_md()->data_type == data_type
                && IMPLICATION(use_scaleshift(),
                        weights_md()->data_type == data_type::f32)
                && (attr()->has_default_values() || with_relu_post_op());
            if (!ok) return status::unimplemented;

            if (src_md()->data_type == data_type::s8 && !stats_is_src())
                return status::unimplemented;

            if (is_training() && fuse_bn_relu()) init_default_ws(8);

            return status::success;
        }
    };

    ref_batch_normalization_fwd_t(const pd_t *apd): cpu_primitive_t(apd) {}

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
struct ref_batch_normalization_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_batch_normalization_bwd_pd_t {
        pd_t(engine_t *engine, const batch_normalization_desc_t *adesc,
                const primitive_attr_t *attr,
                const batch_normalization_fwd_pd_t *hint_fwd_pd)
            : cpu_batch_normalization_bwd_pd_t(engine, adesc, attr, hint_fwd_pd)
        {}

        DECLARE_COMMON_PD_T("ref:any", ref_batch_normalization_bwd_t);

        status_t init() {
            bool ok = true
                && is_bwd()
                && utils::everyone_is(data_type, src_md()->data_type,
                        diff_src_md()->data_type)
                && IMPLICATION(use_scaleshift(), utils::everyone_is(data_type,
                            weights_md()->data_type,
                            diff_weights_md()->data_type))
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (fuse_bn_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_))
                    return status::unimplemented;
            }

            return status::success;
        }
    };

    ref_batch_normalization_bwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward(ctx);
        return status::success;
    }

private:
    void execute_backward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
