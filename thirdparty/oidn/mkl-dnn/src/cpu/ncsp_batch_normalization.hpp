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

#ifndef CPU_NCSP_BATCH_NORMALIZATION_HPP
#define CPU_NCSP_BATCH_NORMALIZATION_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_batch_normalization_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct ncsp_batch_normalization_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_fwd_pd_t {
        using cpu_batch_normalization_fwd_pd_t::cpu_batch_normalization_fwd_pd_t;

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_fwd_t);

        status_t init() {
            using namespace data_type;
            using namespace prop_kind;
            using namespace format_tag;

            bool ok = true
                && is_fwd()
                && !has_zero_dim_memory()
                && src_md()->data_type == f32
                && IMPLICATION(use_scaleshift(), weights_md()->data_type == f32)
                && memory_desc_matches_one_of_tag(*src_md(), ncdhw, nchw, nc)
                && (attr()->has_default_values() || this->with_relu_post_op());
            if (!ok) return status::unimplemented;

            if (is_training() && fuse_bn_relu()) init_default_ws(8);

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            if (!stats_is_src()) {
                scratchpad.book(key_bnorm_reduction,
                        sizeof(data_t) * C() * mkldnn_get_max_threads());

                if (!is_training()) {
                    scratchpad.book(key_bnorm_tmp_mean, sizeof(data_t) * C());
                    scratchpad.book(key_bnorm_tmp_var, sizeof(data_t) * C());
                }
            }
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ncsp_batch_normalization_fwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    ~ncsp_batch_normalization_fwd_t() {}

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct ncsp_batch_normalization_bwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_batch_normalization_bwd_pd_t {
        using cpu_batch_normalization_bwd_pd_t::cpu_batch_normalization_bwd_pd_t;

        DECLARE_COMMON_PD_T("ncsp_bnorm:any", ncsp_batch_normalization_bwd_t);

        status_t init() {
            using namespace data_type;
            using namespace format_tag;

            bool ok = true
                && is_bwd()
                && !has_zero_dim_memory()
                && utils::everyone_is(f32, src_md()->data_type,
                        diff_src_md()->data_type)
                && IMPLICATION(use_scaleshift(),
                        utils::everyone_is(f32,
                            weights_md()->data_type,
                            diff_weights_md()->data_type))
                && memory_desc_matches_one_of_tag(*src_md(), ncdhw, nchw, nc)
                && memory_desc_matches_one_of_tag(*diff_src_md(), ncdhw, nchw, nc)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            if (fuse_bn_relu()) {
                init_default_ws(8);
                if (!compare_ws(hint_fwd_pd_))
                    return status::unimplemented;
            }

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_bnorm_reduction,
                    sizeof(data_t) * 2 * C() * mkldnn_get_max_threads());
            if (!(use_scaleshift() && desc()->prop_kind == prop_kind::backward))
                scratchpad.book(key_bnorm_tmp_diff_ss,
                        sizeof(data_t) * 2 * C());
        }
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ncsp_batch_normalization_bwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
    ~ncsp_batch_normalization_bwd_t() {}

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
