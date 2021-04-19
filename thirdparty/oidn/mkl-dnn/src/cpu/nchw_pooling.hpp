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

#ifndef CPU_NCHW_POOLING_HPP
#define CPU_NCHW_POOLING_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_pooling_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct nchw_pooling_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_fwd_pd_t {
        using cpu_pooling_fwd_pd_t::cpu_pooling_fwd_pd_t;

        DECLARE_COMMON_PD_T("nchw_pooling:any", nchw_pooling_fwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag =
                ndims() == 4 ? format_tag::nchw : format_tag::ncdhw;

            bool ok = true
                && set_default_params() == status::success
                && is_fwd()
                && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                        alg_kind::pooling_avg_include_padding,
                        alg_kind::pooling_avg_exclude_padding)
                && !has_zero_dim_memory()
                && utils::everyone_is(data_type, src_md()->data_type,
                        dst_md()->data_type)
                && attr()->has_default_values()
                && memory_desc_matches_tag(*src_md(), desired_fmt_tag)
                && memory_desc_matches_tag(*dst_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            bool is_training = desc_.prop_kind == prop_kind::forward_training;
            if (desc()->alg_kind == alg_kind::pooling_max && is_training)
                init_default_ws();

            return status::success;
        }
    };

    nchw_pooling_fwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
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
struct nchw_pooling_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_pooling_bwd_pd_t {
        using cpu_pooling_bwd_pd_t::cpu_pooling_bwd_pd_t;

        DECLARE_COMMON_PD_T("nchw:any", nchw_pooling_bwd_t);

        status_t init() {
            const format_tag_t desired_fmt_tag =
                ndims() == 4 ? format_tag::nchw : format_tag::ncdhw;

            bool ok = true
                && set_default_params() == status::success
                && !is_fwd()
                && utils::one_of(desc()->alg_kind, alg_kind::pooling_max,
                        alg_kind::pooling_avg_include_padding,
                        alg_kind::pooling_avg_exclude_padding)
                && !has_zero_dim_memory()
                && utils::everyone_is(data_type,
                        diff_dst_md()->data_type,
                        diff_src_md()->data_type)
                && attr()->has_default_values()
                && memory_desc_matches_tag(*diff_dst_md(), desired_fmt_tag)
                && memory_desc_matches_tag(*diff_src_md(), desired_fmt_tag);
            if (!ok) return status::unimplemented;

            if (desc()->alg_kind == alg_kind::pooling_max) {
                bool ws_ok = true
                    && hint_fwd_pd_
                    && hint_fwd_pd_->workspace_md();
                if (!ws_ok)
                    return status::unimplemented;

                const auto &ws_blk =
                    hint_fwd_pd_->workspace_md()->format_desc.blocking;
                ws_ok = ws_ok
                    && ws_blk.inner_nblks < 1
                    && IMPLICATION(ws_blk.inner_nblks == 1,
                            ws_blk.inner_idxs[0] == 1);
                if (!ws_ok)
                    return status::unimplemented;

                ws_md_ = *hint_fwd_pd_->workspace_md();
            }

            return status::success;
        }
    };

    nchw_pooling_bwd_t(const pd_t *apd): cpu_primitive_t(apd) {}
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
