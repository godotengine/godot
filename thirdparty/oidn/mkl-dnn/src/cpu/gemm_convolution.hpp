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

#ifndef CPU_JIT_GEMM_CONVOLUTION_HPP
#define CPU_JIT_GEMM_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "gemm_convolution_utils.hpp"
#include "gemm/gemm.hpp"
#include "ref_eltwise.hpp"

#include "cpu_convolution_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct gemm_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats_common(dat_tag(), wei_tag(), dat_tag())
                && post_ops_ok()
                && memory_desc_matches_tag(*src_md(), dat_tag())
                && memory_desc_matches_tag(*dst_md(), dat_tag())
                && memory_desc_matches_tag(*weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md(), weights_md(0), dst_md(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups()
                ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }

        bool post_ops_ok() const {
            auto const &po = attr()->post_ops_;
            auto is_eltwise = [&](int idx)
            { return po.entry_[idx].is_eltwise(); };
            auto is_sum = [&](int idx) { return po.entry_[idx].is_sum(); };

            switch (po.len_) {
            case 0: return true; // no post_ops
            case 1: return is_eltwise(0) || is_sum(0); // sum OR eltwise
            case 2: return is_sum(0) && is_eltwise(1); // sum -> eltwise
            default: return false;
            }
            return false;
        }
    };

    gemm_convolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd, true)
        , eltwise_(nullptr)
    {
        const auto &post_ops = pd()->attr()->post_ops_;
        const data_t one = 1.0, zero = 0.0;
        beta_ = post_ops.find(primitive_kind::sum) >= 0 ? one : zero;

        const int entry_idx = post_ops.find(primitive_kind::eltwise);
        if (entry_idx != -1) eltwise_ = new ref_eltwise_scalar_fwd_t(
                post_ops.entry_[entry_idx].eltwise);
    }

    ~gemm_convolution_fwd_t() { delete eltwise_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    data_t beta_;

    ref_eltwise_scalar_fwd_t* eltwise_;
};

struct gemm_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc, const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_data_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::undef, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats_common(dat_tag(), wei_tag(), dat_tag())
                && memory_desc_matches_tag(*diff_src_md(), dat_tag())
                && memory_desc_matches_tag(*diff_dst_md(), dat_tag())
                && memory_desc_matches_tag(*weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), diff_src_md(), weights_md(0), diff_dst_md(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups()
                ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }
    };

    gemm_convolution_bwd_data_t(const pd_t *apd)
        : cpu_primitive_t(apd, true) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

struct gemm_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(GEMM_IMPL_STR, gemm_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats_common(dat_tag(), wei_tag(), dat_tag())
                && memory_desc_matches_tag(*src_md(), dat_tag())
                && memory_desc_matches_tag(*diff_dst_md(), dat_tag())
                && memory_desc_matches_tag(*diff_weights_md(), wei_tag());
            if (!ok) return status::unimplemented;

            auto scratchpad = scratchpad_registry().registrar();
            return jit_gemm_convolution_utils::init_conf(jcp_, scratchpad,
                    *desc(), src_md(), diff_weights_md(0), diff_dst_md(),
                    mkldnn_get_max_threads());
        }

        jit_gemm_conv_conf_t jcp_;

    protected:
        format_tag_t dat_tag() const {
            using namespace format_tag;
            return utils::pick(ndims() - 3, ncw, nchw, ncdhw);
        }

        format_tag_t wei_tag() const {
            using namespace format_tag;
            return with_groups()
                ? utils::pick(ndims() - 3, goiw, goihw, goidhw)
                : utils::pick(ndims() - 3, oiw, oihw, oidhw);
        }
    };

    gemm_convolution_bwd_weights_t(const pd_t *apd)
        : cpu_primitive_t(apd, true) {}

    typedef typename prec_traits<data_type::f32>::type data_t;

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
