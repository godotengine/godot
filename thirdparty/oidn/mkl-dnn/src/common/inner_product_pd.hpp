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

#ifndef INNER_PRODUCT_PD_HPP
#define INNER_PRODUCT_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

memory_desc_t *ip_prop_invariant_src_d(inner_product_desc_t *desc);
memory_desc_t *ip_prop_invariant_wei_d(inner_product_desc_t *desc);
memory_desc_t *ip_prop_invariant_bia_d(inner_product_desc_t *desc);
memory_desc_t *ip_prop_invariant_dst_d(inner_product_desc_t *desc);
const memory_desc_t *ip_prop_invariant_src_d(const inner_product_desc_t *desc);
const memory_desc_t *ip_prop_invariant_wei_d(const inner_product_desc_t *desc);
const memory_desc_t *ip_prop_invariant_bia_d(const inner_product_desc_t *desc);
const memory_desc_t *ip_prop_invariant_dst_d(const inner_product_desc_t *desc);

struct inner_product_fwd_pd_t;

struct inner_product_pd_t: public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::inner_product;

    inner_product_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
    {}

    const inner_product_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::inner_product_d:
            *(const inner_product_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common inner_product aux functions */

    dim_t MB() const { return ip_prop_invariant_src_d(&desc_)->dims[0]; }
    dim_t IC() const { return ip_prop_invariant_src_d(&desc_)->dims[1]; }
    dim_t OC() const { return ip_prop_invariant_dst_d(&desc_)->dims[1]; }

    dim_t ID() const {
        return ndims() >= 5
            ? ip_prop_invariant_src_d(&desc_)->dims[ndims() - 3] : 1;
    }
    dim_t IH() const {
        return ndims() >= 4
            ? ip_prop_invariant_src_d(&desc_)->dims[ndims() - 2] : 1;
    }
    dim_t IW() const {
        return ndims() >= 3
            ? ip_prop_invariant_src_d(&desc_)->dims[ndims() - 1] : 1;
    }

    dim_t OD() const {
        return ndims() >= 5
            ? ip_prop_invariant_dst_d(&desc_)->dims[ndims() - 3] : 1;
    }
    dim_t OH() const {
        return ndims() >= 4
            ? ip_prop_invariant_dst_d(&desc_)->dims[ndims() - 2] : 1;
    }
    dim_t OW() const {
        return ndims() >= 3
            ? ip_prop_invariant_dst_d(&desc_)->dims[ndims() - 1] : 1;
    }

    dim_t KD() const {
        return ndims() >= 5
            ? ip_prop_invariant_wei_d(&desc_)->dims[ndims() - 3] : 1;
    }
    dim_t KH() const {
        return ndims() >= 4
            ? ip_prop_invariant_wei_d(&desc_)->dims[ndims() - 2] : 1;
    }
    dim_t KW() const {
        return ndims() >= 3
            ? ip_prop_invariant_wei_d(&desc_)->dims[ndims() - 1] : 1;
    }

    dim_t IC_total() const {
        return utils::array_product(&ip_prop_invariant_src_d(&desc_)->dims[1],
                ndims() - 1);
    }

    dim_t IC_total_padded() const {
        auto src_d = desc()->prop_kind == prop_kind::backward_data
            ? memory_desc_wrapper(diff_src_md())
            : memory_desc_wrapper(src_md());
        assert(src_d.is_blocking_desc());
        if (!src_d.is_blocking_desc()) return -1;
        return utils::array_product(src_d.padded_dims() + 1, ndims() - 1);
    }

    int ndims() const { return ip_prop_invariant_src_d(&desc_)->ndims; }

    bool with_bias() const
    { return !memory_desc_wrapper(*ip_prop_invariant_bia_d(&desc_)).is_zero(); }

    bool has_zero_dim_memory() const {
        const auto s_d = memory_desc_wrapper(*ip_prop_invariant_src_d(&desc_));
        const auto d_d = memory_desc_wrapper(*ip_prop_invariant_dst_d(&desc_));
        return s_d.has_zero_dim() || d_d.has_zero_dim();
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    inner_product_desc_t desc_;
    const inner_product_fwd_pd_t *hint_fwd_pd_;

    status_t template_set_default_params(memory_desc_t &src_md,
            memory_desc_t &weights_md, memory_desc_t &dst_md,
            memory_desc_t *bias_md) {
        using namespace format_tag;
        if (src_md.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(src_md,
                        utils::pick(ndims() - 2, nc, ncw, nchw, ncdhw)));
        }
        if (dst_md.format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(dst_md, nc));
        if (weights_md.format_kind == format_kind::any) {
            CHECK(memory_desc_init_by_tag(weights_md,
                        utils::pick(ndims() - 2, oi, oiw, oihw, oidhw)));
        }
        if (bias_md && bias_md->format_kind == format_kind::any)
            CHECK(memory_desc_init_by_tag(*bias_md, x));
        return status::success;
    }
};

struct inner_product_fwd_pd_t: public inner_product_pd_t {
    typedef inner_product_fwd_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_fwd_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , weights_md_(desc_.weights_desc)
        , bias_md_(desc_.bias_desc)
        , dst_md_(desc_.dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_SRC, MKLDNN_ARG_WEIGHTS))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_BIAS && with_bias())
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &dst_md_ : nullptr; }
    virtual const memory_desc_t *weights_md(int index = 0) const override {
        if (index == 0) return &weights_md_;
        if (index == 1 && with_bias()) return &bias_md_;
        return nullptr;
    }

    virtual int n_inputs() const override { return 2 + with_bias(); }
    virtual int n_outputs() const override { return 1; }

protected:
    memory_desc_t src_md_;
    memory_desc_t weights_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_md_;

    status_t set_default_params() {
        return template_set_default_params(src_md_, weights_md_, dst_md_,
                &bias_md_);
    }
};

struct inner_product_bwd_data_pd_t: public inner_product_pd_t {
    typedef inner_product_bwd_data_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_bwd_data_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_md_(desc_.diff_src_desc)
        , weights_md_(desc_.weights_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_WEIGHTS, MKLDNN_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DIFF_SRC)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *diff_src_md(int index = 0) const override
    { return index == 0 ? &diff_src_md_ : nullptr; }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override
    { return index == 0 ? &diff_dst_md_ : nullptr; }
    virtual const memory_desc_t *weights_md(int index = 0) const override
    { return index == 0 ? &weights_md_ : nullptr; }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_src_md_;
    memory_desc_t weights_md_;
    memory_desc_t diff_dst_md_;

    status_t set_default_params() {
        return template_set_default_params(diff_src_md_, weights_md_,
                diff_dst_md_, nullptr);
    }
};

struct inner_product_bwd_weights_pd_t: public inner_product_pd_t {
    typedef inner_product_bwd_weights_pd_t base_class;
    typedef inner_product_fwd_pd_t hint_class;

    inner_product_bwd_weights_pd_t(engine_t *engine,
            const inner_product_desc_t *adesc,
            const primitive_attr_t *attr,
            const inner_product_fwd_pd_t *hint_fwd_pd)
        : inner_product_pd_t(engine, adesc, attr, hint_fwd_pd)
        , src_md_(desc_.src_desc)
        , diff_weights_md_(desc_.diff_weights_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_md_(desc_.diff_dst_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_SRC, MKLDNN_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DIFF_WEIGHTS)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_DIFF_BIAS && with_bias())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override
    { return index == 0 ? &diff_dst_md_ : nullptr; }
    virtual const memory_desc_t *diff_weights_md(int index = 0) const override {
        if (index == 0) return &diff_weights_md_;
        if (index == 1 && with_bias()) return &diff_bias_md_;
        return nullptr;
    }

    virtual int n_inputs() const override { return 2; }
    virtual int n_outputs() const override { return 1 + with_bias(); }

protected:
    memory_desc_t src_md_;
    memory_desc_t diff_weights_md_;
    memory_desc_t diff_bias_md_;
    memory_desc_t diff_dst_md_;

    status_t set_default_params() {
        return template_set_default_params(src_md_, diff_weights_md_,
                diff_dst_md_, &diff_bias_md_);
    }
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
