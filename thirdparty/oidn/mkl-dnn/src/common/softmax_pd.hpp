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

#ifndef SOFTMAX_PD_HPP
#define SOFTMAX_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"

namespace mkldnn {
namespace impl {

struct softmax_fwd_pd_t;

struct softmax_pd_t: public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::softmax;

    softmax_pd_t(engine_t *engine,
            const softmax_desc_t *adesc,
            const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , data_md_(desc_.data_desc)
    {}

    const softmax_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::softmax_d:
            *(const softmax_desc_t**)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    /* common softmax aux functions */

    dim_t MB() const { return data_desc().dims[0]; }
    dim_t C() const { return data_desc().dims[1]; }
    dim_t D() const { return ndims() >= 5 ? data_desc().dims[ndims() - 3] : 1; }
    dim_t H() const { return ndims() >= 4 ? data_desc().dims[ndims() - 2] : 1; }
    dim_t W() const { return ndims() >= 3 ? data_desc().dims[ndims() - 1] : 1; }

    int ndims() const { return data_desc().ndims; }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

protected:
    softmax_desc_t desc_;
    const softmax_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t data_md_;

private:
    const memory_desc_t &data_desc() const { return desc_.data_desc; }
};

struct softmax_fwd_pd_t: public softmax_pd_t {
    typedef softmax_fwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    softmax_fwd_pd_t(engine_t *engine,
            const softmax_desc_t *adesc,
            const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(engine, adesc, attr, hint_fwd_pd)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (arg == MKLDNN_ARG_SRC)
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_WORKSPACE && (workspace_md() != nullptr))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &data_md_ : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &data_md_ : nullptr; }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override
    { return 1 + (workspace_md() != nullptr); }
};

struct softmax_bwd_pd_t: public softmax_pd_t {
    typedef softmax_bwd_pd_t base_class;
    typedef softmax_fwd_pd_t hint_class;

    softmax_bwd_pd_t(engine_t *engine,
            const softmax_desc_t *adesc,
            const primitive_attr_t *attr,
            const softmax_fwd_pd_t *hint_fwd_pd)
        : softmax_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_data_md_(desc_.diff_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_DST, MKLDNN_ARG_DIFF_DST))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DIFF_SRC)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_WORKSPACE && (workspace_md() != nullptr))
            return arg_usage_t::input;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &data_md_ : nullptr; }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override
    { return index == 0 ? &diff_data_md_ : nullptr; }
    virtual const memory_desc_t *diff_src_md(int index = 0) const override
    { return index == 0 ? &diff_data_md_ : nullptr; }

    virtual int n_inputs() const override
    { return 2 + (workspace_md() != nullptr); }
    virtual int n_outputs() const override { return 1; }

protected:
    memory_desc_t diff_data_md_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
