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

#ifndef REORDER_PD_HPP
#define REORDER_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "primitive_attr.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct reorder_pd_t: public primitive_desc_t {
    reorder_pd_t(engine_t *engine, const primitive_attr_t *attr,
            engine_t *src_engine, const memory_desc_t *src_md,
            engine_t *dst_engine, const memory_desc_t *dst_md)
        : primitive_desc_t(engine, attr, primitive_kind::reorder)
        , src_engine_(src_engine)
        , dst_engine_(dst_engine)
        , scratchpad_engine_(nullptr)
        , src_md_(*src_md)
        , dst_md_(*dst_md)
    {}

    virtual const op_desc_t *op_desc() const override { return nullptr; }
    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (arg == MKLDNN_ARG_FROM)
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_TO)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index == 0 ? &src_md_ : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &dst_md_ : nullptr; }

    virtual int n_inputs() const override { return 1; }
    virtual int n_outputs() const override { return 1; }

    float alpha() const { return attr()->output_scales_.scales_[0]; }
    float beta() const {
        const int sum_idx = attr()->post_ops_.find(primitive_kind::sum);
        return sum_idx == -1 ? 0 : attr()->post_ops_.entry_[sum_idx].sum.scale;
    }
    virtual mkldnn::impl::engine_t *scratchpad_engine() const override
    { return scratchpad_engine_; }

protected:
    engine_t *src_engine_;
    engine_t *dst_engine_;
    engine_t *scratchpad_engine_;

    memory_desc_t src_md_;
    memory_desc_t dst_md_;
};

}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
