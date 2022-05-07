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

#ifndef RNN_PD_HPP
#define RNN_PD_HPP

#include "mkldnn.h"

#include "c_types_map.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"

namespace mkldnn {
namespace impl {

struct rnn_fwd_pd_t;

struct rnn_pd_t : public primitive_desc_t {
    static constexpr auto base_pkind = primitive_kind::rnn;

    rnn_pd_t(engine_t *engine,
            const rnn_desc_t *adesc,
            const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : primitive_desc_t(engine, attr, base_pkind)
        , desc_(*adesc)
        , hint_fwd_pd_(hint_fwd_pd)
        , src_layer_md_(desc_.src_layer_desc)
        , src_iter_md_(desc_.src_iter_desc)
        , weights_layer_md_(desc_.weights_layer_desc)
        , weights_iter_md_(desc_.weights_iter_desc)
        , bias_md_(desc_.bias_desc)
        , dst_layer_md_(desc_.dst_layer_desc)
        , dst_iter_md_(desc_.dst_iter_desc)
        , ws_md_()
    {}

    const rnn_desc_t *desc() const { return &desc_; }
    virtual const op_desc_t *op_desc() const override
    { return reinterpret_cast<const op_desc_t *>(this->desc()); }
    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual status_t query(query_t what, int idx, void *result) const override {
        switch (what) {
        case query::rnn_d: *(const rnn_desc_t **)result = desc(); break;
        default: return primitive_desc_t::query(what, idx, result);
        }
        return status::success;
    }

    virtual const memory_desc_t *src_md(int index = 0) const override {
        if (index == 0) return &src_layer_md_;
        if (index == 1 && with_src_iter()) return &src_iter_md_;
        return nullptr;
    }
    virtual const memory_desc_t *weights_md(int index = 0) const override {
        if (index == 0) return &weights_layer_md_;
        if (index == 1) return &weights_iter_md_;
        if (index == 2 && with_bias()) return &bias_md_;
        return nullptr;
    }
    virtual const memory_desc_t *dst_md(int index = 0) const override {
        if (index == 0) return &dst_layer_md_;
        if (index == 1 && with_dst_iter()) return &dst_iter_md_;
        return nullptr;
    }
    virtual const memory_desc_t *workspace_md(int index = 0) const override
    { return index == 0 && !types::is_zero_md(&ws_md_) ? &ws_md_ : nullptr; }

    /* common pooling aux functions */

    bool is_training() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::backward);
    }

    bool is_fwd() const {
        return utils::one_of(desc_.prop_kind, prop_kind::forward_training,
                prop_kind::forward_inference);
    }

    dim_t T() const { return desc_.src_layer_desc.dims[0]; }
    dim_t MB() const { return desc_.src_layer_desc.dims[1]; }

    dim_t L() const { return desc_.weights_layer_desc.dims[0]; }
    dim_t D() const { return desc_.weights_layer_desc.dims[1]; }

    dim_t SIC() const { return desc_.weights_iter_desc.dims[2]; }

    dim_t SLC() const { return desc_.weights_layer_desc.dims[2]; }
    dim_t G() const { return desc_.weights_layer_desc.dims[3]; }
    dim_t DIC() const { return desc_.weights_layer_desc.dims[4]; }

    dim_t DLC() const { return desc_.dst_layer_desc.dims[2]; }

    bool with_bias() const
    { return !memory_desc_wrapper(desc_.bias_desc).is_zero(); }

    bool with_src_iter() const
    { return !(memory_desc_wrapper(desc_.src_iter_desc).is_zero()); }

    bool with_dst_iter() const
    { return !memory_desc_wrapper(desc_.dst_iter_desc).is_zero(); }

    mkldnn::impl::alg_kind_t cell_kind() const
    { return desc_.cell_desc.cell_kind; }
    mkldnn::impl::alg_kind_t activation_kind() const
    { return desc_.cell_desc.activation_kind; }

    bool is_lbr() const
    { return cell_kind() == mkldnn_gru_linear_before_reset; }

    mkldnn_rnn_direction_t direction() const { return desc_.direction; }

protected:
    rnn_desc_t desc_;
    const rnn_fwd_pd_t *hint_fwd_pd_;

    memory_desc_t src_layer_md_;
    memory_desc_t src_iter_md_;
    memory_desc_t weights_layer_md_;
    memory_desc_t weights_iter_md_;
    memory_desc_t bias_md_;
    memory_desc_t dst_layer_md_;
    memory_desc_t dst_iter_md_;

    memory_desc_t ws_md_;
};

struct rnn_fwd_pd_t: public rnn_pd_t {
    typedef rnn_fwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    rnn_fwd_pd_t(engine_t *engine,
            const rnn_desc_t *adesc,
            const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_pd_t(engine, adesc, attr, hint_fwd_pd)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (arg == MKLDNN_ARG_SRC_LAYER)
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_SRC_ITER && with_src_iter())
            return arg_usage_t::input;

        if (utils::one_of(arg, MKLDNN_ARG_WEIGHTS_LAYER,
                    MKLDNN_ARG_WEIGHTS_ITER))
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_BIAS && with_bias())
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST_LAYER)
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_DST_ITER && with_dst_iter())
            return arg_usage_t::output;

        if (arg == MKLDNN_ARG_WORKSPACE && is_training())
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual int n_inputs() const override
    { return 3 + with_bias() + with_src_iter(); }
    virtual int n_outputs() const override
    { return 1 + with_dst_iter() + is_training(); }
};

struct rnn_bwd_pd_t : public rnn_pd_t {
    typedef rnn_bwd_pd_t base_class;
    typedef rnn_fwd_pd_t hint_class;

    rnn_bwd_pd_t(engine_t *engine,
            const rnn_desc_t *adesc,
            const primitive_attr_t *attr,
            const rnn_fwd_pd_t *hint_fwd_pd)
        : rnn_pd_t(engine, adesc, attr, hint_fwd_pd)
        , diff_src_layer_md_(desc_.diff_src_layer_desc)
        , diff_src_iter_md_(desc_.diff_src_iter_desc)
        , diff_weights_layer_md_(desc_.diff_weights_layer_desc)
        , diff_weights_iter_md_(desc_.diff_weights_iter_desc)
        , diff_bias_md_(desc_.diff_bias_desc)
        , diff_dst_layer_md_(desc_.diff_dst_layer_desc)
        , diff_dst_iter_md_(desc_.diff_dst_iter_desc)
    {}

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (utils::one_of(arg, MKLDNN_ARG_SRC_LAYER, MKLDNN_ARG_DST_LAYER,
                    MKLDNN_ARG_DIFF_DST_LAYER))
            return arg_usage_t::input;

        if (with_src_iter()) {
            if (arg == MKLDNN_ARG_SRC_ITER)
                return arg_usage_t::input;

            if (arg == MKLDNN_ARG_DIFF_SRC_ITER)
                return arg_usage_t::output;
        }

        if (utils::one_of(arg, MKLDNN_ARG_WEIGHTS_LAYER,
                    MKLDNN_ARG_WEIGHTS_ITER))
            return arg_usage_t::input;

        if (with_bias()) {
            if (arg == MKLDNN_ARG_BIAS)
                return arg_usage_t::input;

            if (arg == MKLDNN_ARG_DIFF_BIAS)
                return arg_usage_t::output;
        }

        if (utils::one_of(arg, MKLDNN_ARG_DST_ITER, MKLDNN_ARG_DIFF_DST_ITER)
                && with_dst_iter())
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_WORKSPACE)
            return arg_usage_t::input;

        if (utils::one_of(arg, MKLDNN_ARG_DIFF_SRC_LAYER,
                    MKLDNN_ARG_DIFF_WEIGHTS_LAYER,
                    MKLDNN_ARG_DIFF_WEIGHTS_ITER))
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *diff_src_md(int index = 0) const override {
        if (index == 0) return &diff_src_layer_md_;
        if (index == 1 && with_src_iter()) return &diff_src_iter_md_;
        return nullptr;
    }
    virtual const memory_desc_t *diff_weights_md(
            int index = 0) const override {
        if (index == 0) return &diff_weights_layer_md_;
        if (index == 1) return &diff_weights_iter_md_;
        if (index == 2 && with_bias()) return &diff_bias_md_;
        return nullptr;
    }
    virtual const memory_desc_t *diff_dst_md(int index = 0) const override {
        if (index == 0) return &diff_dst_layer_md_;
        if (index == 1 && with_dst_iter()) return &diff_dst_iter_md_;
        return nullptr;
    }

    virtual int n_inputs() const override
    { return 6 + with_src_iter() + with_bias() + 2 * with_dst_iter(); }
    virtual int n_outputs() const override
    { return 3 + with_src_iter() + with_bias(); }

protected:
    memory_desc_t diff_src_layer_md_;
    memory_desc_t diff_src_iter_md_;
    memory_desc_t diff_weights_layer_md_;
    memory_desc_t diff_weights_iter_md_;
    memory_desc_t diff_bias_md_;
    memory_desc_t diff_dst_layer_md_;
    memory_desc_t diff_dst_iter_md_;
};

}
}

#endif
