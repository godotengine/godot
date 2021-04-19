/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef SUM_PD_HPP
#define SUM_PD_HPP

#include <assert.h>
#include "mkldnn.h"

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct sum_pd_t: public primitive_desc_t {
    sum_pd_t(engine_t *engine, const primitive_attr_t *attr,
            const memory_desc_t *dst_md, int n, const float *scales,
            const memory_desc_t *src_mds)
        : primitive_desc_t(engine, attr, primitive_kind::sum)
        , n_(n), dst_md_(*dst_md)
    {
        scales_.reserve(n_);
        for (int i = 0; i < n_; ++i) scales_.push_back(scales[i]);
        src_mds_.reserve(n_);
        for (int i = 0; i < n_; ++i) src_mds_.push_back(src_mds[i]);
    }

    virtual void init_info() override { impl::init_info(this, this->info_); }

    virtual arg_usage_t arg_usage(primitive_arg_index_t arg) const override {
        if (arg >= MKLDNN_ARG_MULTIPLE_SRC
                && arg < MKLDNN_ARG_MULTIPLE_SRC + n_inputs())
            return arg_usage_t::input;

        if (arg == MKLDNN_ARG_DST)
            return arg_usage_t::output;

        return primitive_desc_t::arg_usage(arg);
    }

    virtual const memory_desc_t *src_md(int index = 0) const override
    { return index < n_inputs() ? &src_mds_[index] : nullptr; }
    virtual const memory_desc_t *dst_md(int index = 0) const override
    { return index == 0 ? &dst_md_ : nullptr; }

    virtual int n_inputs() const override { return n_; }
    virtual int n_outputs() const override { return 1; }

    const float *scales() const { return &scales_[0]; }

protected:
    int n_;
    nstl::vector<float> scales_;
    memory_desc_t dst_md_;
    nstl::vector<memory_desc_t> src_mds_;

protected:
    /* inits dst_md_ in simple cases. The call may fail. */
    status_t init() {
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(&src_mds_[i]);
            if (!src_d.is_blocking_desc() || src_d.is_additional_buffer())
                return status::unimplemented;
        }
        bool ok = true
            && set_default_params() == status::success
            && attr()->has_default_values();
        return ok ? status::success : status::unimplemented;
    }

    status_t set_default_params() {
        if (dst_md_.format_kind != format_kind::any)
            return status::success;

        /* The stupidest ever heuristics (but not the same as we had before):
         *  - Pick the first non-plain format;
         *  - If all formats are plain, pick the format of the first input
         */
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(src_mds_[i]);
            if (!src_d.is_plain() && src_d.is_blocking_desc()) {
                return memory_desc_init_by_blocking_desc(dst_md_,
                        src_d.blocking_desc());
            }
        }

        if (src_mds_[0].format_kind != format_kind::blocked)
            return status::unimplemented;

        dst_md_ = src_mds_[0];

        return status::success;
    }
};

#define DECLARE_SUM_PD_t(impl_name, ...) \
    static status_t create(sum_pd_t **sum_pd, \
            engine_t *engine, const primitive_attr_t *attr, \
            const memory_desc_t *dst_md, int n, const float *scales, \
            const memory_desc_t *src_mds) { \
        using namespace status; \
        auto _pd = new pd_t(engine, attr, dst_md, n, scales, src_mds); \
        if (_pd == nullptr) return out_of_memory; \
        if (_pd->init() != success) { delete _pd; return unimplemented; } \
        return safe_ptr_assign<sum_pd_t>(*sum_pd, _pd); \
    } \
    virtual status_t create_primitive(primitive_t **p) const override { \
        double ms = get_msec(); \
        auto ret = safe_ptr_assign<primitive_t>(*p, new (__VA_ARGS__)(this)); \
        ms = get_msec() - ms; \
        if (mkldnn_verbose()->level >= 2) { \
            printf("mkldnn_verbose,create,%s,%g\n", this->info(), ms); \
            fflush(0); \
        } \
        return ret; \
    } \
    virtual pd_t *clone() const override { return new pd_t(*this); } \
    virtual const char *name() const override { return impl_name; } \

#define DECLARE_SUM_PD_T(impl_name, ...) \
    DECLARE_SUM_PD_t(impl_name, __VA_ARGS__)

}
}

#endif
