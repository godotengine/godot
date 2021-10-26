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

#ifndef CONCAT_PD_HPP
#define CONCAT_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "primitive_desc.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {

struct concat_pd_t: public primitive_desc_t {
    concat_pd_t(engine_t *engine, const primitive_attr_t *attr,
            const memory_desc_t *dst_md, int n, int concat_dim,
            const memory_desc_t *src_mds)
        : primitive_desc_t(engine, attr, primitive_kind::concat)
        , n_(n), concat_dim_(concat_dim), dst_md_(*dst_md)
    {
        src_mds_.reserve(n_);
        for (int i = 0; i < n_; ++i) src_mds_.push_back(src_mds[i]);
    }

    concat_pd_t(const concat_pd_t &rhs) = default;

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

    int concat_dim() const { return concat_dim_; }

    const memory_desc_t *src_image_md(int index = 0) const
    { return index < n_inputs() ? &src_image_mds_[index] : nullptr; }

protected:
    int n_, concat_dim_;
    memory_desc_t dst_md_;
    nstl::vector<memory_desc_t> src_mds_;

    /* contains images of srcs in the dst memory (if possible)
     * Lives here to simplify some implementations. An implementation might
     * use this auxiliary array iff init() returned success */
    nstl::vector<memory_desc_t> src_image_mds_;

protected:
    /* inits src_image_mds_ and dst_md_ in simple cases. The call may fail */
    status_t init() {
        bool ok = true
            && set_default_params() == status::success
            && attr()->has_default_values();
        if (!ok) return status::unimplemented;

        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper i_d(&src_mds_[i]);
            if (!i_d.is_blocking_desc() || i_d.is_additional_buffer())
                return status::unimplemented;
        }

        const int ndims = dst_md_.ndims;
        int current_concat_dim_offset = 0;
        for (int i = 0; i < n_; ++i) {
            const int dim = src_mds_[i].dims[concat_dim_];
            dims_t dims, offsets = {};
            utils::array_copy(dims, dst_md_.dims, ndims);
            dims[concat_dim_] = dim;
            offsets[concat_dim_] = current_concat_dim_offset;

            memory_desc_t src_img_d;
            status_t status = mkldnn_memory_desc_init_submemory(&src_img_d,
                    &dst_md_, dims, offsets);
            if (status != status::success) return status;
            src_image_mds_.push_back(src_img_d);
            current_concat_dim_offset += dim;
        }

        return status::success;
    }

    status_t set_default_params() {
        if (dst_md_.format_kind != format_kind::any)
            return status::success;

        const int ndims = dst_md_.ndims;

        /* The stupidest ever heuristics (but not the same as we had before):
         *  - Pick the first non-plain format;
         *  - If all formats are plain or it is not possible to create a
         *    blocked format for the output, pick the format of the plain input
         *  - If this fails as well, use plain layout (abcd...)
         */
        status_t status = status::unimplemented;
        for (int i = 0; i < n_; ++i) {
            const memory_desc_wrapper src_d(src_mds_[i]);
            if (src_d.is_blocking_desc() && !src_d.is_plain()) {
                status = memory_desc_init_by_blocking_desc(dst_md_,
                        src_d.blocking_desc());
                if (status == status::success) break;
            }
        }

        if (status == status::success) {
            /* check if we can create a sub-memory for the dst */
            bool desired_format_ok = true;
            int current_concat_dim_offset = 0;
            for (int i = 0; i < n_; ++i) {
                const int dim = src_mds_[i].dims[concat_dim_];
                dims_t dims, offsets = {};
                utils::array_copy(dims, dst_md_.dims, ndims);
                dims[concat_dim_] = dim;
                offsets[concat_dim_] = current_concat_dim_offset;

                memory_desc_t src_img_d;
                status_t status = mkldnn_memory_desc_init_submemory(&src_img_d,
                        &dst_md_, dims, offsets);
                if (status != status::success) {
                    desired_format_ok = false;
                    break;
                }
                current_concat_dim_offset += dim;
            }

            if (!desired_format_ok)
                status = status::unimplemented;
        }

        /* if no success so far, try using the format of the first plain input */
        if (status != status::success) {
            for (int i = 0; i < n_; ++i) {
                const memory_desc_wrapper src_d(src_mds_[i]);
                if (src_d.is_blocking_desc() && src_d.is_plain()) {
                    status = memory_desc_init_by_blocking_desc(dst_md_,
                            memory_desc_wrapper(src_mds_[0]).blocking_desc());
                    if (status == status::success) return status;
                }
            }
        }

        /* the last line of defense: use plain abcd... format */
        if (status != status::success)
            status = memory_desc_init_by_strides(dst_md_, nullptr);

        return status;
    }
};

#define DECLARE_CONCAT_PD_t(impl_name, ...) \
    static status_t create(concat_pd_t **concat_pd, \
            engine_t *engine, const primitive_attr_t *attr, \
            const memory_desc_t *dst_md, int n, int concat_dim, \
            const memory_desc_t *src_mds) { \
        using namespace status; \
        auto _pd = new pd_t(engine, attr, dst_md, n, concat_dim, src_mds); \
        if (_pd == nullptr) return out_of_memory; \
        if (_pd->init() != success) { delete _pd; return unimplemented; } \
        return safe_ptr_assign<concat_pd_t>(*concat_pd, _pd); \
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

#define DECLARE_CONCAT_PD_T(impl_name, ...) \
    DECLARE_CONCAT_PD_t(impl_name, __VA_ARGS__)

}
}

#endif
