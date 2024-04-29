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

#ifndef CPU_REF_SOFTMAX_HPP
#define CPU_REF_SOFTMAX_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_softmax_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t data_type>
struct ref_softmax_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_softmax_fwd_pd_t {
        using cpu_softmax_fwd_pd_t::cpu_softmax_fwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && src_md()->data_type == data_type
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            init_scratchpad();

            return status::success;
        }

    private:
        void init_scratchpad() {
            const int inner_size = utils::array_product(
                    desc()->data_desc.dims + desc()->softmax_axis + 1,
                    desc()->data_desc.ndims - desc()->softmax_axis - 1);

            if (inner_size > 1) {
                auto scratchpad = scratchpad_registry().registrar();
                scratchpad.book(memory_tracking::names::key_softmax_reduction,
                        sizeof(data_t) * 2 * inner_size);
            }
        }
    };

    ref_softmax_fwd_t(const pd_t *apd): cpu_primitive_t(apd)
    {
        auto ndims = pd()->desc()->data_desc.ndims;
        auto dims = pd()->desc()->data_desc.dims;
        auto axis = pd()->desc()->softmax_axis;

        outer_size_ = utils::array_product(dims, axis);
        channels_ = dims[axis];
        inner_size_ = utils::array_product(dims + axis + 1, ndims - axis - 1);

        const memory_desc_wrapper data_d(pd()->src_md());

        bool no_axis_blocking = true;
        for (int iblk = 0; iblk < data_d.blocking_desc().inner_nblks; ++iblk)
            if (data_d.blocking_desc().inner_idxs[iblk] == axis)
                no_axis_blocking = false;

        use_dense_ = inner_size_ == 1 && data_d.is_dense()
            && no_axis_blocking
            && data_d.blocking_desc().strides[axis] == 1;
    }

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            execute_forward_dense(ctx);
        else
            execute_forward_generic(ctx);
        return status::success;
    }

private:
    void execute_forward_dense(const exec_ctx_t &ctx) const;
    void execute_forward_generic(const exec_ctx_t &ctx) const;

    void _max(int n, const data_t *x, data_t *max_data) const;
    void _sub(int n, data_t alpha, const data_t *x, data_t *y) const;
    void _exp(int n, const data_t *a, data_t *r) const;
    void _sum(int n, const data_t *x, data_t *sum_data) const;
    void _scal(int n, data_t alpha, data_t *x) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    bool use_dense_;
    int outer_size_, channels_, inner_size_;
};

template <impl::data_type_t data_type>
struct ref_softmax_bwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_softmax_bwd_pd_t {
        using cpu_softmax_bwd_pd_t::cpu_softmax_bwd_pd_t;

        DECLARE_COMMON_PD_T("ref:any", ref_softmax_bwd_t);

        status_t init() {
            bool ok = true
                && !is_fwd()
                && utils::everyone_is(data_type,
                        dst_md()->data_type,
                        diff_src_md()->data_type)
                && attr()->has_default_values();
            if (!ok) return status::unimplemented;

            return status::success;
        }
    };

    ref_softmax_bwd_t(const pd_t *apd): cpu_primitive_t(apd) {
        auto dims = pd()->desc()->diff_desc.dims;
        auto axis = pd()->desc()->softmax_axis;
        auto ndims = pd()->desc()->diff_desc.ndims;

        outer_size_ = utils::array_product(dims, axis);
        channels_ = dims[axis];
        inner_size_ = utils::array_product(dims + axis + 1, ndims - axis - 1);

        const memory_desc_wrapper data_d(pd()->dst_md());
        const memory_desc_wrapper diff_d(pd()->diff_dst_md());

        bool no_axis_blocking = true;
        for (int iblk = 0; iblk < diff_d.blocking_desc().inner_nblks; ++iblk)
            if (diff_d.blocking_desc().inner_idxs[iblk] == axis)
                no_axis_blocking = false;

        use_dense_ = true
            && inner_size_ == 1
            && diff_d == data_d
            && diff_d.is_dense()
            && no_axis_blocking
            && diff_d.blocking_desc().strides[axis] == 1;
    }

    typedef typename prec_traits<data_type>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        if (use_dense_)
            execute_backward_dense(ctx);
        else
            execute_backward_generic(ctx);
        return status::success;
    }

private:
    void execute_backward_dense(const exec_ctx_t &ctx) const;
    void execute_backward_generic(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    bool use_dense_;
    int outer_size_, channels_, inner_size_;
};


}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
