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

#ifndef CPU_JIT_AVX512_COMMON_CONVOLUTION_HPP
#define CPU_JIT_AVX512_COMMON_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_barrier.hpp"
#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"
#include "cpu_reducer.hpp"

#include "jit_transpose_src_utils.hpp"
#include "jit_avx512_common_conv_kernel.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
struct jit_avx512_common_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx512_common, ""),
                jit_avx512_common_convolution_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(src_type, wei_type, dst_type, dst_type,
                        data_type::undef)
                && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_common_conv_fwd_kernel::init_conf(
                    jcp_, *desc(), src_md_, weights_md_, dst_md_, bias_md_,
                    *attr(), mkldnn_get_max_threads());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_fwd_kernel::init_scratchpad(scratchpad,
                    jcp_);

            return status;
        }

        jit_conv_conf_t jcp_;
    };

    jit_avx512_common_convolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd)
    {
        kernel_ = new jit_avx512_common_conv_fwd_kernel(pd()->jcp_,
                *pd()->attr());
    }
    ~jit_avx512_common_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_forward_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_forward_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_forward_3d(ctx);
        else
            assert(false);

        if (pd()->wants_zero_pad_dst())
            ctx.memory(MKLDNN_ARG_DST)->zero_pad();

        return status::success;
    }

private:
    void prepare_padded_bias(const dst_data_t *&bias,
            const memory_tracking::grantor_t &scratchpad) const;
    void execute_forward_1d(const exec_ctx_t &ctx) const;
    void execute_forward_2d(const exec_ctx_t &ctx) const;
    void execute_forward_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_conv_fwd_kernel *kernel_;
};

template <impl::data_type_t diff_dst_type,
          impl::data_type_t wei_type = diff_dst_type,
          impl::data_type_t diff_src_type = diff_dst_type>
struct jit_avx512_common_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx512_common, ""),
                jit_avx512_common_convolution_bwd_data_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(diff_src_type, wei_type,
                        data_type::undef, diff_dst_type, data_type::undef)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status =
                jit_avx512_common_conv_bwd_data_kernel_f32::init_conf(jcp_,
                        *desc(), *diff_src_md(), *weights_md(), *diff_dst_md());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_bwd_data_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw16o16i, gOIw16o16i, OIhw16o16i, gOIhw16o16i,
                    OIdhw16o16i, gOIdhw16o16i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    jit_avx512_common_convolution_bwd_data_t(const pd_t *apd)
        : cpu_primitive_t(apd)
    { kernel_ = new jit_avx512_common_conv_bwd_data_kernel_f32(pd()->jcp_); }
    ~jit_avx512_common_convolution_bwd_data_t() { delete kernel_; };

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        if (pd()->ndims() == 3)
            execute_backward_data_1d(ctx);
        else if (pd()->ndims() == 4)
            execute_backward_data_2d(ctx);
        else if (pd()->ndims() == 5)
            execute_backward_data_3d(ctx);
        else
            assert(false);
        return status::success;
    }

private:
    void execute_backward_data_1d(const exec_ctx_t &ctx) const;
    void execute_backward_data_2d(const exec_ctx_t &ctx) const;
    void execute_backward_data_3d(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_conv_bwd_data_kernel_f32 *kernel_;
};

template <impl::data_type_t src_type,
          impl::data_type_t diff_dst_type = src_type,
          impl::data_type_t diff_weights_type = src_type>
struct jit_avx512_common_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public  cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx512_common, ""),
                jit_avx512_common_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(src_type, diff_weights_type,
                        diff_weights_type, diff_dst_type, data_type::undef)
                && !has_zero_dim_memory();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx512_common_conv_bwd_weights_kernel_f32::
                init_conf(jcp_, *desc(), src_md_, diff_weights_md_,
                        diff_bias_md_, diff_dst_md_);
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_conv_bwd_weights_kernel_f32::init_scratchpad(
                    scratchpad, jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            return status;
        }

        jit_conv_conf_t jcp_;
        typename cpu_reducer_t<diff_weights_type>::conf_t reducer_bia_conf_;

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                            jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                            max_buffer_size));
            }
        }
    };

    jit_avx512_common_convolution_bwd_weights_t(const pd_t *apd);
    ~jit_avx512_common_convolution_bwd_weights_t() {
        delete kernel_;
        if (trans_kernel_)
            delete trans_kernel_;
        if (acc_ker_)
            delete acc_ker_;
        delete reducer_bias_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<diff_weights_type>::type diff_weights_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    void prepare_scratchpad_data(const exec_ctx_t &ctx) const;
    struct thread_info_t;
    void compute_diff_weights(const thread_info_t *) const;
    void compute_diff_weights_3d(const thread_info_t *) const;
    void reduce_diff_weights(const thread_info_t *) const;
    void reduce_diff_weights_3d(const thread_info_t *) const;
    void compute_diff_bias(const thread_info_t *) const;
    void compute_diff_bias_3d(const thread_info_t *) const;

    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    int nthr_, nthr_mb_, nthr_g_, nthr_oc_b_, nthr_ic_b_;

    jit_avx512_common_conv_bwd_weights_kernel_f32 *kernel_;
    jit_trans_src_t *trans_kernel_;
    cpu_accumulator_1d_t<diff_weights_type> *acc_ker_;
    cpu_reducer_t<diff_weights_type> *reducer_bias_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
