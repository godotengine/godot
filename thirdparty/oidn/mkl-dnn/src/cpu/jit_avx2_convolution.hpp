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

#ifndef CPU_JIT_AVX2_CONVOLUTION_HPP
#define CPU_JIT_AVX2_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_reducer.hpp"

#include "jit_avx2_conv_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_convolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_fwd_t);

        status_t init() {
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx2_conv_fwd_kernel_f32::init_conf(jcp_,
                    *desc(), src_md(), weights_md(), dst_md(), *attr());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_fwd_kernel_f32::init_scratchpad(scratchpad, jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            const bool flat = IC() < 8;
            auto src_tag = flat
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto dst_tag =
                utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                        gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                        OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }
    };

    jit_avx2_convolution_fwd_t(const pd_t *apd): cpu_primitive_t(apd)
    { kernel_ = new jit_avx2_conv_fwd_kernel_f32(pd()->jcp_, *pd()->attr()); }
    ~jit_avx2_convolution_fwd_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

private:
    void execute_forward(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx2_conv_fwd_kernel_f32 *kernel_;
};

struct jit_avx2_convolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_()
        {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_bwd_data_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::undef, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx2_conv_bwd_data_kernel_f32::init_conf(
                    jcp_, *desc(), *diff_src_md(), *weights_md(),
                    *diff_dst_md());
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_bwd_data_kernel_f32::init_scratchpad(scratchpad,
                    jcp_);

            return status::success;
        }

        jit_conv_conf_t jcp_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                ? utils::pick(ndims() - 3, gOIw8o8i, gOIhw8o8i, gOIdhw8o8i)
                : utils::pick(ndims() - 3, OIw8o8i, OIhw8o8i, OIdhw8o8i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    jit_avx2_convolution_bwd_data_t(const pd_t *apd): cpu_primitive_t(apd)
    { kernel_ = new jit_avx2_conv_bwd_data_kernel_f32(pd()->jcp_); }
    ~jit_avx2_convolution_bwd_data_t() { delete kernel_; }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx2_conv_bwd_data_kernel_f32 *kernel_;
};

struct jit_avx2_convolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public  cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit:", avx2, ""),
                jit_avx2_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            status_t status = jit_avx2_conv_bwd_weights_kernel_f32::init_conf(
                    jcp_, *desc(), *src_md(), *diff_weights_md(),
                    *diff_dst_md());
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx2_conv_bwd_weights_kernel_f32::init_scratchpad(scratchpad,
                    jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            auto reducer_wei_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_wei);
            reducer_wei_conf_.init_scratchpad(reducer_wei_scratchpad);

            return status::success;
        }

        jit_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_wei_conf_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;
            const bool flat = IC() == 3;

            auto src_tag = flat
                ? utils::pick(ndims() - 3, ncw, nchw, ncdhw)
                : utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto dst_tag =
                utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c);
            auto wei_tag = with_groups()
                ? utils::pick(2 * ndims() - 6 + flat, gOIw8i8o, gOwi8o,
                        gOIhw8i8o, gOhwi8o, gOIdhw8i8o, gOdhwi8o)
                : utils::pick(2 * ndims() - 6 + flat, OIw8i8o, Owi8o,
                        OIhw8i8o, Ohwi8o, OIdhw8i8o, Odhwi8o);

            return set_default_formats_common(src_tag, wei_tag, dst_tag);
        }

    private:
        void init_balancers() {
            const int max_threads = mkldnn_get_max_threads();
            const size_t max_buffer_size = 1<<21; /* just a heuristic */

            if(with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(max_threads,
                            jcp_.oc_block, jcp_.ngroups * jcp_.nb_oc, jcp_.mb,
                            max_buffer_size));
            }

            reducer_wei_conf_.init(reduce_balancer_t(max_threads,
                        jcp_.kd * jcp_.kh * jcp_.kw
                        * jcp_.ic_block * jcp_.oc_block,
                        jcp_.ngroups * jcp_.nb_ic * jcp_.nb_oc,
                        jcp_.mb * jcp_.od, max_buffer_size));
        }
    };

    jit_avx2_convolution_bwd_weights_t(const pd_t *apd)
        : cpu_primitive_t(apd)
        , kernel_(nullptr)
        , reducer_weights_(nullptr)
        , reducer_bias_(nullptr)
    {
        kernel_ = new jit_avx2_conv_bwd_weights_kernel_f32(pd()->jcp_);
        reducer_bias_ =
            new cpu_reducer_t<data_type::f32>(pd()->reducer_bia_conf_);
        reducer_weights_ =
            new cpu_reducer_t<data_type::f32>(pd()->reducer_wei_conf_);
    }

    ~jit_avx2_convolution_bwd_weights_t() {
        delete kernel_;
        delete reducer_weights_;
        delete reducer_bias_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx2_conv_bwd_weights_kernel_f32 *kernel_;
    cpu_reducer_t<data_type::f32> *reducer_weights_, *reducer_bias_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
