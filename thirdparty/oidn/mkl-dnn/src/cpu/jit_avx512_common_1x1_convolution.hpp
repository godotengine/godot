/*******************************************************************************
* Copyright 2017-2018 Intel Corporation
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

#ifndef CPU_JIT_AVX512_COMMON_1x1_CONVOLUTION_HPP
#define CPU_JIT_AVX512_COMMON_1x1_CONVOLUTION_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_primitive.hpp"
#include "cpu_reducer.hpp"

#include "jit_avx512_common_1x1_conv_kernel.hpp"
#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_transpose_src_utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <impl::data_type_t src_type,
         impl::data_type_t wei_type = src_type,
         impl::data_type_t dst_type = src_type>
struct jit_avx512_common_1x1_convolution_fwd_t : public cpu_primitive_t {
    struct pd_t: public cpu_convolution_fwd_pd_t {
        pd_t(engine_t *engine, const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const typename pd_t::base_class *hint_fwd_pd)
            : cpu_convolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_fwd_t);

        status_t init() {
            using namespace utils;
            bool ok = true
                && is_fwd()
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(src_type, wei_type, dst_type, dst_type,
                        data_type::undef)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, dst_md());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *weights_md(), *dst_md(), *attr(),
                    mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw16i16o, gOIw16i16o, OIhw16i16o, gOIhw16i16o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_fwd_t(const pd_t *apd)
        : cpu_primitive_t(apd)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ =
            new jit_avx512_common_1x1_conv_kernel(pd()->jcp_, *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_common_1x1_convolution_fwd_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<src_type>::type src_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<dst_type>::type dst_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_forward(ctx);
        return status::success;
    }

  private:
    void execute_forward(const exec_ctx_t &ctx) const;
    void execute_forward_thr(const int ithr, const int nthr,
            const src_data_t *src, const wei_data_t *weights,
            const dst_data_t *bias, dst_data_t *dst,
            const memory_tracking::grantor_t &scratchpad) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

using jit_avx512_common_1x1_convolution_fwd_f32_t
        = jit_avx512_common_1x1_convolution_fwd_t<data_type::f32>;

template <impl::data_type_t diff_dst_type,
          impl::data_type_t wei_type = diff_dst_type,
          impl::data_type_t diff_src_type = diff_dst_type>
struct jit_avx512_common_1x1_convolution_bwd_data_t : public cpu_primitive_t {
    struct pd_t : public cpu_convolution_bwd_data_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_bwd_data_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(diff_src_type, wei_type, data_type::undef,
                        diff_dst_type, data_type::undef)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *diff_src_d = diff_src_md();
            rtus_prepare(this, conv_d, diff_src_d, diff_dst_md());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *diff_src_d, *weights_md(), *diff_dst_md(),
                    *attr(), mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    IOw16o16i, gIOw16o16i, IOhw16o16i, gIOhw16o16i);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_bwd_data_t(const pd_t *apd)
        : cpu_primitive_t(apd)
        , kernel_(nullptr), rtus_driver_(nullptr)
    {
        kernel_ = new jit_avx512_common_1x1_conv_kernel(pd()->jcp_,
                    *pd()->attr());
        init_rtus_driver<avx512_common>(this);
    }

    ~jit_avx512_common_1x1_convolution_bwd_data_t() {
        delete kernel_;
        delete rtus_driver_;
    }

    typedef typename prec_traits<diff_dst_type>::type diff_dst_data_t;
    typedef typename prec_traits<wei_type>::type wei_data_t;
    typedef typename prec_traits<diff_src_type>::type diff_src_data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_data(ctx);
        return status::success;
    }

  private:
    void execute_backward_data(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_1x1_conv_kernel *kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

using jit_avx512_common_1x1_convolution_bwd_data_f32_t
        = jit_avx512_common_1x1_convolution_bwd_data_t<data_type::f32>;

struct jit_avx512_common_1x1_convolution_bwd_weights_t : public cpu_primitive_t
{
    struct pd_t : public cpu_convolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const convolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const convolution_fwd_pd_t *hint_fwd_pd)
            : cpu_convolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , jcp_(), rtus_() {}

        DECLARE_COMMON_PD_T(
                JIT_IMPL_NAME_HELPER("jit_1x1:", avx512_common, ""),
                jit_avx512_common_1x1_convolution_bwd_weights_t);

        status_t init() {
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && set_default_alg_kind(alg_kind::convolution_direct)
                && expect_data_types(data_type::f32, data_type::f32,
                        data_type::f32, data_type::f32, data_type::f32)
                && !has_zero_dim_memory()
                && set_default_formats();
            if (!ok) return status::unimplemented;

            const convolution_desc_t *conv_d = desc();
            const memory_desc_t *src_d = src_md();
            rtus_prepare(this, conv_d, src_d, diff_dst_md());

            status_t status = jit_avx512_common_1x1_conv_kernel::init_conf(
                    jcp_, *conv_d, *src_d, *diff_weights_md(), *diff_dst_md(),
                    *attr(), mkldnn_get_max_threads(), rtus_.reduce_src_);
            if (status != status::success) return status;

            init_balancers();

            auto scratchpad = scratchpad_registry().registrar();
            jit_avx512_common_1x1_conv_kernel::init_scratchpad(scratchpad,
                    jcp_);

            auto reducer_bia_scratchpad = memory_tracking::registrar_t(
                    scratchpad, memory_tracking::names::prefix_reducer_bia);
            reducer_bia_conf_.init_scratchpad(reducer_bia_scratchpad);

            rtus_prepare_space_info(this, scratchpad);

            return status::success;
        }

        // TODO (Roma): structs conf header cleanup
        jit_1x1_conv_conf_t jcp_;
        cpu_reducer_t<data_type::f32>::conf_t reducer_bia_conf_;
        reduce_to_unit_stride_t rtus_;

    protected:
        bool set_default_formats() {
            using namespace format_tag;

            auto dat_tag = utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c);
            auto wei_tag = utils::pick(2 * ndims() - 6 + with_groups(),
                    OIw16i16o, gOIw16i16o, OIhw16i16o, gOIhw16i16o);

            return set_default_formats_common(dat_tag, wei_tag, dat_tag);
        }

    private:
        void init_balancers() {
            const size_t max_buffer_size = jcp_.nthr * 3 * 5 * 5 * 16 * 16;
            if (with_bias()) {
                reducer_bia_conf_.init(reduce_balancer_t(jcp_.nthr,
                            jcp_.oc_block, jcp_.ngroups * jcp_.nb_load,
                            jcp_.mb, max_buffer_size));
            }
        }
    };

    template <cpu_isa_t isa, typename conv_t>
    friend void init_rtus_driver(conv_t *self);

    jit_avx512_common_1x1_convolution_bwd_weights_t(const pd_t *apd);

    ~jit_avx512_common_1x1_convolution_bwd_weights_t() {
        delete kernel_;
        delete acc_ker_;
        delete reducer_bias_;
        delete rtus_driver_;
        delete trans_kernel_;
    }

    typedef typename prec_traits<data_type::f32>::type data_t;

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        execute_backward_weights(ctx);
        return status::success;
    }

  private:
    void execute_backward_weights(const exec_ctx_t &ctx) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }

    jit_avx512_common_1x1_conv_kernel *kernel_;
    cpu_accumulator_1d_t<data_type::f32> *acc_ker_;
    cpu_reducer_t<data_type::f32> *reducer_bias_;
    jit_transpose4x16_src *trans_kernel_;
    rtus_driver_t<avx512_common> *rtus_driver_;
};

}
}
}

#endif
