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

#ifndef CPU_REF_DECONVOLUTION_HPP
#define CPU_REF_DECONVOLUTION_HPP

#include <assert.h>
#include <string.h>

#include "c_types_map.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"
#include "primitive_iterator.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_deconvolution_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

static status_t compute_blocked_format(bool with_groups,
        const memory_desc_t *oi_md, memory_desc_t *io_md)
{
    /* Computes blocking for *i*o* format from *o*i* format */

    bool sanity_check_ok = true
        && oi_md->ndims == io_md->ndims
        && oi_md->format_kind == format_kind::blocked;
    if (!sanity_check_ok) return status::invalid_arguments;

    const blocking_desc_t &oi_blk = oi_md->format_desc.blocking;
    blocking_desc_t io_blk = io_md->format_desc.blocking;

    io_md->format_kind = format_kind::blocked;
    io_blk = oi_blk;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;

    nstl::swap(io_blk.strides[ID_OC], io_blk.strides[ID_IC]);
    for (int i_blk = 0; i_blk < io_blk.inner_nblks; ++i_blk) {
        if (utils::one_of(io_blk.inner_idxs[i_blk], ID_OC, ID_IC)) {
            io_blk.inner_idxs[i_blk] =
                (io_blk.inner_idxs[i_blk] == ID_OC ? ID_IC : ID_OC);
        }
    }

    return memory_desc_init_by_blocking_desc(*io_md, io_blk);
}

static status_t conv_descr_create(const deconvolution_desc_t *dd,
        convolution_desc_t *cd)
{
    using namespace prop_kind;
    alg_kind_t alg_kind = dd->alg_kind == alg_kind::deconvolution_direct
        ? alg_kind::convolution_direct : alg_kind::convolution_winograd;

    const memory_desc_t *src_md, *dst_md, *d_weights_d;
    prop_kind_t prop_kind;
    memory_desc_t c_weights_d;
    if (utils::one_of(dd->prop_kind, forward_training, forward_inference)) {
        prop_kind = backward_data;
        src_md = &dd->dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->weights_desc;
    } else if (dd->prop_kind == backward_data) {
        prop_kind = forward_training;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->diff_src_desc;
        d_weights_d = &dd->weights_desc;
    } else {
        prop_kind = dd->prop_kind;
        src_md = &dd->diff_dst_desc;
        dst_md = &dd->src_desc;
        d_weights_d = &dd->diff_weights_desc;
    }

    const bool with_groups = d_weights_d->ndims == src_md->ndims + 1;

    /* create weights desc for convolution */
    c_weights_d = *d_weights_d;

    const int ID_OC = 0 + with_groups;
    const int ID_IC = 1 + with_groups;

    nstl::swap(c_weights_d.dims[ID_OC], c_weights_d.dims[ID_IC]);
    nstl::swap(c_weights_d.padded_dims[ID_OC], c_weights_d.padded_dims[ID_IC]);
    nstl::swap(c_weights_d.padded_offsets[ID_OC], c_weights_d.padded_offsets[ID_IC]);

    if (c_weights_d.format_kind != format_kind::any)
        CHECK(compute_blocked_format(with_groups, d_weights_d, &c_weights_d));

    return conv_desc_init(cd, prop_kind, alg_kind, src_md, &c_weights_d,
            prop_kind != backward_weights ? &dd->bias_desc : nullptr,
            dst_md, dd->strides, dd->dilates,
            dd->padding[0], dd->padding[1], dd->padding_kind);
}

struct ref_deconvolution_fwd_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_fwd_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_fwd_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_fwd_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , conv_supports_bias_(other.conv_supports_bias_)
            , dst_tag_(other.dst_tag_)
        {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_fwd_t);

        status_t init_convolution() {
            using namespace types;

            convolution_desc_t cd;
            CHECK(conv_descr_create(desc(), &cd));

            mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                conv_supports_bias_ =
                    static_cast<cpu_convolution_bwd_data_pd_t *>(conv_pd_)
                    ->support_bias();
                bool output_f32 = utils::everyone_is(data_type::f32,
                        desc()->accum_data_type, desc()->dst_desc.data_type);

                bool ok = true
                    && conv_pd_->weights_md()->extra.flags == 0
                    /* deconv reference code can process only f32 bias */
                    && IMPLICATION(with_bias(),
                            conv_supports_bias_ || output_f32);
                if (ok) return status::success;

                delete conv_pd_;
            }
            conv_pd_ = nullptr;
            return status::unimplemented;
        }

        status_t init() {
            using namespace format_tag;
            bool ok = true
                && is_fwd()
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && attr()->post_ops_.has_default_values();

            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                                conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (dst_md_.format_kind == format_kind::any)
                    dst_md_ = *conv_pd_->diff_src_md();
                if (bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
        bool conv_supports_bias_;
        format_tag_t dst_tag_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_fwd_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_fwd_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_DIFF_DST] = args.at(MKLDNN_ARG_SRC);
        conv_args[MKLDNN_ARG_WEIGHTS] = args.at(MKLDNN_ARG_WEIGHTS);
        if (pd()->with_bias() && pd()->conv_supports_bias_)
            conv_args[MKLDNN_ARG_BIAS] = args.at(MKLDNN_ARG_BIAS);
        conv_args[MKLDNN_ARG_DIFF_SRC] = args.at(MKLDNN_ARG_DST);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[MKLDNN_ARG_SCRATCHPAD] = args.at(MKLDNN_ARG_SCRATCHPAD);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);

        if (pd()->with_bias() && !pd()->conv_supports_bias_) {
            using namespace format_tag;

            auto bias = CTX_IN_MEM(const data_t *, MKLDNN_ARG_BIAS);
            auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);

            switch (pd()->dst_tag_) {
            case ncdhw: case nchw: case ncw:
                compute_fwd_bias_ncdhw(bias, dst);
                break;
            case nCdhw8c: case nChw8c: case nCw8c:
                compute_fwd_bias_nCdhwXc<8>(bias, dst);
                break;
            case nCdhw16c: case nChw16c: case nCw16c:
                compute_fwd_bias_nCdhwXc<16>(bias, dst);
                break;
            default:
                compute_fwd_bias(bias, dst);
                break;
            }
        }
        return status::success;
    }

private:
    void compute_fwd_bias(const data_t *bias, data_t *dst) const;
    void compute_fwd_bias_ncdhw(const data_t *bias, data_t *dst) const;
    template <int blksize> void compute_fwd_bias_nCdhwXc(const data_t *bias,
            data_t *dst) const;
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_data_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_data_pd_t {
        pd_t(engine_t *engine, const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_data_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_data_pd_t(other)
            , conv_pd_(other.conv_pd_->clone()) {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_data_t);

        status_t init_convolution() {
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

            mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                    &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                if (conv_pd_->weights_md()->extra.flags == 0)
                    return status::success;
                delete conv_pd_;
            }

            return status::unimplemented;
        }

        status_t init() {
            using namespace data_type;
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_data
                && utils::everyone_is(data_type::f32,
                        desc()->diff_src_desc.data_type,
                        desc()->weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd);

            if (ok) {
                CHECK(init_convolution());
                if (weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                                conv_pd_->weights_md(), &desc_.weights_desc));
                    weights_md_ = desc_.weights_desc;
                }
                if (diff_src_md_.format_kind == format_kind::any)
                    diff_src_md_ = *conv_pd_->dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_data_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_bwd_data_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_SRC] = args.at(MKLDNN_ARG_DIFF_DST);
        conv_args[MKLDNN_ARG_WEIGHTS] = args.at(MKLDNN_ARG_WEIGHTS);
        conv_args[MKLDNN_ARG_DST] = args.at(MKLDNN_ARG_DIFF_SRC);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[MKLDNN_ARG_SCRATCHPAD] = args.at(MKLDNN_ARG_SCRATCHPAD);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        conv_p_->execute(conv_ctx);
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    primitive_t *conv_p_;
};

struct ref_deconvolution_bwd_weights_t: public cpu_primitive_t {
    struct pd_t: public cpu_deconvolution_bwd_weights_pd_t {
        pd_t(engine_t *engine,
                const deconvolution_desc_t *adesc,
                const primitive_attr_t *attr,
                const deconvolution_fwd_pd_t *hint_fwd_pd)
            : cpu_deconvolution_bwd_weights_pd_t(engine, adesc, attr, hint_fwd_pd)
            , conv_pd_(nullptr)
        {}

        pd_t(const pd_t &other)
            : cpu_deconvolution_bwd_weights_pd_t(other)
            , conv_pd_(other.conv_pd_->clone())
            , dst_tag_(other.dst_tag_)
        {}

        ~pd_t() { delete conv_pd_; }

        DECLARE_COMMON_PD_T(conv_pd_->name(), ref_deconvolution_bwd_weights_t);

        status_t init_convolution() {
            using namespace types;

            convolution_desc_t cd;
            status_t status = conv_descr_create(desc(), &cd);
            if (status != status::success) return status;

            mkldnn_primitive_desc_iterator it(engine_, (op_desc_t *)&cd,
                    &attr_, nullptr);
            while (++it != it.end()) {
                conv_pd_ = *it;
                if (conv_pd_->diff_weights_md()->extra.flags == 0)
                    return status::success;
                delete conv_pd_;
            }
            return status::unimplemented;
        }

        status_t init() {
            using namespace format_tag;
            bool ok = true
                && desc()->prop_kind == prop_kind::backward_weights
                && utils::everyone_is(data_type::f32,
                        desc()->src_desc.data_type,
                        desc()->diff_weights_desc.data_type,
                        desc()->diff_dst_desc.data_type)
                && utils::one_of(desc()->alg_kind,
                        alg_kind::deconvolution_direct,
                        alg_kind::deconvolution_winograd)
                && attr()->has_default_values();
            if (ok) {
                CHECK(init_convolution());
                if (diff_weights_md_.format_kind == format_kind::any) {
                    CHECK(compute_blocked_format(with_groups(),
                                conv_pd_->diff_weights_md(),
                                &desc_.diff_weights_desc));
                    diff_weights_md_ = desc_.diff_weights_desc;
                }
                if (src_md_.format_kind == format_kind::any)
                    src_md_ = *conv_pd_->diff_dst_md();
                if (diff_dst_md_.format_kind == format_kind::any)
                    diff_dst_md_ = *conv_pd_->src_md();
                if (diff_bias_md_.format_kind == format_kind::any)
                    CHECK(memory_desc_init_by_tag(diff_bias_md_, x));

                dst_tag_ = memory_desc_matches_one_of_tag(diff_dst_md_,
                        utils::pick(ndims() - 3, ncw, nchw, ncdhw),
                        utils::pick(ndims() - 3, nCw8c, nChw8c, nCdhw8c),
                        utils::pick(ndims() - 3, nCw16c, nChw16c, nCdhw16c));

                return status::success;
            }

            return status::unimplemented;
        }

        virtual void init_scratchpad_md() override {
            scratchpad_md_ = *conv_pd_->scratchpad_md();
        }

        primitive_desc_t *conv_pd_;
        format_tag_t dst_tag_;
    };

    typedef typename prec_traits<data_type::f32>::type data_t;

    ref_deconvolution_bwd_weights_t(const pd_t *apd): cpu_primitive_t(apd)
    { pd()->conv_pd_->create_primitive((primitive_t **)&conv_p_); }
    ~ref_deconvolution_bwd_weights_t() { delete conv_p_; }

    virtual status_t execute(const exec_ctx_t &ctx) const override {
        const auto &args = ctx.args();
        exec_args_t conv_args;
        conv_args[MKLDNN_ARG_DIFF_DST] = args.at(MKLDNN_ARG_SRC);
        conv_args[MKLDNN_ARG_SRC] = args.at(MKLDNN_ARG_DIFF_DST);
        conv_args[MKLDNN_ARG_DIFF_WEIGHTS] = args.at(MKLDNN_ARG_DIFF_WEIGHTS);
        if (!types::is_zero_md(pd()->scratchpad_md()))
            conv_args[MKLDNN_ARG_SCRATCHPAD] = args.at(MKLDNN_ARG_SCRATCHPAD);
        const exec_ctx_t conv_ctx(ctx.stream(), std::move(conv_args));

        status_t status = conv_p_->execute(conv_ctx);
        if (status != status::success) return status;

        if (pd()->with_bias()) {
            using namespace format_tag;

            auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
            auto diff_bias = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_BIAS);

            switch (pd()->dst_tag_) {
            case ncdhw: case nchw: case ncw:
                compute_bwd_bias_ncdhw(diff_dst, diff_bias);
                break;
            case nCdhw8c: case nChw8c: case nCw8c:
                compute_bwd_bias_nCdhwXc<8>(diff_dst, diff_bias);
                break;
            case nCdhw16c: case nChw16c: case nCw16c:
                compute_bwd_bias_nCdhwXc<16>(diff_dst, diff_bias);
                break;
            default:
                compute_bwd_bias(diff_dst, diff_bias);
                break;
            }
        }
        return status::success;
    }

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
    void compute_bwd_bias(const data_t *diff_dst, data_t *diff_bias) const;
    void compute_bwd_bias_ncdhw(const data_t *diff_dst,
            data_t *diff_bias) const;
    template <int blksize> void compute_bwd_bias_nCdhwXc(
            const data_t *diff_dst, data_t *diff_bias) const;

    primitive_t *conv_p_;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
