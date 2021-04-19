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

#ifndef SIMPLE_CONCAT_HPP
#define SIMPLE_CONCAT_HPP

#include "memory_tracking.hpp"

#include "cpu_concat_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_concat_t: public cpu_primitive_t {
    struct pd_t: public cpu_concat_pd_t {
        using cpu_concat_pd_t::cpu_concat_pd_t;

        pd_t(const pd_t &rhs): cpu_concat_pd_t(rhs) {
            int ndims = rhs.dst_md_.ndims;
            utils::array_copy(perm_, rhs.perm_, ndims);
            utils::array_copy(iperm_, rhs.iperm_, ndims);
            utils::array_copy(blocks_, rhs.blocks_, ndims);
        }

        DECLARE_CONCAT_PD_T("simple:any", simple_concat_t);

        status_t init() {
            const memory_desc_wrapper dst_d(dst_md());
            bool ok = true
                && cpu_concat_pd_t::init() == status::success
                && dst_d.ndims() <= 6;
            if (!ok) return status::unimplemented;

            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                const memory_desc_wrapper o_d(&src_image_mds_[i]);

                const int ignore_strides = 0;

                ok = ok
                    && utils::everyone_is(data_type, i_d.data_type(),
                            o_d.data_type())
                    && utils::everyone_is(format_kind::blocked,
                            i_d.format_kind(), o_d.format_kind())
                    && types::blocking_desc_is_equal(i_d.blocking_desc(),
                            o_d.blocking_desc(), ignore_strides)
                    && types::blocking_desc_is_equal(i_d.blocking_desc(),
                            dst_d.blocking_desc(), ignore_strides)
                    && !i_d.is_additional_buffer();
                if (!ok) return status::unimplemented;
            }

            dst_d.compute_blocks(blocks_);
            format_perm();

            // start dim is the first dimension after which the concatenation
            // would happen contiguously
            const int start_dim = perm_[concat_dim()];

            // check that contiguous part is indeed contiguous (i.e. dense)
            if (nelems_to_concat(dst_d) !=
                    dst_d.padded_dims()[concat_dim()] / blocks_[concat_dim()]
                    * dst_d.blocking_desc().strides[concat_dim()])
                return status::unimplemented;

            // check that all inputs have the same strides for the
            // contiguous part [concat_dim .. ndims] for the *major* dims.
            // the block part is already checked above
            for (size_t i = 0; i < src_mds_.size(); ++i) {
                const memory_desc_wrapper i_d(&src_mds_[i]);
                for (int d = start_dim; d < dst_d.ndims(); ++d) {
                    if (dst_d.blocking_desc().strides[iperm_[d]]
                            != i_d.blocking_desc().strides[iperm_[d]])
                        return status::unimplemented;
                }
            }

            init_scratchpad();

            return status::success;
        }

        int perm_[MKLDNN_MAX_NDIMS] {};
        int iperm_[MKLDNN_MAX_NDIMS] {};
        dims_t blocks_ {};

        dim_t nelems_to_concat(const memory_desc_wrapper &data_d) const {
            const int ndims = data_d.ndims();

            dim_t nelems = 1;
            for (int i = perm_[concat_dim()]; i < ndims; i++)
                nelems *= data_d.dims()[iperm_[i]] / blocks_[iperm_[i]];
            for (int i = 0; i < ndims; i++)
                nelems *= blocks_[i];

            return nelems;
        }

    private:
        void format_perm() {
            const memory_desc_wrapper dst_d(dst_md());
            const int ndims = dst_d.ndims();

            strides_t strides;
            utils::array_copy(strides, dst_d.blocking_desc().strides, ndims);
            for (int i = 0; i < ndims; i++) iperm_[i] = i;

            utils::simultaneous_sort(strides, iperm_, ndims,
                    [](stride_t a, stride_t b) { return b - a; });

            for (int i = 0; i < ndims; i++) perm_[iperm_[i]] = i;
        }

        void init_scratchpad() {
            using namespace memory_tracking::names;
            auto scratchpad = scratchpad_registry().registrar();
            scratchpad.book(key_concat_iptrs, sizeof(data_t *) * n_inputs());
            scratchpad.book(key_concat_optrs, sizeof(data_t *) * n_inputs());
            scratchpad.book(key_concat_nelems, sizeof(dim_t) * n_inputs());
            scratchpad.book(key_concat_istrides,
                    sizeof(strides_t) * n_inputs());
        }
    };

    simple_concat_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif
