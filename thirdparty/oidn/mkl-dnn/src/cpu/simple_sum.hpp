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

#ifndef SIMPLE_SUM_HPP
#define SIMPLE_SUM_HPP

#include "cpu_sum_pd.hpp"
#include "cpu_primitive.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <data_type_t data_type>
struct simple_sum_t: public cpu_primitive_t {
    struct pd_t: public cpu_sum_pd_t {
        using cpu_sum_pd_t::cpu_sum_pd_t;

        DECLARE_SUM_PD_T("simple:any", simple_sum_t);

        status_t init() {
            const int n = n_inputs();

            bool ok = true
                && cpu_sum_pd_t::init() == status::success
                && n <= max_num_arrs;
            if (!ok) return status::unimplemented;

            const memory_desc_wrapper o_d(dst_md());
            ok = ok
                && o_d.data_type() == data_type
                && o_d.is_dense();
            if (!ok) return status::unimplemented;

            for (int i = 0; i < n; ++i) {
                const memory_desc_wrapper i_d(src_md(i));
                if (i_d != o_d) return status::unimplemented;
            }

            return status::success;
        }
    };

    simple_sum_t(const pd_t *apd): cpu_primitive_t(apd) {}

    virtual status_t execute(const exec_ctx_t &ctx) const override;

    enum {max_num_arrs = 16 };
    typedef typename prec_traits<data_type>::type data_t;

private:
    const pd_t *pd() const { return (const pd_t *)primitive_t::pd(); }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
