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

#ifndef CPU_INNER_PRODUCT_PD_HPP
#define CPU_INNER_PRODUCT_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "inner_product_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace {
inline bool dense_gemm_consitency_check(const memory_desc_wrapper &src_d,
        const memory_desc_wrapper &wei_d, const memory_desc_wrapper &dst_d) {
    using namespace utils;

    auto strides_compatible = [&]() {
        bool ok = true;
        auto w_str = wei_d.blocking_desc().strides;
        auto d_str = src_d.blocking_desc().strides;
        for (int i = 1; i < src_d.ndims() - 1; i++) {
            ok = ok && w_str[i] / d_str[i] == w_str[i + 1] / d_str[i + 1];
        }
        return ok && one_of(w_str[1] / d_str[1], 1, wei_d.padded_dims()[0]);
    };
    return true && src_d.is_blocking_desc() && wei_d.is_blocking_desc()
            && src_d.ndims() == wei_d.ndims()
            && src_d.blocking_desc().inner_nblks
            == wei_d.blocking_desc().inner_nblks
            && utils::one_of(src_d.blocking_desc().inner_nblks, 0, 1)
            && array_cmp(src_d.blocking_desc().inner_blks,
                       wei_d.blocking_desc().inner_blks,
                       wei_d.blocking_desc().inner_nblks)
            && array_cmp(src_d.blocking_desc().inner_idxs,
                       wei_d.blocking_desc().inner_idxs,
                       wei_d.blocking_desc().inner_nblks)
            && strides_compatible()
            && dst_d.matches_tag(format_tag::nc)
            && src_d.only_padded_dim(1)
            && wei_d.only_padded_dim(1)
            && src_d.padded_dims()[1] == wei_d.padded_dims()[1]
            && src_d.is_dense(true)
            && dst_d.is_dense()
            && wei_d.is_dense(true);
}
}

struct cpu_inner_product_fwd_pd_t: public inner_product_fwd_pd_t {
    using inner_product_fwd_pd_t::inner_product_fwd_pd_t;
};

struct cpu_inner_product_bwd_data_pd_t: public inner_product_bwd_data_pd_t {
    using inner_product_bwd_data_pd_t::inner_product_bwd_data_pd_t;
};

struct cpu_inner_product_bwd_weights_pd_t: public inner_product_bwd_weights_pd_t {
    using inner_product_bwd_weights_pd_t::inner_product_bwd_weights_pd_t;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
