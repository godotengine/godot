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

#ifndef CPU_BATCH_NORMALIZATION_PD_HPP
#define CPU_BATCH_NORMALIZATION_PD_HPP

#include "batch_normalization_pd.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_batch_normalization_fwd_pd_t: public batch_normalization_fwd_pd_t {
    using batch_normalization_fwd_pd_t::batch_normalization_fwd_pd_t;
};

struct cpu_batch_normalization_bwd_pd_t: public batch_normalization_bwd_pd_t {
    using batch_normalization_bwd_pd_t::batch_normalization_bwd_pd_t;
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
