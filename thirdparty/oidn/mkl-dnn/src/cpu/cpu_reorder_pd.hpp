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

#ifndef CPU_REORDER_PD_HPP
#define CPU_REORDER_PD_HPP

#include <assert.h>

#include "c_types_map.hpp"
#include "reorder_pd.hpp"
#include "utils.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct cpu_reorder_pd_t: public reorder_pd_t {
    using reorder_pd_t::reorder_pd_t;

    status_t init() {
        const auto &post_ops = attr()->post_ops_;
        bool args_ok = IMPLICATION(post_ops.len_ != 0, post_ops.len_ == 1
                && post_ops.entry_[0].kind == primitive_kind::sum);
        scratchpad_engine_ = src_engine_;
        return args_ok ? status::success : status::unimplemented;
    }
};

}
}
}

#endif

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
