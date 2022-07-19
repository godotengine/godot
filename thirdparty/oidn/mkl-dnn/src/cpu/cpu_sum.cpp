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

#include "cpu_engine.hpp"

/*
#include "cpu/ref_sum.hpp"
#include "cpu/simple_sum.hpp"
*/

namespace mkldnn {
namespace impl {
namespace cpu {

using spd_create_f = mkldnn::impl::engine_t::sum_primitive_desc_create_f;

namespace {
#define INSTANCE(...) __VA_ARGS__::pd_t::create
static const spd_create_f cpu_sum_impl_list[] = {
    /*
    INSTANCE(simple_sum_t<data_type::f32>),
    INSTANCE(ref_sum_t),
    */
    nullptr,
};
#undef INSTANCE
}

const spd_create_f *cpu_engine_t::get_sum_implementation_list() const {
    return cpu_sum_impl_list;
}

}
}
}
