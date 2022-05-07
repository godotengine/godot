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

#include "memory.hpp"
#include "primitive.hpp"
#include "primitive_exec_types.hpp"

namespace mkldnn {
namespace impl {

status_t cvt_primtive_args(const primitive_desc_t *pd, int nargs,
        const mkldnn_exec_arg_t *c_args, exec_args_t &args) {
    using namespace status;

    if (!IMPLICATION(nargs > 0, c_args != nullptr)) return invalid_arguments;

    int n_inputs = 0;
    int n_outputs = 0;

    for (int i = 0; i < nargs; ++i) {
        primitive_arg_index_t arg = c_args[i].arg;
        auto *mem = c_args[i].memory;

        switch (pd->arg_usage(arg)) {
        case primitive_desc_t::arg_usage_t::input:
            if (args.count(arg) != 0) return invalid_arguments;
            args[arg] = {mem, true};
            n_inputs++;
            break;
        case primitive_desc_t::arg_usage_t::output:
            if (args.count(arg) != 0) return invalid_arguments;
            args[arg] = {mem, false};
            n_outputs++;
            break;
        case primitive_desc_t::arg_usage_t::unused:
            break;
        }
    }

    bool scratchpad_required = !types::is_zero_md(pd->scratchpad_md());

    if (n_inputs != pd->n_inputs()) return invalid_arguments;
    if (n_outputs != pd->n_outputs() + (scratchpad_required ? 1 : 0))
        return invalid_arguments;

    return success;
}

const void *exec_ctx_t::input(primitive_arg_index_t arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto ma = args_.at(arg);
    assert(ma.is_const);
    void *ptr;
    status_t status = ma.mem->get_data_handle(&ptr);
    assert(status == status::success); MAYBE_UNUSED(status);
    return ptr;
}

void *exec_ctx_t::output(primitive_arg_index_t arg) const {
    if (args_.count(arg) != 1) return nullptr;
    const auto ma = args_.at(arg);
    assert(!ma.is_const);
    void *ptr;
    status_t status = ma.mem->get_data_handle(&ptr);
    assert(status == status::success); MAYBE_UNUSED(status);
    return ptr;
}

const memory_t *exec_ctx_t::memory(primitive_arg_index_t arg) const {
    assert(args_.count(arg) == 1);
    const auto ma = args_.at(arg);
    assert(!ma.is_const);
    return ma.mem;
}

}
}
