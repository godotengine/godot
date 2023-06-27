/*******************************************************************************
* Copyright 2019 Intel Corporation
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

#ifndef JIT_SUPPORT_HPP
#define JIT_SUPPORT_HPP

namespace mkldnn {
namespace impl {
namespace cpu {
namespace jit_utils {

void register_jit_code(const void *code, size_t code_size,
        const char *code_name, const char *source_file_name);

}
}
}
}
#endif
