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

#ifndef REF_GEMM_S8X8S32_HPP
#define REF_GEMM_S8X8S32_HPP

#include <stdint.h>

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename b_dt>
mkldnn_status_t ref_gemm_s8x8s32(const char *transa, const char *transb,
        const char *offsetc, const int *M, const int *N, const int *K,
        const float *alpha, const int8_t *A, const int *LDA, const int8_t *ao,
        const b_dt *B, const int *LDB, const int8_t *bo, const float *beta,
        int32_t *C, const int *LDC, const int32_t *co);

}
}
}
#endif
