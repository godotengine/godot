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

#ifndef REF_GEMM_F32_HPP
#define REF_GEMM_F32_HPP

#include "mkldnn_types.h"

namespace mkldnn {
namespace impl {
namespace cpu {

template <typename data_t>
mkldnn_status_t ref_gemm(const char *transa, const char *transb, const int *M,
        const int *N, const int *K, const data_t *alpha, const data_t *A,
        const int *lda, const data_t *B, const int *ldb, const data_t *beta,
        data_t *C, const int *ldc, const data_t *bias);

}
}
}

#endif
