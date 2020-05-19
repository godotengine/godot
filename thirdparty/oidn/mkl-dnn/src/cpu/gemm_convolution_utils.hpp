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

#ifndef CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP
#define CPU_JIT_GEMM_CONVOLUTION_UTILS_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"

#include "cpu_convolution_pd.hpp"
#include "cpu_engine.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace jit_gemm_convolution_utils {

void im2col_3d(const jit_gemm_conv_conf_t &jcp, const float *im, float *col,
        int od);
void im2col(const jit_gemm_conv_conf_t &jcp, const float *__restrict im,
       float *__restrict col, int hs, int hb, int ws, int wb);
template <typename T>
void im2col_u8(const jit_gemm_conv_conf_t &jcp, const T *__restrict im,
        T* __restrict imtr, uint8_t *__restrict col,
        int hs, int hb, int ws, int wb);

void col2im_s32(const jit_gemm_conv_conf_t &jcp, const int32_t *__restrict col,
        int32_t *__restrict im);
void col2im_3d(const jit_gemm_conv_conf_t &jcp, const float *col, float *im,
        int od);
void col2im(const jit_gemm_conv_conf_t &jcp, const float *col, float *im);

status_t init_conf(jit_gemm_conv_conf_t &jcp,
        memory_tracking::registrar_t &scratchpad, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, int max_threads);

void bwd_weights_balance(int ithr, int nthr, int ngroups, int mb,
        int &ithr_g, int &nthr_g, int &ithr_mb, int &nthr_mb);
void bwd_weights_reduction_par(int ithr, int nthr,
        const jit_gemm_conv_conf_t &jcp, const float *weights_reduce_ws,
        float *weights);

}

}
}
}

#endif
