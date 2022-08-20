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

#ifndef JIT_AVX512_COMMON_CONV_WINOGRAD_KERNEL_F32_HPP
#define JIT_AVX512_COMMON_CONV_WINOGRAD_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

//alpha determines the output tile_size
constexpr int alpha = 6;
constexpr int tile_size = 4;
//simd length used for vectorization
constexpr int simd_w = 16;

struct _jit_avx512_common_conv_winograd_data_kernel_f32 : public jit_generator {
    _jit_avx512_common_conv_winograd_data_kernel_f32(
            jit_conv_winograd_conf_t ajcp)
        : jcp(ajcp)
    {
        //******************* First iter kernel ********************//
        this->gemm_loop_generate(true);
        gemm_loop_ker_first_iter
                = (decltype(gemm_loop_ker_first_iter)) this->getCode();

        //************** Subsequent iterations kernel **************//
        if (jcp.dimK_nb_block > 1) {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->gemm_loop_generate(false);
            gemm_loop_ker = (decltype(gemm_loop_ker))addr;
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_common_conv_winograd_data_kernel_f32)

    static status_t init_conf_common(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d);

    static status_t init_conf_kernel(
            jit_conv_winograd_conf_t &jcp, int dimM, int dimN, int dimK);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *);
    void (*gemm_loop_ker_first_iter)(float *, const float *, const float *);

protected:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    void gemm_loop_generate(bool is_beta_zero);

    /* registers used for GEMM */
    reg64_t reg_dstC = abi_param1;
    reg64_t reg_srcA = abi_param2;
    reg64_t reg_srcB = abi_param3;

    reg64_t reg_dimM_block_loop_cnt = r10;
    reg64_t reg_dimK_block_loop_cnt = r11;
};

struct jit_avx512_common_conv_winograd_fwd_kernel_f32
        : _jit_avx512_common_conv_winograd_data_kernel_f32 {
    using _jit_avx512_common_conv_winograd_data_kernel_f32::
            _jit_avx512_common_conv_winograd_data_kernel_f32;

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);
};

struct jit_avx512_common_conv_winograd_bwd_data_kernel_f32
        : public _jit_avx512_common_conv_winograd_data_kernel_f32 {
    using _jit_avx512_common_conv_winograd_data_kernel_f32::
            _jit_avx512_common_conv_winograd_data_kernel_f32;

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
};

struct jit_avx512_common_conv_winograd_bwd_weights_kernel_f32
        : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_common_conv_winograd_bwd_weights_kernel_f32)

    jit_avx512_common_conv_winograd_bwd_weights_kernel_f32(
            jit_conv_winograd_conf_t ajcp)
        : jcp(ajcp)
    {

        //******************* First iter kernel ********************//
        {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->gemm_loop_generate(true);
            gemm_loop_ker_first_iter = (decltype(gemm_loop_ker_first_iter))addr;
        }

        if (jcp.tile_block > 1) {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->gemm_loop_generate(false);
            gemm_loop_ker = (decltype(gemm_loop_ker))addr;
        }

        if (jcp.ver == ver_4fma) {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->transpose_ker_generate();
            transpose_4fma_ker = (decltype(transpose_4fma_ker))addr;
        }
    }

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_dst_d,
            const memory_desc_wrapper &diff_weights_d);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *);
    void (*gemm_loop_ker_first_iter)(float *, const float *, const float *);
    void (*transpose_4fma_ker)(float *, float *);

private:
    using reg64_t = const Xbyak::Reg64;
    enum { typesize = sizeof(float) };

    void gemm_loop_generate(bool is_first_tile);
    void transpose_ker_generate();

    reg64_t reg_origB = abi_param2;
    reg64_t reg_transB = abi_param1;

    reg64_t reg_dstC = abi_param1;
    reg64_t reg_srcA_const = abi_param2;
    reg64_t reg_srcB = abi_param3;

    reg64_t reg_sp = rsp;
    reg64_t reg_srcA = r9;
    reg64_t reg_nb_ic = r10;
    reg64_t reg_loop_cpt = r11;
    reg64_t reg_transB_idx = r13;

    /* Registers used by new kernel */
    reg64_t reg_dimM_block_loop_cnt = r10;
    reg64_t reg_dimK_block_loop_cnt = r12;
    reg64_t reg_dimN_block_loop_cnt = r11;
};
}
}
}

#endif
