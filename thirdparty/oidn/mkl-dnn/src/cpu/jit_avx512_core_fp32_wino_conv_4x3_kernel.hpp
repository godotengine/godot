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

#ifndef JIT_AVX512_CORE_FP32_WINO_CONV_4x3_KERNEL_HPP
#define JIT_AVX512_CORE_FP32_WINO_CONV_4x3_KERNEL_HPP

#include "c_types_map.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"

#include "jit_avx512_common_conv_winograd_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct _jit_avx512_core_fp32_wino_conv_4x3_data_kernel
    : public jit_generator {
    _jit_avx512_core_fp32_wino_conv_4x3_data_kernel(
            jit_conv_winograd_conf_t ajcp)
        : jcp(ajcp) {
        {
            this->weights_transform_data_ker_generate();
            weights_transform_data_ker
                    = (decltype(weights_transform_data_ker)) this->getCode();
        }
        {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->input_transform_data_ker_generate();
            input_transform_data_ker = (decltype(input_transform_data_ker))addr;
        }
        {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->output_transform_data_ker_generate();
            output_transform_data_ker
                = (decltype(output_transform_data_ker))addr;
        }
        {
            align();
            const Xbyak::uint8 *addr = getCurr();
            this->gemm_loop_generate();
            gemm_loop_ker = (decltype(gemm_loop_ker))addr;
        }
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_fp32_wino_conv_4x3_data_kernel)

    static status_t init_conf_common(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d);

    static status_t init_conf_kernel(
            jit_conv_winograd_conf_t &jcp, int dimM, int dimN, int dimK);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *, const int);
    void (*input_transform_data_ker)(jit_wino_transform_call_s *);
    void (*output_transform_data_ker)(jit_wino_transform_call_s *);
    void (*weights_transform_data_ker)(jit_wino_transform_call_s *);

protected:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    enum { typesize = sizeof(float) };

    void gemm_loop_generate();
    void input_transform_data_ker_generate();
    void output_transform_data_ker_generate();
    void weights_transform_data_ker_generate();

    /* registers used for GEMM */
    reg64_t reg_dstC = abi_param1;
    reg64_t reg_srcA = abi_param2;
    reg64_t reg_srcB = abi_param3;
    reg64_t reg_is_beta_zero = abi_param4;

    reg64_t reg_dimM_block_loop_cnt = r10;
    reg64_t reg_dimK_block_loop_cnt = r11;

    /* registers used for transforms*/
    reg64_t param = abi_param1;

    /* registers used for output_transform_data_ker */
    reg64_t oreg_temp = abi_not_param1;
    reg64_t oreg_Ow = r9;
    reg64_t oreg_src = r11;
    reg64_t oreg_tile_block = r12;
    reg64_t oreg_tile_block_ur = r13;
    reg64_t oreg_nb_tile_block_ur = r14;
    reg64_t oreg_O = r8;
    reg64_t oreg_T = r10;
    reg64_t oreg_dst = r11;
    reg64_t oreg_ydim = r14;
    reg64_t oreg_xdim = r15;
    reg64_t oreg_out_j = r12;
    reg64_t oreg_bias = rbx;
    reg64_t imm_addr64 = rax;

    /* registers used for input_transform_data_ker */
    reg64_t ireg_temp = abi_not_param1;
    reg64_t ireg_jtiles = rax;
    reg64_t ireg_itiles = rbx;
    reg64_t ireg_I = r8;
    reg64_t ireg_src = r13;
    reg64_t ireg_ydim = r14;
    reg64_t ireg_xdim = r15;
    reg64_t ireg_inp_j = r12;
    reg64_t ireg_inp_i = rdx;
    reg64_t ireg_mask_j = r11;
    reg64_t ireg_mask = rsi;
    reg32_t ireg_mask_32 = esi;
    reg64_t ireg_zero = r9;
    reg64_t ireg_Iw = r9;
    reg64_t ireg_T = r10;
    reg64_t ireg_tile_block = r12;
    reg64_t ireg_tile_block_ur = r13;
    reg64_t ireg_nb_tile_block_ur = r14;
    reg64_t ireg_output = r15;

    /* registers used for wei transform */
    reg64_t wreg_temp = abi_not_param1;
    reg64_t wreg_F = r8;
    reg64_t wreg_src = r9;
    reg64_t wreg_MT = r15;
    reg64_t wreg_M = r14;
    reg64_t wreg_dst = r10;
    reg64_t wreg_dst_aux = r9;
    reg64_t wreg_dst_idx = r8;
    reg64_t wreg_Fw = r11;
    reg64_t wreg_T = r12;
    reg64_t wreg_cnt_j = rdx;
    reg64_t wreg_F_aux = r14;
    reg64_t wreg_Fw_aux = r15;
};

struct jit_avx512_core_fp32_wino_conv_4x3_fwd_kernel
        : _jit_avx512_core_fp32_wino_conv_4x3_data_kernel {
    using _jit_avx512_core_fp32_wino_conv_4x3_data_kernel::
            _jit_avx512_core_fp32_wino_conv_4x3_data_kernel;

    static bool post_ops_ok(jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_t &src_md,
            memory_desc_t &weights_md, const memory_desc_t &dst_md,
            const primitive_attr_t &attr);
};

struct jit_avx512_core_fp32_wino_conv_4x3_bwd_data_kernel
        : public _jit_avx512_core_fp32_wino_conv_4x3_data_kernel {
    using _jit_avx512_core_fp32_wino_conv_4x3_data_kernel::
            _jit_avx512_core_fp32_wino_conv_4x3_data_kernel;

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
};

struct jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel
        : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
        _jit_avx512_core_conv_winograd_bwd_weights_kernel_f32)

    jit_avx512_core_fp32_wino_conv_4x3_bwd_weights_kernel(
            jit_conv_winograd_conf_t ajcp)
        : jcp(ajcp)
    {
        //******************* First iter kernel ********************//
        this->gemm_loop_generate(true);
        gemm_loop_ker_first_iter = (decltype(gemm_loop_ker_first_iter))this->getCode();

        align();
        const Xbyak::uint8 *addr = getCurr();
        this->src_transform_generate();
        src_transform = (decltype(src_transform))addr;

        if (jcp.with_bias) {
            align();
            addr = getCurr();
            this->diff_dst_transform_generate(true);
            diff_dst_transform_wbias = (decltype(diff_dst_transform_wbias))addr;
        }

        align();
        addr = getCurr();
        this->diff_dst_transform_generate(false);
        diff_dst_transform = (decltype(diff_dst_transform))addr;

        if (jcp.sched_policy != WSCHED_WEI_SDGtWo && jcp.tile_block > 1) {
            align();
            addr = getCurr();
            this->gemm_loop_generate(false);
            gemm_loop_ker = (decltype(gemm_loop_ker))addr;
        }

        align();
        addr = getCurr();
        this->diff_weights_transform_generate(true);
        diff_weights_transform = (decltype(diff_weights_transform))addr;

        if (jcp.sched_policy == WSCHED_WEI_SDGtWo) {
            align();
            addr = getCurr();
            this->diff_weights_transform_generate(false);
            diff_weights_transform_accum =
                (decltype(diff_weights_transform_accum))addr;
        };
    }

    static status_t init_conf(jit_conv_winograd_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_dst_d,
            const memory_desc_wrapper &diff_weights_d);

    jit_conv_winograd_conf_t jcp;
    void (*gemm_loop_ker)(float *, const float *, const float *);
    void (*gemm_loop_ker_first_iter)(float *, const float *, const float *);
    void (*src_transform)(jit_wino_transform_call_s *);
    void (*diff_dst_transform)(jit_wino_transform_call_s *);
    void (*diff_dst_transform_wbias)(jit_wino_transform_call_s *);
    void (*diff_weights_transform)(jit_wino_transform_call_s *);
    void (*diff_weights_transform_accum)(jit_wino_transform_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using reg32_t = const Xbyak::Reg32;
    enum { typesize = sizeof(float) };

    void src_transform_generate();
    void diff_dst_transform_generate(bool with_bias);
    void diff_weights_transform_generate(bool first_tile);

    /*registers common to transforms*/
    reg64_t reg_transp = abi_param1;
    reg64_t reg_ti = rbx;
    reg64_t reg_tj = abi_not_param1;
    reg64_t reg_src = r8;
    reg64_t reg_dst = r9;
    reg64_t reg_G = rsi; /*TODO: check if this is ok*/
    reg64_t reg_temp = rsi;

    /*registers common to src/diff_dst transform*/
    reg64_t reg_I = r10;
    reg64_t reg_ydim = r11;
    reg64_t reg_xdim = r12;
    reg64_t reg_src_offset = r13;
    reg64_t reg_zero = r14;
    reg64_t reg_tile_count = r15;
    reg64_t reg_maski = rsi;
    reg32_t reg_maski_32 = esi;
    reg64_t reg_maskj = rdx;

    reg64_t reg_T = rax;
    reg64_t reg_oc_ur = rax;
    reg64_t reg_ic_simd = r14;
    reg64_t reg_bias = r10;

    void gemm_loop_generate(bool is_first_tile);

    reg64_t reg_dstC = abi_param1;
    reg64_t reg_srcA = abi_param2;
    reg64_t reg_srcB = abi_param3;

    reg64_t reg_dimM_block_loop_cnt = r9;
    reg64_t reg_dimN_block_loop_cnt = r10;
    reg64_t reg_nb_dimN_bcast_ur = r11;
    reg64_t reg_dimK_block_loop_cnt = r12;
};
}
}
}

#endif
