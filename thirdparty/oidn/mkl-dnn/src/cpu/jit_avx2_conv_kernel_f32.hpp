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

#ifndef JIT_AVX2_CONV_KERNEL_F32_HPP
#define JIT_AVX2_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "cpu_memory.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_avx2_conv_fwd_kernel_f32: public jit_generator {
    jit_avx2_conv_fwd_kernel_f32(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx2>(this,
                    jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_avx2_conv_fwd_kernel_f32() {
        delete eltwise_injector_;
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_fwd_kernel_f32)

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t aux_reg_input = r8;
    reg64_t reg_kernel = rdx;
    reg64_t aux_reg_kernel = r9;
    reg64_t reg_output = rsi;
    reg64_t reg_bias = rbx;

    reg64_t aux_reg_inp_d = r11;
    reg64_t aux_reg_ker_d = abi_not_param1;

    reg64_t reg_ki = rsi;
    reg64_t kj = r10;
    reg64_t oi_iter = r11;
    reg64_t ki_iter = r12;
    reg64_t reg_kh = abi_not_param1;
    reg64_t reg_oc_blocks = r14;
    reg64_t imm_addr64 = r15;
    reg64_t reg_long_offt = r15;
    Xbyak::Reg32 reg_ci_flag = r13d;

    Xbyak::Ymm ytmp = Xbyak::Ymm(14);

    jit_uni_eltwise_injector_f32<avx2> *eltwise_injector_;

    inline void oh_step_unroll_kw(int ur_w, int pad_l, int pad_r,
            int oc_blocks);
    inline void oh_step_nopad(int ur_w, int pad_l, int pad_r,
            char pad_label, int oc_blocks, char oc_blocks_label);
    inline void width_blk_step(int ur_w, int pad_l, int pad_r,
            char pad_label, int oc_blocks, char oc_blocks_label);
    inline void solve_common(int oc_blocks, char oc_blocks_label);

    void generate();
};

struct jit_avx2_conv_bwd_data_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_data_kernel_f32)

    jit_avx2_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;

    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_ddst   = r8;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t reg_dsrc       = rsi;
    reg64_t aux_reg_ddst_oc_loop  = rbx; // used in ndims < 5 case only
    reg64_t aux_reg_kernel_oc_loop = abi_not_param1; /* used in ndims < 5
                                                        case only */

    reg64_t aux_reg_dst_d = r12; // used in ndims == 5 case only
    reg64_t aux_reg_ker_d = r14; // used in ndims == 5 case only

    reg64_t reg_ki  = abi_not_param1; // used in ndims == 5 case only
    reg64_t kj      = r11;
    reg64_t oi_iter = r12;
    reg64_t reg_kh  = r14;
    reg64_t reg_channel = r13;  // used in ndims < 5 case only
    reg64_t reg_channel_work = r9;  // used in ndims < 5 case only
    reg64_t reg_long_offt = r15;

    inline void compute_loop(int ur_w, int l_overflow, int r_overflow);

    void generate();

    inline int get_iw_start(int ki, int l_overflow)
    {
        int res = (jcp.iw - 1 + jcp.r_pad) % jcp.stride_w
                + l_overflow * jcp.stride_w
                - (jcp.kw - 1 - ki) * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return res;
    }

    inline int get_iw_end(int ur_w, int ki, int r_overflow)
    {
        if (utils::one_of(ur_w, jcp.iw, jcp.ur_w_tail))
            ur_w += nstl::min(0, jcp.r_pad); // remove negative padding
        int res = (ur_w - 1 + jcp.l_pad) % jcp.stride_w
                + r_overflow * jcp.stride_w - ki * (jcp.dilate_w + 1);
        while (res < 0)
            res += jcp.stride_w;

        return ur_w - res;
    }
};

struct jit_avx2_conv_bwd_weights_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx2_conv_bwd_weights_kernel_f32)

    jit_avx2_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp)
    {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    reg64_t reg_input = rax;
    reg64_t reg_kernel = rdx;
    reg64_t reg_output = rsi;
    reg64_t b_ic = abi_not_param1;
    reg64_t kj = r8;
    reg64_t reg_kh = r9;
    reg64_t reg_ur_w_trips = r10;
    reg64_t reg_tmp = r11;
    reg64_t reg_oj = r15;
    reg64_t reg_ih_count = rbx;
    reg64_t aux_reg_input = r12;
    reg64_t aux_reg_kernel = r13;
    reg64_t ki = r14;
    reg64_t reg_long_offt = r11;

    inline void od_step_comeback_pointers();
    inline void oh_step_comeback_pointers();
    inline void compute_ic_block_step(int ur_w, int pad_l, int pad_r,
            int ic_block_step, int input_offset, int kernel_offset,
            int output_offset);
    inline void compute_oh_step_disp();
    inline void compute_oh_step_unroll_ow(int ic_block_step, int max_ur_w);
    inline void compute_oh_step_common(int ic_block_step, int max_ur_w);
    inline void compute_oh_loop_common();

    void generate();
};

}
}
}

#endif
