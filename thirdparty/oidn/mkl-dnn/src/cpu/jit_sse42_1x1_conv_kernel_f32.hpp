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

#ifndef JIT_SSE42_1x1_CONV_KERNEL_F32_HPP
#define JIT_SSE42_1x1_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "cpu_memory.hpp"
#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_sse42_1x1_conv_kernel_f32: public jit_generator {
    jit_sse42_1x1_conv_kernel_f32(jit_1x1_conv_conf_t ajcp,
            const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<sse42>(this,
                    jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_1x1_conv_call_s *))this->getCode();
    }

    ~jit_sse42_1x1_conv_kernel_f32() {
        delete eltwise_injector_;
    }

    static bool post_ops_ok(jit_1x1_conv_conf_t &jcp,
            const primitive_attr_t &attr);

    static status_t init_conf(jit_1x1_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d,
            const primitive_attr_t &attr);

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_sse42_1x1_conv_kernel_f32)

    jit_1x1_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker)(jit_1x1_conv_call_s *);

private:
    using reg64_t = const Xbyak::Reg64;
    using xmm_t = const Xbyak::Xmm;

    reg64_t reg_bcast_data = rax;
    reg64_t reg_load_data = rsi;
    reg64_t reg_output_data = rbx;
    reg64_t aux_reg_bcast_data = rdx;
    reg64_t aux1_reg_bcast_data = abi_not_param1;
    reg64_t aux_reg_load_data = abi_param1;
    reg64_t aux_reg_output_data = rbp;
    reg64_t reg_load_loop_work = r9;
    reg64_t reg_bcast_loop_work = r10;
    reg64_t reg_reduce_loop_work = r11;
    reg64_t load_loop_iter = r13;
    reg64_t imm_addr64 = load_loop_iter;
    reg64_t bcast_loop_iter = r14;
    reg64_t reduce_loop_iter = r15;
    reg64_t reg_reduce_pos_flag = r8;
    reg64_t reg_output_stride = r12;
    reg64_t reg_bias_data = r12;
    reg64_t reg_diff_bias_data = bcast_loop_iter;

    int reg_diff_bias_data_stack_offt = 0;
    int stack_space_needed = 8;

    xmm_t reg_bcast = xmm_t(15);

    jit_uni_eltwise_injector_f32<sse42> *eltwise_injector_;

    void generate_bcast_loop(int load_loop_blk);
    void generate_reduce_loop(int load_loop_blk, int ur);
    void generate_diff_bias_loop(int load_loop_blk);

    void generate();
};

}
}
}

#endif
