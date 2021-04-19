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

#ifndef JIT_UNI_DW_CONV_KERNEL_F32_HPP
#define JIT_UNI_DW_CONV_KERNEL_F32_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template <cpu_isa_t isa>
struct jit_uni_dw_conv_fwd_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_fwd_kernel_f32)

    jit_uni_dw_conv_fwd_kernel_f32(jit_conv_conf_t ajcp)
        : jcp(ajcp), eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<isa>(this,
                    jcp.eltwise);

        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    ~jit_uni_dw_conv_fwd_kernel_f32() {
        delete eltwise_injector_;
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);
    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &dst_d, const primitive_attr_t &attr);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    const Xbyak::AddressFrame &vmmword = (isa == sse42)
        ? xword : (isa == avx2) ? yword : zword;
    const int vlen = cpu_isa_traits<isa>::vlen;

    // dw convolution
    reg64_t reg_input = r8;
    reg64_t aux_reg_input = r9;
    reg64_t aux1_reg_input = r10;
    reg64_t reg_kernel = r11;
    reg64_t aux_reg_kernel = r12;
    reg64_t aux1_reg_kernel = r13;
    reg64_t reg_output = r14;
    reg64_t reg_bias = r15;
    reg64_t reg_kh = rax;
    reg64_t reg_kw = rbx;
    reg64_t iter_kh = rdx;
    reg64_t iter_kw = rsi;
    reg64_t reg_ur_w = rbp;
    reg64_t reg_ch_blocks = aux1_reg_input;
    reg64_t imm_addr64 = aux1_reg_input;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    inline void load_src(int ur_ch_blocks, int ur_w);
    inline void apply_filter(int ur_ch_blocks, int ur_w);
    inline void apply_filter_unrolled(int ur_ch_blocks, int ur_w);
    inline void apply_activation(int ur_ch_blocks, int ur_w);
    inline void store_dst(int ur_ch_blocks, int ur_w);
    inline void loop_body(int ur_ch_blocks);

    jit_uni_eltwise_injector_f32<isa> *eltwise_injector_;

    void generate();
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_data_kernel_f32: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_data_kernel_f32)

    jit_uni_dw_conv_bwd_data_kernel_f32(jit_conv_conf_t ajcp): jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_conv_call_s *))this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            const memory_desc_wrapper &diff_src_d,
            const memory_desc_wrapper &weights_d,
            const memory_desc_wrapper &diff_dst_d);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
        isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;

    inline Vmm get_ker_reg(int idx) { return Vmm(idx + 0); }
    inline Vmm get_src_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 4); }

    reg64_t reg_ddst       = rax;
    reg64_t aux_reg_ddst   = r8;
    reg64_t aux1_reg_ddst = abi_not_param1;
    reg64_t reg_kernel     = rdx;
    reg64_t aux_reg_kernel = r10;
    reg64_t aux1_reg_kernel = rbp;
    reg64_t reg_dsrc       = rsi;

    reg64_t reg_ur_str_w = r9;
    reg64_t reg_ch_blocks = rbx;

    reg64_t iter_kh = r11;
    reg64_t iter_kw = r12;
    reg64_t reg_kh  = r13;
    reg64_t reg_kw  = r14;

    inline void loop_body(int ur_ch_blocks);
    inline void load_ddst(int ur_ch_blocks, int ur_str_w);
    inline void apply_filter(int ur_ch_blocks, int ur_str_w);
    inline void store_dsrc(int ur_ch_blocks, int ur_str_w);

    void generate();
};

template <cpu_isa_t isa>
struct jit_uni_dw_conv_bwd_weights_kernel_f32 : public jit_generator {

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_dw_conv_bwd_weights_kernel_f32)

    jit_uni_dw_conv_bwd_weights_kernel_f32(jit_conv_conf_t ajcp) : jcp(ajcp) {
        this->generate();
        jit_ker = (void (*)(jit_dw_conv_call_s *)) this->getCode();
    }

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd, const memory_desc_wrapper &src_d,
            const memory_desc_wrapper &diff_weights_d,
            const memory_desc_wrapper &diff_dst_d, int nthreads);

    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp);

    static void balance(jit_conv_conf_t &jcp, int nthreads);

    jit_conv_conf_t jcp;
    void (*jit_ker)(jit_dw_conv_call_s *);

private:
    using Vmm = typename utils::conditional3<isa == sse42, Xbyak::Xmm,
            isa == avx2, Xbyak::Ymm, Xbyak::Zmm>::type;
    using reg64_t = const Xbyak::Reg64;
    const int simd_w = cpu_isa_traits<isa>::vlen / sizeof(float);
    const int reg_repeats = (isa == sse42) ? 2 : 1;

    const Xbyak::AddressFrame &vmmword
            = (isa == sse42) ? xword : (isa == avx2) ? yword : zword;

    /* XXX: offset between input and accummulators is 3, therefore, assume 'kw'
     * is no larger than 3*/
    inline Vmm get_bias_reg(int idx = 0) { return Vmm(idx); }
    inline Vmm get_output_reg(int idx) { return Vmm(idx + 1); }
    inline Vmm get_input_reg(int idx) { return Vmm(idx + 4 * reg_repeats + 1); }
    inline Vmm get_acc_reg(int idx) { return Vmm(idx + 1 * reg_repeats + 1); }
    inline Vmm get_aux_reg() { return Vmm(0); }

    reg64_t reg_tmp_input = r9;
    reg64_t reg_tmp_output = r10;
    reg64_t reg_tmp_filter = r13;
    reg64_t reg_kh_offset = rax;

    /* parameter passed by driver into kernel */
    Xbyak::Reg8 reg_exec_flags = bl;

    reg64_t reg_oh_worksize = r14;
    reg64_t reg_oh = rax;

    reg64_t iter_ow_blk = r11;

    reg64_t reg_kh = rsi;
    reg64_t reg_kh_count = rdx;

    /* Base addresses for convolution parameters. */
    reg64_t reg_input_baddr = r15;
    reg64_t reg_output_baddr = r12;
    reg64_t reg_filter_baddr = abi_not_param1;
    reg64_t reg_bias_baddr = r13;

    /* Micro-kernel JIT'ing, fusing 'kw' and 'ow_block' loops into unrolled FMAs
     */
    inline void compute_ow_step_unroll(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* JIT'ing the outer loops for the micro-kernel -> {kh, oh_block} */
    inline void compute_h_step(
            int unroll_w, int l_pad, int pad_offset, int ow_block);
    inline void compute_h_loop(
            int unroll_w, int l_pad, int pad_offset, int ow_block);

    /* Write 'width' micro-kernel JITs; depending on the padding and convolution
     * size, write a micro-kernel for the left ow-block, middle ow-block(s), and
     * right ow-block.*/
    inline void compute_ow_block_unroll();

    inline void compute_zero_filter();
    inline void load_filter();
    inline void zero_filter();
    inline void load_bias();
    inline void zero_bias();
    inline void compute_bias_step_unroll(const int unroll_w);
    inline void compute_bias_loop(const int block_size);
    inline void store_filter();
    inline void store_bias();

    void generate();
};
}
}
}

#endif
