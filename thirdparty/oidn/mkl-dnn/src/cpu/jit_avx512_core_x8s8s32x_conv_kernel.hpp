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

#ifndef CPU_JIT_AVX512_CORE_X8S8S32X_CONV_KERNEL_HPP
#define CPU_JIT_AVX512_CORE_X8S8S32X_CONV_KERNEL_HPP

#include "c_types_map.hpp"
#include "memory_tracking.hpp"

#include "jit_generator.hpp"
#include "jit_primitive_conf.hpp"
#include "jit_uni_eltwise.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

template<typename Vmm>
struct _jit_avx512_core_x8s8s32x_fwd_kernel : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(_jit_avx512_core_x8s8s32x_conv_fwd_ker_t)

    enum { STATE_FIRST_DST_LOAD = 0x1U };

    _jit_avx512_core_x8s8s32x_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr) : jcp(ajcp), attr_(attr),
            eltwise_injector_(nullptr)
    {
        if (jcp.with_eltwise)
            eltwise_injector_ = new jit_uni_eltwise_injector_f32<avx512_common>(
                this, jcp.eltwise);

        generate();
        jit_ker_ = (void (*)(jit_conv_call_s *))getCode();
    }

    ~_jit_avx512_core_x8s8s32x_fwd_kernel() {
        delete eltwise_injector_;
    }

    jit_conv_conf_t jcp;
    const primitive_attr_t &attr_;
    void (*jit_ker_)(jit_conv_call_s *);

private:
    jit_uni_eltwise_injector_f32<avx512_common> *eltwise_injector_;

    enum {
        typesize = sizeof(float),
        ker_reg_base_idx = 28,
        ker_dw_reg_base_idx = 30,
    };
    typedef enum {
        no_last_block,
        last_ic_block,
        last_sp_block,
    } ic_block_t;

    /* data regs */
    const Xbyak::Reg64 reg_ptr_scales = rax;
    const Xbyak::Reg64 reg_inp = r8;
    const Xbyak::Reg64 reg_ker = r9;
    const Xbyak::Reg64 reg_out = r10;
    const Xbyak::Reg64 aux_reg_inp = r11;
    const Xbyak::Reg64 reg_ptr_sum_scale = r11;
    const Xbyak::Reg64 aux_reg_ker = r12;
    const Xbyak::Reg64 reg_compensation = r14;
    /* counter regs */
    const Xbyak::Reg64 reg_bias_alpha = abi_not_param1;
    const Xbyak::Reg64 reg_oi = rbx;
    const Xbyak::Reg64 reg_bias = rdx;
    const Xbyak::Reg64 reg_oc_blocks = rsi;
    const Xbyak::Reg64 reg_owb = aux_reg_ker;
    const Xbyak::Reg64 reg_scratch = reg_compensation;
    const Xbyak::Reg64 reg_kj = reg_ptr_scales;
    const Xbyak::Reg64 reg_overflow = reg_ptr_scales;
    const Xbyak::Reg64 reg_icb = reg_bias;

    const Xbyak::Opmask ktail_mask = Xbyak::Opmask(2);
    const Xbyak::Opmask kblend_mask = Xbyak::Opmask(3);

    const Vmm vmm_wei = Vmm(31);
    /* used during bias section of store_output */
    const Vmm vmm_comp = Vmm(30); // only for signed input
    const Vmm vmm_bias = Vmm(31);
    /* used during post_op sum section of store_output */
    const Vmm vmm_prev_dst = Vmm(31);
    /* used during write-out section of store_output */
    const Vmm vmm_zero = Vmm(31);

    /* used in compute_ker (but set during prepare_output) */
    const Vmm vmm_shift = vmm_comp; // only for signed input
    /* used in compute_ker (but only for pre-VNNI machines) */
    const Vmm vmm_tmp = Vmm(28); // not used for depthwise
    const Vmm vmm_one = Vmm(29); // set at start of kernel, not used for depthwise.

    /* registers use only for depthwise
       groups are always blocked by 16(padded if needed),
       hence use only Zmm registers */
    const Xbyak::Zmm zmm_wei = Xbyak::Zmm(31);
    Xbyak::Zmm zmm_tmp;
    Xbyak::Zmm zmm_src;
    Xbyak::Zmm zmm_shifted_zero;
    Xbyak::Zmm zmm_permute;

    Vmm vmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < (jcp.is_depthwise
                    ? ker_dw_reg_base_idx : ker_reg_base_idx));
        return Vmm(idx);
    }
    Xbyak::Zmm zmm_out(int i_ur, int i_oc) {
        int idx = i_ur + i_oc * jcp.ur_w;
        assert(idx < (jcp.is_depthwise
                    ? ker_dw_reg_base_idx : ker_reg_base_idx));
        return Xbyak::Zmm(idx);
    }
    Vmm vmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Vmm(idx);
    }
    Xbyak::Zmm zmm_inp(int i_ic, int nb_x_blocking) {
        int idx = i_ic + nb_x_blocking * jcp.ur_w;
        assert(idx < 31);
        return Xbyak::Zmm(idx);
    }
    Vmm vmm_bias_alpha() {
        int nb_c_block = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return Vmm(nb_c_block * jcp.ur_w);
    }
    Xbyak::Xmm xmm_bias_alpha() {
        int nb_c_block = jcp.is_depthwise ? jcp.nb_ch_blocking : jcp.nb_oc_blocking;
        return Xbyak::Xmm(nb_c_block * jcp.ur_w);
    }
    int get_ow_start(int ki, int pad_l) {
        return nstl::max(0,
                utils::div_up(pad_l - ki * (jcp.dilate_w + 1), jcp.stride_w));
    }
    int get_ow_end(int ur_w, int ki, int pad_r) {
        return ur_w - nstl::max(0, utils::div_up(pad_r
                                                   - (jcp.kw - 1 - ki)
                                                           * (jcp.dilate_w + 1),
                                           jcp.stride_w));
    }

    bool maybe_eltwise(int position);
    void prepare_output(int ur_w);
    void store_output(int ur_w, bool last_oc_block_flag);
    void compute_ker_dw(
            int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag, bool h_padded);
    void compute_ker(int ur_w, int pad_l, int pad_r,
            ic_block_t last_ic_block_flag, bool h_padded = false);
    void compute_eltwise(int ur_w);
    void kh_loop(int ur_w, int pad_l, int pad_r, ic_block_t last_ic_block_flag);
    void icb_loop(
            int ur_w, int pad_l, int pad_r, bool is_last_spatial_block);
    void generate();
    void cvt2ps(data_type_t type_in, Vmm ymm_in, const Xbyak::Operand &op,
        bool mask_flag);
    const Vmm vmm_mask(const Vmm vmm_in, bool mask_flag, bool store = false);
};

struct jit_avx512_core_x8s8s32x_fwd_kernel {

    jit_avx512_core_x8s8s32x_fwd_kernel(jit_conv_conf_t ajcp,
            const primitive_attr_t &attr) :
        jit_ker(nullptr),
        zmm_kernel_(nullptr),
        ymm_kernel_(nullptr),
        xmm_kernel_(nullptr) {
            int ch_block = ajcp.is_depthwise ? ajcp.ch_block : ajcp.ic_block;
            switch (ch_block) {
                case 16:
                    zmm_kernel_ =
                        new _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Zmm>(
                                ajcp, attr);
                    jit_ker = zmm_kernel_->jit_ker_;
                    return;
                case 8:
                    ymm_kernel_ =
                        new _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Ymm>(
                                ajcp, attr);
                    jit_ker = ymm_kernel_->jit_ker_;
                    return;
                case 4:
                    xmm_kernel_ =
                        new _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Xmm>(
                                ajcp, attr);
                    jit_ker = xmm_kernel_->jit_ker_;
                    return;
                default:
                    assert(!"invalid channel blocking");
            }
    }

    ~jit_avx512_core_x8s8s32x_fwd_kernel() {
        delete xmm_kernel_;
        delete ymm_kernel_;
        delete zmm_kernel_;
    }

    static bool post_ops_ok(jit_conv_conf_t &jcp,
            const primitive_attr_t &attr);

    static status_t init_conf(jit_conv_conf_t &jcp,
            const convolution_desc_t &cd,
            memory_desc_t &src_pd,
            memory_desc_t &weights_pd,
            memory_desc_t &dst_pd,
            memory_desc_t &bias_pd,
            const primitive_attr_t &attr,
            int nthreads);
    static void init_scratchpad(memory_tracking::registrar_t &scratchpad,
            const jit_conv_conf_t &jcp, const primitive_attr_t &attr);

    void (*jit_ker)(jit_conv_call_s *);
    _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Zmm> *zmm_kernel_;
    _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Ymm> *ymm_kernel_;
    _jit_avx512_core_x8s8s32x_fwd_kernel<Xbyak::Xmm> *xmm_kernel_;
};

}
}
}

#endif
