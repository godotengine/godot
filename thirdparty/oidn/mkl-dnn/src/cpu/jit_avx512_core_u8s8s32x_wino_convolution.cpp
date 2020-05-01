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

#include <assert.h>

#include "c_types_map.hpp"
#include "memory_tracking.hpp"
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_u8s8s32x_wino_convolution.hpp"
#include "jit_generator.hpp"

#include <string.h>

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

namespace {
    // Below scales are applied to source and weights data accordingly
    // because this winograd implementation
    // transforms source which may increase values up to 4x
    // and transforms weights which may increase values up to 9/4x
    const float adj_src_scale = 1.f / 4.f;
    const float adj_wei_scale = 4.f / 9.f;
    // Winograd transforms need ic and oc to be multiples of 16
    const int load_block = 16;
}

/// SRC TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_src_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_u8s8s32x_wino_conv_src_trans_t)

    jit_conv_conf_2x3_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *src;
        const void *wino_src;
        const void *v_y_masks;
        const void *v_x_masks;
    };
    void (*ker_)(const call_params_t *);

    jit_avx512_core_u8s8s32x_wino_conv_src_trans_t(
        jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr), unsign_val_in_wino_domain(5) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }
    void generate();

    int reg_inp_ind(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return (31 - i);
    }

    Xmm vreg_inp(int i) {
        return Xmm(reg_inp_ind(i));
    }

    Zmm zmm_inp(int i) {
        return Zmm(reg_inp_ind(i));
    }

    Xmm vreg_tmp(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Xmm(15 - i);
    }
    Xmm vreg_out(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Xmm(31 - i);
    }

    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert(id < 4);
        return Opmask(3 + id);
    }

    Reg64 reg_ptr_src = r14;
    Reg64 reg_ptr_dst = r13;

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_ic_block = r8;

    int unsign_val_in_wino_domain;

    Reg64 reg_scratch_src_alpha = rdx;
    Xmm xmm_src_alpha = Xmm(0);
    Zmm zmm_src_alpha = Zmm(0);

    Reg64 reg_shift = rax;
    Xmm xmm_shift = Xmm(1);
    Xmm xmm_zero = Xmm(0);

    Reg64 reg_maskx = rbx;
    Reg64 reg_masky = rsi;
    Reg64 reg_nomask = reg_maskx;
};

void jit_avx512_core_u8s8s32x_wino_conv_src_trans_t::generate() {
    Label ic_block_label;
    Label end_label;
    Label mask_label;
    Label nomask_label;

    auto load_src = [=](bool mask) {
        for (int y = 0; y < jcp.alpha; y++) {
            if (mask)
                kmovw(y_mask, ptr[reg_ptr_v_y_masks + sizeof(uint16_t) * y]);
            for (int x = 0; x < jcp.alpha; x++) {
                Zmm zmm_i = zmm_inp(y * jcp.alpha + x);
                Xmm vreg_i = vreg_inp(y * jcp.alpha + x);
                int inp_offset = sizeof(uint8_t)
                        * ((-jcp.t_pad + y) * jcp.iw * jcp.ic
                                + (-jcp.l_pad + x) * jcp.ic);
                if (mask) {
                    kandw(r_mask, y_mask, x_mask(x));
                    vmovdqu8(vreg_i | r_mask | T_z,
                            EVEX_compress_addr(reg_aux_ptr_src, inp_offset));
                } else {
                    vmovdqu8(vreg_i,
                            EVEX_compress_addr(reg_aux_ptr_src, inp_offset));
                }
                vpmovzxbd(zmm_i, vreg_i); // to int32
                vcvtdq2ps(zmm_i, zmm_i); // to fp32
                vmulps(zmm_i, zmm_i, zmm_src_alpha); // *alpha
                vcvtps2dq(zmm_i, zmm_i); // to int32
                vpmovusdb(vreg_i, zmm_i); // to u8
            }
        }
    };

    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_ptr_dst, wino_src);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
#   undef READ_PARAM

    mov(reg_maskx, ptr[reg_ptr_v_x_masks]);
    mov(reg_masky, ptr[reg_ptr_v_y_masks]);
    test(reg_maskx, reg_maskx);
    jz(end_label, T_NEAR); // skip kernel if x mask is all 0's
    test(reg_masky, reg_masky);
    jz(end_label, T_NEAR); // skip kernel if y mask is all 0's
    and_(reg_maskx, reg_masky);
    mov(reg_nomask, reg_maskx);
    not_(reg_nomask); // zero if x and y masks are all 1's

    xor_(reg_shift, reg_shift);
    mov(reg_shift.cvt8(), (int8_t)-128);

    mov(reg_aux_ptr_src, reg_ptr_src);
    mov(reg_aux_ptr_dst, reg_ptr_dst);

    for (int i = 0; i < jcp.alpha; i++) {
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(uint16_t) * i]);
    }

    mov(reg_scratch_src_alpha, float2int(adj_src_scale));

    mov(reg_ic_block, jcp.ic / load_block);
    L(ic_block_label);
    {
        vmovq(xmm_src_alpha, reg_scratch_src_alpha);
        vbroadcastss(zmm_src_alpha, xmm_src_alpha);

        test(reg_nomask, reg_nomask);
        jz(nomask_label, T_NEAR);
        load_src(true);
        jmp(mask_label, T_NEAR);
        L(nomask_label);
        load_src(false);
        L(mask_label);

        for(int y = 0; y < 4; y++) {
            vpsubb(vreg_tmp(y*4+0), vreg_inp(y*4+0), vreg_inp(y*4+2));
            vpaddb(vreg_tmp(y*4+1), vreg_inp(y*4+1), vreg_inp(y*4+2));
            vpsubb(vreg_tmp(y*4+2), vreg_inp(y*4+2), vreg_inp(y*4+1));
            vpsubb(vreg_tmp(y*4+3), vreg_inp(y*4+1), vreg_inp(y*4+3));
        }
        for(int x = 0;x < 4; x++) {
            vpsubb(vreg_out(x+0*4), vreg_tmp(x+4*0), vreg_tmp(x+4*2));
            vpaddb(vreg_out(x+1*4), vreg_tmp(x+4*1), vreg_tmp(x+4*2));
            vpsubb(vreg_out(x+2*4), vreg_tmp(x+4*2), vreg_tmp(x+4*1));
            vpsubb(vreg_out(x+3*4), vreg_tmp(x+4*1), vreg_tmp(x+4*3));
        }

        vmovd(xmm_shift, reg_shift.cvt32());
        vpxor(xmm_zero, xmm_zero, xmm_zero);
        vpshufb(xmm_shift, xmm_shift, xmm_zero);

        for (int i = 0; i < 16; i++) {
            int out_offset = sizeof(uint8_t) * (jcp.inp_stride * i);
            if (i != unsign_val_in_wino_domain)
                vpsubb(vreg_out(i), vreg_out(i), Xmm(1));
            vmovups(EVEX_compress_addr(reg_aux_ptr_dst, out_offset), vreg_out(i));
        }

        add(reg_aux_ptr_src, sizeof(uint8_t) * load_block);
        add(reg_aux_ptr_dst, sizeof(uint8_t) * load_block);
    }
    dec(reg_ic_block);
    jnz(ic_block_label, T_NEAR);

    L(end_label);
    postamble();
}

/// DST TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t)

    jit_conv_conf_2x3_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *wino_dst;
        const void *dst;
        const void *v_y_masks;
        const void *v_x_masks;

        const void *bias;
        const void *scales;
    };
    void (*ker_)(const call_params_t *);

    jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t(
        jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    void generate();
    bool maybe_relu(int position);

    Zmm vreg_inp(int i) { // 16
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(31 - i);
    }
    Zmm vreg_stg(int id) { // 8
        const int id_reg_stg = jcp.alpha * jcp.alpha + id;
        assert(id < 8);
        return Zmm(31 - id_reg_stg);
    }
    Zmm vreg_out(int id) { // 4
        const int id_reg_out = jcp.alpha * jcp.alpha + 8 + id;
        assert(id < 4);
        return Zmm(31 - id_reg_out);
    }
    Xmm xmm_out(int id) { // 4
        const int id_reg_out = jcp.alpha * jcp.alpha + 8 + id;
        assert(id < 4);
        return Xmm(31 - id_reg_out);
    }
    Zmm vreg_tmp(int id) { // 2
        const int id_reg_tmp = jcp.alpha * jcp.alpha + 12 + id;
        assert(id < 2);
        return Zmm(31 - id_reg_tmp);
    }

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_bias = Zmm(1);
    Zmm vreg_prev_dst = Zmm(2);
    Zmm zmm_bias_alpha = Zmm(2);
    Xmm xmm_bias_alpha = Xmm(2);

    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert(id < 4);
        return Opmask(3 + id);
    }

    Reg64 reg_scratch_bias_alpha = r15;

    Reg64 reg_ptr_src = r14;
    Reg64 reg_ptr_dst = r13;

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_oc_block = r8;

    Reg64 reg_ptr_bias = rbx;
    Reg64 reg_ptr_scales = abi_not_param1;
    Reg64 reg_ptr_sum_scale = rdx;
};

bool jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::maybe_relu(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* relu before sum */
        return false
            || p.contain(eltwise, 0)
            || (jcp.dst_dt == data_type::u8 && !p.contain(sum, 0));
    } else if (position == 1) {
        /* relu after sum */
        const int sum_idx = p.contain(sum, 0)
            ? 0 : (p.contain(sum, 1) ? 1 : -1);
        if (sum_idx == -1)
            return false;

        return false
            || p.contain(eltwise, sum_idx + 1)
            || jcp.dst_dt == data_type::u8;
    }

    return false;
}

void jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::generate() {
    Label oc_block_label;

    auto loop_body = [=]() {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = (sum_idx != -1)
                ? &p.entry_[sum_idx].sum.scale
                : nullptr;
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        for(int i = 0; i < 16; i++) {
            int internal_offset = sizeof(int32_t) * jcp.out_stride * i;
            vmovups(vreg_inp(i),
                EVEX_compress_addr(reg_aux_ptr_src, internal_offset));
        }
        for(int y = 0; y < jcp.alpha; y++) {
            vpaddd(vreg_tmp(0), vreg_inp(y*4 + 0), vreg_inp(y*4 + 1));
            vpaddd(vreg_stg(y*2), vreg_tmp(0), vreg_inp(y*4 + 2));

            vpsubd(vreg_tmp(1), vreg_inp(y*4 + 1), vreg_inp(y*4 + 2));
            vpsubd(vreg_stg(y*2+1), vreg_tmp(1), vreg_inp(y*4 + 3));
        }
        for(int x = 0; x < jcp.m; x++) {
            vpaddd(vreg_tmp(0), vreg_stg(x), vreg_stg(x+2*1));
            vpaddd(vreg_out(x), vreg_tmp(0), vreg_stg(x+2*2));

            vpsubd(vreg_tmp(1), vreg_stg(x+2*1), vreg_stg(x+2*2));
            vpsubd(vreg_out(x+2), vreg_tmp(1), vreg_stg(x+2*3));
        }


        if (jcp.with_bias) {
            vmovq(xmm_bias_alpha, reg_scratch_bias_alpha);
            vbroadcastss(zmm_bias_alpha, xmm_bias_alpha);

            auto bias_addr = ptr [ reg_ptr_bias ];
            switch (jcp.bia_dt) {
            case data_type::f32:
            case data_type::s32: vmovups(vreg_bias, bias_addr); break;
            case data_type::s8: vpmovsxbd(vreg_bias, bias_addr); break;
            case data_type::u8: vpmovzxbd(vreg_bias, bias_addr); break;
            default: assert(!"unsupported dst data type");
            }
            if (jcp.bia_dt != data_type::f32)
                vcvtdq2ps(vreg_bias, vreg_bias);
            vmulps(vreg_bias, vreg_bias, zmm_bias_alpha); // *alpha
        }
        for(int y = 0; y < jcp.m; y++) {
            kmovw(y_mask, ptr[ reg_ptr_v_y_masks + sizeof(uint16_t) * y ]);
            for(int x = 0; x < jcp.m; x++) {
                kandw(r_mask, y_mask, x_mask(x));

                int i = y * jcp.m + x;
                int offset = jcp.typesize_out *
                    (y * jcp.ow * jcp.oc + x * jcp.oc);
                Address addr = EVEX_compress_addr(reg_aux_ptr_dst, offset);

                Zmm zmm = vreg_out(i);
                Xmm xmm = xmm_out(i);
                vcvtdq2ps(zmm, zmm);
                if (jcp.with_bias)
                    vaddps(zmm, zmm, vreg_bias);
                vmulps(zmm, zmm, ptr [reg_ptr_scales]);
                if (maybe_relu(0))
                    vmaxps(zmm, vreg_zero, zmm);
                if (p_sum_scale) { // post_op: sum
                    vpxord(vreg_prev_dst, vreg_prev_dst, vreg_prev_dst);
                    switch (jcp.dst_dt) {
                    case data_type::f32:
                    case data_type::s32:
                        vmovups(vreg_prev_dst | r_mask, addr); break;
                    case data_type::s8:
                        vpmovsxbd(vreg_prev_dst | r_mask, addr); break;
                    case data_type::u8:
                        vpmovzxbd(vreg_prev_dst | r_mask, addr); break;
                    default: assert(!"unknown dst_dt");
                    }
                    if (jcp.dst_dt != data_type::f32)
                        vcvtdq2ps(vreg_prev_dst, vreg_prev_dst);
                    if (*p_sum_scale == 1.f)
                        vaddps(zmm, vreg_prev_dst);
                    else
                        vfmadd231ps(zmm, vreg_prev_dst,
                            zword_b[reg_ptr_sum_scale]);
                }
                if (maybe_relu(1))
                    vmaxps(zmm, vreg_zero, zmm);
                if (jcp.dst_dt != data_type::f32)
                    vcvtps2dq(zmm, zmm);
                switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32:
                    vmovups(addr,  zmm | r_mask); break;
                case data_type::s8:
                    vpmovsdb(xmm, zmm); vmovups(addr, xmm | r_mask); break;
                case data_type::u8:
                    vpmovusdb(xmm, zmm); vmovups(addr, xmm | r_mask); break;
                default: assert(!"unknown dst_dt");
                }
            }
        }
    };

    preamble();

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, wino_dst);
    READ_PARAM(reg_ptr_dst, dst);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
    READ_PARAM(reg_ptr_bias, bias);
    READ_PARAM(reg_ptr_scales, scales);
#   undef READ_PARAM

    if (jcp.with_bias)
        mov(reg_scratch_bias_alpha, float2int(adj_src_scale * adj_wei_scale));

    mov(reg_aux_ptr_src, reg_ptr_src);
    mov(reg_aux_ptr_dst, reg_ptr_dst);

    vpxord(vreg_zero, vreg_zero, vreg_zero);

    for (int i = 0; i < jcp.m; i++)
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(uint16_t) * i]);

    int oc_blocks = jcp.oc / load_block;
    mov(reg_oc_block, oc_blocks);
    L(oc_block_label); {
        loop_body();
        add(reg_aux_ptr_src, sizeof(int32_t) * load_block);
        add(reg_aux_ptr_dst, jcp.typesize_out * load_block);

        add(reg_ptr_scales, jcp.is_oc_scale * sizeof(float) * load_block);
        add(reg_ptr_bias, sizeof(jcp.typesize_bia) * load_block);
    }
    dec(reg_oc_block);
    jnz(oc_block_label, T_NEAR);

    postamble();

}

/// GEMM kernel ////////////////////////////////////////////////////////////////
struct jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t)
    jit_conv_conf_2x3_wino_t jcp;
    const primitive_attr_t &attr_;

    struct call_params_t {
        const void *src;
        const void *dst;
        const void *wei;
        const void *dst_b;
    };
    void (*ker_)(const call_params_t *);

    void generate();
    static bool post_ops_ok(jit_conv_conf_2x3_wino_t &jcp,
                            const primitive_attr_t &attr);

    jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t(
        jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr)
    {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    static status_t init_conf(
            jit_conv_conf_2x3_wino_t &jcp, const convolution_desc_t &cd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            const primitive_attr_t &attr);

    Zmm vreg_out(int n, int m) {
        const int id_reg_out = n * jcp.m_block + m;
        assert(id_reg_out < jcp.n2_block * jcp.m_block);
        return Zmm(31 - id_reg_out);
    }
    Zmm vreg_wei(int i) {
        assert(31 - jcp.n2_block * jcp.m_block - i
                > (jcp.ver == ver_vnni ? 0 : 2));
        return Zmm(31 - jcp.n2_block * jcp.m_block - i);
    }

    Zmm vreg_src = Zmm(0);
    Zmm vreg_one = Zmm(1);
    Zmm vreg_tmp = Zmm(2);

    Reg64 reg_ptr_src = r15;

    Reg64 reg_aux_dst_b = r13;
    Reg64 reg_aux_dst = r12;
    Reg64 reg_aux_dst2 = r11;
    Reg64 reg_aux_wei = r10;
    Reg64 reg_aux_wei2 = r9;
    Reg64 reg_aux_src = r8;
    Reg64 reg_aux_src2 = rax;
    Reg64 reg_mb = rbx;
    Reg64 reg_nnb = abi_not_param1;
    Reg64 reg_scratch = rdx;
    Reg64 reg_K = rsi;
};

bool jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::post_ops_ok(
        jit_conv_conf_2x3_wino_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_relu = [&](int idx) { return p.entry_[idx].is_relu(); };

    switch (p.len_) {
    case 0: return true;
    case 1: return is_relu(0) || p.contain(sum, 0);
    case 2: return (p.contain(sum, 0) && is_relu(1)) ||
                       (p.contain(sum, 1) && is_relu(0));
    case 3: return is_relu(0) && p.contain(sum, 1) && is_relu(2);
    default: return false;
    }

    return false;
}

void jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::generate() {
    Label nnb_loop_label, K_loop_label, mb_loop_label;

    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(vreg_tmp, vreg_src, vreg_wei);
            vpmaddwd(vreg_tmp, vreg_tmp, vreg_one);
            vpaddd(vreg_acc, vreg_acc, vreg_tmp);
        }
    };

    preamble();
#   define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_aux_dst, dst);
    READ_PARAM(reg_aux_wei, wei);
    READ_PARAM(reg_aux_dst_b, dst_b);
#   undef READ_PARAM

    if (jcp.ver != ver_vnni) {
        xor_(reg_scratch, reg_scratch);
        Reg16 _t = reg_scratch.cvt16();
        mov(_t, 0x1);
        vpbroadcastw(vreg_one, _t);
    }

    if (!jcp.small_mb) {
        mov(reg_nnb, jcp.n_chunks);
        L(nnb_loop_label);
    }
    mov(reg_aux_dst2, reg_aux_dst);
    mov(reg_aux_src, reg_ptr_src);
    mov(reg_mb, jcp.M / jcp.m_block);
    L(mb_loop_label);
    {
        for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
            for (int m = 0; m < jcp.m_block; m++) {
                int offset = jcp.typesize_acc * nb2 * jcp.n_block;
                vmovups(vreg_out(nb2, m),
                        EVEX_compress_addr(reg_aux_dst_b, offset));
            }
        }
        mov(reg_aux_src2, reg_aux_src);
        mov(reg_aux_wei2, reg_aux_wei);
        mov(reg_K, jcp.k_chunks);
        L(K_loop_label);
        {
            for (int k = 0; k < jcp.k2_block; k += 4) {
                for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                    int wei_offset
                            = jcp.typesize_in * (nb2 * jcp.n_block * jcp.K);
                    vmovups(vreg_wei(nb2),
                            EVEX_compress_addr(reg_aux_wei2, wei_offset));
                }
                for (int m = 0; m < jcp.m_block; m++) {
                    int inp_offset = jcp.typesize_in * m * jcp.K;
                    vpbroadcastd(vreg_src,
                            EVEX_compress_addr(reg_aux_src2, inp_offset));
                    for (int nb2 = 0; nb2 < jcp.n2_block; nb2++)
                        compute(vreg_out(nb2, m), vreg_wei(nb2), vreg_src);
                }
                add(reg_aux_src2, jcp.typesize_in * 4);
                add(reg_aux_wei2, jcp.typesize_in * 4 * jcp.n_block);
            }
        }
        dec(reg_K);
        jnz(K_loop_label, T_NEAR);

        for (int m = 0; m < jcp.m_block; m++) {
            for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                int offset = jcp.typesize_acc * (m * jcp.N + nb2 * jcp.n_block);
                vmovups(EVEX_compress_addr(reg_aux_dst2, offset),
                        vreg_out(nb2, m));
            }
        }
        add(reg_aux_src, jcp.typesize_in * jcp.m_block * jcp.K);
        add(reg_aux_dst2, jcp.typesize_acc * jcp.m_block * jcp.N);
    }
    dec(reg_mb);
    jnz(mb_loop_label, T_NEAR);

    if (!jcp.small_mb) {
        add(reg_aux_dst, jcp.typesize_acc * jcp.n2_block * jcp.n_block);
        add(reg_aux_dst_b, jcp.typesize_acc * jcp.n2_block * jcp.n_block);
        add(reg_aux_wei, jcp.typesize_in * jcp.n2_block * jcp.n_block * jcp.K);

        dec(reg_nnb);
        jnz(nnb_loop_label, T_NEAR);
    }

    postamble();
}
namespace {
bool is_winograd_faster_than_direct(const jit_conv_conf_2x3_wino_t &jcp) {
    if (jcp.ver == ver_vnni) {
        return (jcp.mb <= mkldnn_get_max_threads()
            && (jcp.mb > 4
                && jcp.ic > 64
                && !(jcp.oc > 128 && jcp.ih < 14)))
            || jcp.mb > mkldnn_get_max_threads();
    }
    return true;
}
}

status_t jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t
::init_conf(jit_conv_conf_2x3_wino_t &jcp,
            const convolution_desc_t &cd, memory_desc_t &src_md,
            memory_desc_t &wei_md, memory_desc_t &dst_md,
            memory_desc_t &bias_md, const primitive_attr_t &attr) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper wei_d(&wei_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = wei_d.ndims() == src_d.ndims() + 1;

    jcp.nthr = mkldnn_get_max_threads();

    jcp.ngroups = with_groups ? wei_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = wei_d.dims()[with_groups + 2];
    jcp.kw = wei_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.b_pad = cd.padding[1][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.r_pad = cd.padding[1][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.dilate_h = cd.dilates[0];
    jcp.dilate_w = cd.dilates[1];

    jcp.ver = ver_avx512_core;
    if (!(mayiuse(avx512_core) &&
            src_d.data_type() == data_type::u8
         && wei_d.data_type() == data_type::s8
         && one_of(dst_d.data_type(), data_type::f32, data_type::s32,
            data_type::s8, data_type::u8)))
        return status::unimplemented;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;

    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
               is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;

    // block sizes needed for GEMM kernel
    jcp.ic_block = 4;
    jcp.oc_block = 16;

    bool ok = true
        && jcp.ngroups == 1
        && jcp.oc % load_block == 0 && jcp.ic % load_block == 0
        && jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0
        && everyone_is(3, jcp.kh, jcp.kw)
        && everyone_is(1, jcp.stride_h, jcp.stride_w)
        && everyone_is(0, jcp.dilate_h, jcp.dilate_w)
        && jcp.t_pad == jcp.b_pad && jcp.l_pad == jcp.r_pad
        && one_of(jcp.t_pad, 0, 1)
        && one_of(jcp.l_pad, 0, 1);
    if (!ok) return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_acc = sizeof(int32_t);
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    jcp.m = 2;
    jcp.r = 3;
    jcp.alpha = jcp.m + jcp.r - 1;

    int aa = jcp.alpha * jcp.alpha;
    int L1_cap = get_cache_size(1, true);
    int L2_cap = get_cache_size(2, true);
    // need 1 extra reg for bcast, and 2 tmp regs for non-vnni
    int free_regs = jcp.ver == ver_vnni ? 31 : 29;

    auto get_thr_eff = [&](int small_mb, int ix, int iy, int n2_b) {
        float thr_eff;
        float Z = (float)jcp.ic + jcp.oc;
        float Y = (float)jcp.ic * jcp.oc;
        if (small_mb == 0) { // outer par
            int nblocks = jcp.mb * div_up(jcp.oh, iy) * div_up(jcp.ow, ix);
            thr_eff = (float)nblocks / rnd_up(nblocks, jcp.nthr);
        } else { // inner par
            int tranw = iy * ix / jcp.alpha;
            int gemmw = aa * (jcp.nb_oc / n2_b);
            int tranw_r = rnd_up(tranw, jcp.nthr);
            int gemmw_r = rnd_up(gemmw, jcp.nthr);
            thr_eff = (Z * tranw / tranw_r + Y * gemmw / gemmw_r) / (Z + Y);
        }
        return thr_eff;
    };

    auto get_mem_eff = [&](int small_mb, int ix, int iy, int n2_b) {
        float mem_eff, req_mem;
        int M = ix * iy / jcp.alpha;
        if (small_mb == 0) { // outer parallelization strategy
            // memory for wino transforms (other memory has poor reuse)
            req_mem = (float)aa * M * (jcp.ic + jcp.typesize_acc * jcp.oc);
            mem_eff = req_mem < L1_cap ? 1.f : req_mem < L2_cap ? 0.5f : 0.f;
        } else { // inner parallelization strategy
            // memory used during gemm
            int N = jcp.oc_block * n2_b;
            req_mem = (float)jcp.ic * (M + N) + jcp.typesize_acc * M * N;
            mem_eff = nstl::min(1.f, L2_cap / req_mem);
            // memory used during wino transforms
            int M_per_thr = div_up(M, jcp.nthr);
            req_mem = (float)aa * M_per_thr
                    * (jcp.ic + jcp.typesize_acc * jcp.oc);
            if (req_mem > L2_cap)
                mem_eff = 0.1f;
        }
        return mem_eff;
    };

    auto get_tot_eff = [&](int small_mb, float thr_eff, float work_eff,
            float mem_eff, float reg_eff) {
        // these coefficients are chosen empirically
        float mem_fac = 0.1f, reg_fac = 0.2f;
        // normalized overhead relative to memory and register components
        float tot_eff = 1.f + mem_fac * mem_eff + reg_fac * reg_eff;
        // thread and work components affect all others
        tot_eff *= thr_eff * work_eff;
        return tot_eff;
    };

    auto find_m_n2_blocks = [&](bool small_mb, int ix, int iy, float work_eff,
            int &m_block, int &n2_block, float &tot_eff) {
        int M = (ix * iy) / jcp.alpha;
        int max_m_block = nstl::min(M, free_regs);
        int max_n2_block = nstl::min(jcp.nb_oc, free_regs);
        tot_eff = 0.f;
        for (int im = max_m_block; im > 0; im--) {
            if (M % im)
                continue;
            for (int in2 = max_n2_block; in2 > 0; in2--) {
                int used_regs = (im + 1) * in2;
                float mem_eff = get_mem_eff(small_mb, ix, iy, in2);
                float reg_eff = (float)(im * in2) / (im + in2);
                float thr_eff = get_thr_eff(small_mb, ix, iy, in2);
                float cur_tot_eff = get_tot_eff(
                        small_mb, thr_eff, work_eff, mem_eff, reg_eff);
                if (jcp.nb_oc % in2 || used_regs > free_regs
                        || cur_tot_eff <= tot_eff)
                    continue;
                tot_eff = cur_tot_eff;
                m_block = im;
                n2_block = in2;
            }
        }
    };

    /* Selecting xb and yb blocking */
    int min_yb = jcp.m;
    int min_xb = jcp.m;
    int max_yb = nstl::max(min_yb, rnd_up(jcp.oh, 2));
    int max_xb = nstl::max(min_xb, rnd_up(jcp.ow, 2));
    float best_eff = 0.f;
    for (int ix = min_xb; ix <= max_xb; ix += 2) {
        assert(rnd_up(jcp.ow, ix) >= jcp.iw - 2);
        for (int iy = max_yb; iy >= min_yb; iy -= 2) {
            assert(rnd_up(jcp.oh, iy) >= jcp.ih - 2);

            int m_b[2];
            int n2_b[2];
            bool small_mb;
            float inner_eff, outer_eff, work_eff;

            int tiled_area = rnd_up(jcp.oh, iy) * rnd_up(jcp.ow, ix);
            work_eff = (float)jcp.oh * jcp.ow / tiled_area;
            if (best_eff > 0.f && work_eff < 4.f / 9.f)
                continue; // no gain from Winograd transformation

            /* outer parallelization */
            find_m_n2_blocks(0, ix, iy, work_eff, m_b[0], n2_b[0], outer_eff);

            /* inner parallelization */
            find_m_n2_blocks(1, ix, iy, work_eff, m_b[1], n2_b[1], inner_eff);

            small_mb = inner_eff > outer_eff;
            float eff = small_mb ? inner_eff : outer_eff;
            if (eff > best_eff) {
                best_eff = eff;
                jcp.yb = iy;
                jcp.xb = ix;
                jcp.m_block = m_b[small_mb];
                jcp.n2_block = n2_b[small_mb];
                jcp.small_mb = small_mb;
            }
        }
    }

    assert((jcp.m_block + 1) * jcp.n2_block <= free_regs);
    assert(jcp.xb % 2 == 0 && jcp.yb % 2 == 0);

    jcp.mb_block = 1;
    if (jcp.small_mb) {
        // For small mb harness, set mb_block as large as possible subject to
        // the constraint that winograd activations fit into available L3 cache
        int L3_cap = get_cache_size(3, true);
        int M = jcp.xb * jcp.yb / 4;
        int wino_src_size = 16 * M * jcp.ic * jcp.typesize_in;
        int wino_dst_size = 16 * M * jcp.oc * jcp.typesize_acc;
        int max_mb_block = nstl::min(
                jcp.mb, jcp.nthr * L3_cap / (wino_src_size + wino_dst_size));
        for (int i = max_mb_block; i > 1; i--) {
            if (jcp.mb % i == 0) {
                jcp.mb_block = i;
                break;
            }
        }
    }
    jcp.nb_mb = jcp.mb / jcp.mb_block;

    jcp.M = jcp.mb_block * jcp.xb * jcp.yb / 4;
    jcp.N = jcp.oc;
    jcp.K = jcp.ic;

    jcp.inp_stride = jcp.M * jcp.ic;
    jcp.out_stride = jcp.M * jcp.oc;
    jcp.wei_stride = jcp.ic * jcp.oc;
    jcp.bia_stride = jcp.oc;

    jcp.n_block = jcp.oc_block;
    jcp.k_block = jcp.ic_block;

    jcp.n_chunks = (jcp.N / jcp.n_block) / jcp.n2_block;

    // We need jcp.k2_block to be a multiple of jcp.k_block = jcp.ic_block = 4
    // and jcp.K = jcp.ic to be a multiple of jcp.k2_block. Since jcp.ic is
    // a multiple of load_block = 16, we just use that for now.
    jcp.k2_block = load_block;
    jcp.k_chunks = jcp.K / jcp.k2_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    /* re-create weights primitive descriptor
                                    and set weights wino_blocking */
    memory_desc_t expect_wei_md = wei_md;

    expect_wei_md.format_kind = format_kind::wino;
    expect_wei_md.data_type = data_type::s8;
    mkldnn_wino_desc_t &wd = expect_wei_md.format_desc.wino_desc;
    wd.wino_format = mkldnn_wino_wei_aaOIoi;
    wd.r = jcp.r;
    wd.alpha = jcp.alpha;
    wd.ic = jcp.ic;
    wd.oc = jcp.oc;
    wd.ic_block = jcp.ic_block;
    wd.oc_block = jcp.oc_block;
    wd.oc2_block = jcp.n2_block;
    wd.ic2_block = 1;
    wd.adj_scale = adj_wei_scale;

    size_t max_size = types::data_type_size(data_type::s8) *
                        jcp.alpha * jcp.alpha * jcp.ic * jcp.oc;
    max_size += types::data_type_size(data_type::s32) *
                                jcp.alpha * jcp.alpha * jcp.oc;
    wd.size = max_size;

    if (wei_md.format_kind == format_kind::any)
        wei_md = expect_wei_md;
    if (wei_md != expect_wei_md)
        return status::unimplemented;

    const int tilesize = jcp.alpha * jcp.alpha;
    const int numtiles = jcp.M;
    const int alltiles = numtiles * tilesize;

    jcp.size_wino_src
        = utils::rnd_up(jcp.typesize_in * alltiles * jcp.ic, PAGE_4K)
        / jcp.typesize_in;
    jcp.size_wino_wei = tilesize * jcp.oc * jcp.ic;
    jcp.size_wino_dst = alltiles * jcp.oc;

    return status::success;
}
////////////////////////////////////////////////////////////////////////////////

template <data_type_t dst_data_type>
status_t jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
        pd_t::jit_conf() {
    return jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::init_conf(
            jcp_, *this->desc(), this->src_md_, this->weights_md_,
            this->dst_md_,this->bias_md_, *this->attr());
}

template <data_type_t dst_data_type>
void jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::pd_t::
init_scratchpad() {
    auto scratchpad = this->scratchpad_registry().registrar();

    int nthr_multiplier = jcp_.small_mb ? 1 : jcp_.nthr;
    scratchpad.book(key_wino_V,
            sizeof(src_data_t) * jcp_.size_wino_src * nthr_multiplier, PAGE_4K);
    scratchpad.book(key_wino_M,
            sizeof(acc_data_t) * jcp_.size_wino_dst * nthr_multiplier, PAGE_4K);

    dim_t scale_count = attr()->output_scales_.count_;
    scratchpad.book(key_conv_adjusted_scales,
            sizeof(float) * nstl::max<dim_t>(scale_count, 16));
}

template <data_type_t dst_data_type>
jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
        jit_avx512_core_u8s8s32x_wino_convolution_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd)
{
    kernel_ = new jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t(
            pd()->jcp_, *pd()->attr());
    src_trans_ = new jit_avx512_core_u8s8s32x_wino_conv_src_trans_t(
            pd()->jcp_, *pd()->attr());
    dst_trans_ = new jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t(
            pd()->jcp_, *pd()->attr());
}

template <data_type_t dst_data_type>
jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
        ~jit_avx512_core_u8s8s32x_wino_convolution_fwd_t() {
    delete kernel_;
    delete src_trans_;
    delete dst_trans_;
}

template <data_type_t dst_data_type>
const float *jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
adjust_oscales(const memory_tracking::grantor_t &scratchpad) const {
    const float *oscales = pd()->attr()->output_scales_.scales_;
    auto loc_scales = scratchpad.template get<float>(key_conv_adjusted_scales);
    size_t count = pd()->attr()->output_scales_.count_;
    float factor = 1.f / (adj_src_scale * adj_wei_scale);
    if (count == 1)
        utils::array_set(loc_scales, oscales[0] * factor, 16);
    else
        for (size_t c = 0; c < count; c++) loc_scales[c] = oscales[c] * factor;
    return loc_scales;
}

template <data_type_t dst_data_type>
void jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
execute_forward(const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const auto &jcp = kernel_->jcp;
    if (jcp.small_mb)
        execute_forward_small_mb(src, weights, bias, dst, this->scratchpad(ctx));
    else
        execute_forward_mbN(src, weights, bias, dst, this->scratchpad(ctx));
}

template <data_type_t dst_data_type>
void jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
execute_forward_mbN(const src_data_t *src, const wei_data_t *wei,
        const char *bia, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const float *oscales = adjust_oscales(scratchpad);

    auto dst_bias = (const acc_data_t *)(wei + jcp.size_wino_wei);
    auto wino_src_base = scratchpad.template get<src_data_t>(key_wino_V);
    auto wino_dst_base = scratchpad.template get<acc_data_t>(key_wino_M);

    parallel_nd(jcp.mb, div_up(jcp.oh, jcp.yb), div_up(jcp.ow, jcp.xb),
            [&](int mb, int tile_y_b, int tile_x_b) {

        int tile_y = tile_y_b * jcp.yb;
        int tile_x = tile_x_b * jcp.xb;

        int ithr = mkldnn_get_thread_num();
        auto wino_src = wino_src_base + jcp.size_wino_src * ithr;
        auto wino_dst = wino_dst_base + jcp.size_wino_dst * ithr;

        auto src_trans_p =
            jit_avx512_core_u8s8s32x_wino_conv_src_trans_t::call_params_t();
        auto dst_trans_p =
            jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::call_params_t();
        auto gemm_p =
            jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::call_params_t();

        /* transformation of input tensor to winograd domain */
        for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                uint16_t v_y_masks[4], v_x_masks[4];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

                int v_ys = nstl::max(0, jcp.t_pad - y);
                int v_ye = nstl::min(jcp.alpha,
                        nstl::max(0, jcp.ih + jcp.t_pad - y));

                int v_xs = nstl::max(0, jcp.l_pad - x);
                int v_xe = nstl::min(jcp.alpha,
                        nstl::max(0, jcp.iw + jcp.l_pad - x));

#pragma unroll(4)
                for (int i = 0; i < jcp.alpha; i++) {
                    v_y_masks[i] = uint16_t(i < v_ys || i >= v_ye ? 0 : 0xffff);
                    v_x_masks[i] = uint16_t(i < v_xs || i >= v_xe ? 0 : 0xffff);
                }
                auto local_s = src
                        + mb * jcp.ih * jcp.iw * jcp.ic
                        + y * jcp.iw * jcp.ic + x * jcp.ic;
                auto local_w = wino_src + m * jcp.ic;

                src_trans_p.src = local_s;
                src_trans_p.wino_src = local_w;
                src_trans_p.v_y_masks = v_y_masks;
                src_trans_p.v_x_masks = v_x_masks;

                src_trans_->ker_(&src_trans_p);
            }
        }
        /* gemms */
        for (int tile_ij = 0; tile_ij < 16; tile_ij++) {
            // start threads at different GEMMs to help bring weights into LLC
            int offset = (tile_ij + ithr) % 16;
            gemm_p.src = wino_src + jcp.inp_stride * offset;
            gemm_p.dst = wino_dst + jcp.out_stride * offset;
            gemm_p.wei = wei + jcp.wei_stride * offset;
            gemm_p.dst_b = dst_bias + jcp.bia_stride * offset;

            kernel_->ker_(&gemm_p);
        }

        /* transformation from winograd domain to output tensor */
        for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb; x_in_block += 2) {
                uint16_t v_y_masks[2], v_x_masks[2];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

#pragma unroll(2)
                for (int i = 0; i < jcp.m; i++) {
                    v_x_masks[i] = uint16_t(x + i < jcp.ow ? 0xffff : 0);
                    v_y_masks[i] = uint16_t(y + i < jcp.oh ? 0xffff : 0);
                }
                auto local_d = dst
                        + mb * jcp.oh * jcp.ow * jcp.oc
                        + y * jcp.ow * jcp.oc + x * jcp.oc;
                auto local_w = wino_dst + m * jcp.oc;

                auto scales = oscales;
                dst_trans_p.dst = local_d;
                dst_trans_p.wino_dst = local_w;
                dst_trans_p.v_y_masks = v_y_masks;
                dst_trans_p.v_x_masks = v_x_masks;

                dst_trans_p.scales = scales;
                dst_trans_p.bias = bia;

                dst_trans_->ker_(&dst_trans_p);
            }
        }
    });
}

template <data_type_t dst_data_type>
void jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<dst_data_type>::
execute_forward_small_mb(const src_data_t *src, const wei_data_t *wei,
        const char *bia, dst_data_t *dst,
        const memory_tracking::grantor_t &scratchpad) const {
    const auto &jcp = kernel_->jcp;
    const float *oscales = adjust_oscales(scratchpad);

    auto dst_bias = (const acc_data_t *)(wei + jcp.size_wino_wei);
    auto wino_src = scratchpad.template get<src_data_t>(key_wino_V);
    auto wino_dst = scratchpad.template get<acc_data_t>(key_wino_M);

    for (int mbb = 0; mbb < jcp.nb_mb; mbb++) {
    for (int tile_y = 0; tile_y < jcp.oh; tile_y += jcp.yb) {
    for (int tile_x = 0; tile_x < jcp.ow; tile_x += jcp.xb) {
        /* transformation of input tensor to winograd domain */
        parallel_nd(div_up(jcp.yb, 2), div_up(jcp.xb, 2), jcp.mb_block,
            [&](int y_in_block_b, int x_in_block_b, int mb) {
            int y_in_block = y_in_block_b * 2;
            int x_in_block = x_in_block_b * 2;

            auto src_trans_p =
                jit_avx512_core_u8s8s32x_wino_conv_src_trans_t::call_params_t();

            uint16_t v_y_masks[4], v_x_masks[4];

            int y = y_in_block + tile_y;
            int x = x_in_block + tile_x;
            int m = (mb * (jcp.yb / 2) + (y_in_block / 2)) * (jcp.xb / 2)
                    + (x_in_block / 2);

            int v_ys = nstl::max(0, jcp.t_pad - y);
            int v_ye = nstl::min(
                    jcp.alpha, nstl::max(0, jcp.ih + jcp.t_pad - y));

            int v_xs = nstl::max(0, jcp.l_pad - x);
            int v_xe = nstl::min(
                    jcp.alpha, nstl::max(0, jcp.iw + jcp.l_pad - x));

#pragma unroll(4)
            for (int i = 0; i < jcp.alpha; i++) {
                v_y_masks[i] = uint16_t(i < v_ys || i >= v_ye ? 0 : 0xffff);
                v_x_masks[i] = uint16_t(i < v_xs || i >= v_xe ? 0 : 0xffff);
            }
            auto local_s = src
                    + (mbb * jcp.mb_block + mb) * jcp.ih * jcp.iw * jcp.ic
                    + y * jcp.iw * jcp.ic + x * jcp.ic;
            auto local_w = wino_src + m * jcp.ic;

            src_trans_p.src = local_s;
            src_trans_p.wino_src = local_w;
            src_trans_p.v_y_masks = v_y_masks;
            src_trans_p.v_x_masks = v_x_masks;

            src_trans_->ker_(&src_trans_p);
        });

        /* gemms */
        parallel_nd(16, jcp.n_chunks, [&](int tile_ij, int nnb) {
            auto gemm_p = jit_avx512_core_u8s8s32x_wino_conv_fwd_ker_t::
                    call_params_t();

            gemm_p.src = wino_src + jcp.inp_stride * tile_ij;
            gemm_p.dst = wino_dst + jcp.out_stride * tile_ij
                    + nnb * jcp.n2_block * jcp.n_block;
            gemm_p.wei = wei + jcp.wei_stride * tile_ij
                    + nnb * jcp.n2_block * jcp.n_block * jcp.K;
            gemm_p.dst_b = dst_bias + jcp.bia_stride * tile_ij
                    + nnb * jcp.n2_block * jcp.n_block;

            kernel_->ker_(&gemm_p);
        });

        /* transformation from winograd domain to output tensor */
        parallel_nd(div_up(jcp.yb, 2), div_up(jcp.xb, 2), jcp.mb_block,
            [&](int y_in_block_b, int x_in_block_b, int mb) {
            int y_in_block = y_in_block_b * 2;
            int x_in_block = x_in_block_b * 2;

            auto dst_trans_p =
                jit_avx512_core_u8s8s32x_wino_conv_dst_trans_t::call_params_t();

            uint16_t v_y_masks[2], v_x_masks[2];

            int y = y_in_block + tile_y;
            int x = x_in_block + tile_x;
            int m = (mb * (jcp.yb / 2) + (y_in_block / 2)) * (jcp.xb / 2)
                    + (x_in_block / 2);

#pragma unroll(2)
            for (int i = 0; i < jcp.m; i++) {
                v_x_masks[i] = uint16_t(x + i < jcp.ow ? 0xffff : 0);
                v_y_masks[i] = uint16_t(y + i < jcp.oh ? 0xffff : 0);
            }
            auto local_d = dst
                    + (mbb * jcp.mb_block + mb) * jcp.oh * jcp.ow * jcp.oc
                    + y * jcp.ow * jcp.oc + x * jcp.oc;
            auto local_w = wino_dst + m * jcp.oc;

            auto scales = oscales;
            dst_trans_p.dst = local_d;
            dst_trans_p.wino_dst = local_w;
            dst_trans_p.v_y_masks = v_y_masks;
            dst_trans_p.v_x_masks = v_x_masks;

            dst_trans_p.scales = scales;
            dst_trans_p.bias = bia;

            dst_trans_->ker_(&dst_trans_p);
        });
    }}}
}

template struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<data_type::s8>;
template struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<data_type::u8>;
template struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<data_type::s32>;
template struct jit_avx512_core_u8s8s32x_wino_convolution_fwd_t<data_type::f32>;

} // namespace cpu
} // namespace impl
} // namespace mkldnn
