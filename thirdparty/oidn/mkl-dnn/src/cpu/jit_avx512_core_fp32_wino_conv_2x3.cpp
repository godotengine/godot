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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_core_fp32_wino_conv_2x3.hpp"
#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::format_kind;
using namespace mkldnn::impl::memory_tracking::names;
using namespace mkldnn::impl::utils;
using namespace Xbyak;

/// SRC TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_fp32_wino_conv_2x3_src_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_fp32_wino_conv_2x3_src_trans_t)

    jit_conv_conf_2x3_wino_t jcp;

    struct call_params_t {
        const void *src;
        const void *wino_src;
        const void *v_y_masks;
        const void *v_x_masks;
    };
    void (*ker_)(const call_params_t *);

    jit_avx512_core_fp32_wino_conv_2x3_src_trans_t(
        jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp) {
        generate();
        ker_ =
            reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(getCode()));
    }

    void generate();

    Zmm vreg_inp(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(31 - i);
    }

    Zmm vreg_tmp(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(15 - i);
    }

    Zmm vreg_out(int i) {
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(31 - i);
    }

    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert (id < 4);
        return Opmask(3 + id);
    }

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_ic_block = r8;

};

void jit_avx512_core_fp32_wino_conv_2x3_src_trans_t::generate() {
    Label ic_block_label;

    const int load_block = 16;
    int out_offset = 0, inp_offset = 0;
    preamble();

#define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_aux_ptr_src, src);
    READ_PARAM(reg_aux_ptr_dst, wino_src);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
#undef READ_PARAM

    for (int i = 0; i < jcp.alpha; i++) {
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(int16_t) * i]);
    }
    mov(reg_ic_block, jcp.ic / load_block);
    L(ic_block_label);
    {
        for (int y = 0; y < jcp.alpha; y++) {
            kmovw(y_mask, ptr[reg_ptr_v_y_masks + sizeof(int16_t) * y]);
            for (int x = 0; x < jcp.alpha; x++) {
                Zmm zmm = vreg_inp(y * jcp.alpha + x);

                vxorps(zmm, zmm, zmm);
                kandw(r_mask, y_mask, x_mask(x));
                inp_offset = sizeof(float)
                        * ((-jcp.t_pad + y) * jcp.iw * load_block
                                  + (-jcp.l_pad + x) * load_block);
                vmovups(zmm | r_mask,
                        EVEX_compress_addr(reg_aux_ptr_src, inp_offset));
            }
        }
        for (int y = 0; y < jcp.alpha; y++) {
            vsubps(vreg_tmp(y * jcp.alpha + 0), vreg_inp(y * jcp.alpha + 0),
                    vreg_inp(y * jcp.alpha + 2));
            vaddps(vreg_tmp(y * jcp.alpha + 1), vreg_inp(y * jcp.alpha + 1),
                    vreg_inp(y * jcp.alpha + 2));
            vsubps(vreg_tmp(y * jcp.alpha + 2), vreg_inp(y * jcp.alpha + 2),
                    vreg_inp(y * jcp.alpha + 1));
            vsubps(vreg_tmp(y * jcp.alpha + 3), vreg_inp(y * jcp.alpha + 1),
                    vreg_inp(y * jcp.alpha + 3));
        }
        for (int x = 0; x < jcp.alpha; x++) {
            vsubps(vreg_out(x + 0 * jcp.alpha), vreg_tmp(x + jcp.alpha * 0),
                    vreg_tmp(x + jcp.alpha * 2));
            vaddps(vreg_out(x + 1 * jcp.alpha), vreg_tmp(x + jcp.alpha * 1),
                    vreg_tmp(x + jcp.alpha * 2));
            vsubps(vreg_out(x + 2 * jcp.alpha), vreg_tmp(x + jcp.alpha * 2),
                    vreg_tmp(x + jcp.alpha * 1));
            vsubps(vreg_out(x + 3 * jcp.alpha), vreg_tmp(x + jcp.alpha * 1),
                    vreg_tmp(x + jcp.alpha * 3));
        }

        for (int i = 0; i < 16; i++) {
            out_offset = sizeof(float) * (jcp.inp_stride * i);
            vmovups(EVEX_compress_addr(reg_aux_ptr_dst, out_offset),
                    vreg_out(i));
        }

        add(reg_aux_ptr_src, sizeof(float) * jcp.ih * jcp.iw * load_block);
        add(reg_aux_ptr_dst, sizeof(float) * load_block);
    }
    dec(reg_ic_block);
    cmp(reg_ic_block, 0);
    jg(ic_block_label, T_NEAR);
    postamble();
}

/// DST TRANSFORMS /////////////////////////////////////////////////////////////
struct jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(
            jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t)

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

    jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t(
            jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp), attr_(attr) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(
                const_cast<uint8_t *>(getCode()));
    }

    void generate();
    bool maybe_relu(int position);

    Zmm vreg_inp(int i) { // 16
        assert(i < jcp.alpha * jcp.alpha);
        return Zmm(31 - i);
    }

    Zmm vreg_stg(int id) { // 8
        const int id_reg_stg = jcp.alpha * jcp.alpha + id;
        assert(id_reg_stg < jcp.alpha * jcp.alpha + 8);
        return Zmm(31 - id_reg_stg);
    }

    Zmm vreg_out(int id) { // 4
        const int id_reg_out = jcp.alpha * jcp.alpha + 8 + id;
        assert(id_reg_out < jcp.alpha * jcp.alpha + 12);
        return Zmm(31 - id_reg_out);
    }

    Zmm vreg_tmp(int id) { // 2
        const int id_reg_tmp = jcp.alpha * jcp.alpha + 12 + id;
        assert(id_reg_tmp < jcp.alpha * jcp.alpha + 14);
        return Zmm(31 - id_reg_tmp);
    }

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_prev_dst = Zmm(0);
    Zmm vreg_bias = Zmm(2);

    Opmask y_mask = Opmask(1);
    Opmask r_mask = Opmask(2);
    Opmask x_mask(int id) {
        assert (id < 4);
        return Opmask(3 + id);
    }

    Reg64 reg_ptr_v_y_masks = r12;
    Reg64 reg_ptr_v_x_masks = r11;

    Reg64 reg_aux_ptr_src = r10;
    Reg64 reg_aux_ptr_dst = r9;

    Reg64 reg_oc_block = r8;

    Reg64 reg_ptr_bias = rbx;
    Reg64 reg_ptr_scales = abi_not_param1;
    Reg64 reg_ptr_sum_scale = rdx;
};

bool jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t::maybe_relu(int position) {
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* relu before sum */
        return false
            || p.contain(eltwise, 0);
    } else if (position == 1) {
        /* relu after sum */
        const int sum_idx = p.contain(sum, 0)
            ? 0 : (p.contain(sum, 1) ? 1 : -1);
        if (sum_idx == -1)
            return false;

        return false
            || p.contain(eltwise, sum_idx + 1);
    }

    return false;
}

void jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t::generate() {
    Label oc_block_label;

    const int load_block = 16;

    auto loop_body = [=]() {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = (sum_idx != -1)
                ? &p.entry_[sum_idx].sum.scale
                : nullptr;
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_ptr_sum_scale, (size_t)p_sum_scale);

        for (int i = 0; i < 16; i++) {
            int internal_offset = sizeof(float) * jcp.out_stride * i;
            vmovups(vreg_inp(i),
                EVEX_compress_addr(reg_aux_ptr_src, internal_offset));
        }
        for (int y = 0; y < jcp.alpha; y++) {
            vaddps(vreg_tmp(0), vreg_inp(y * 4 + 0), vreg_inp(y * 4 + 1));
            vaddps(vreg_stg(y * 2), vreg_tmp(0), vreg_inp(y * 4 + 2));

            vsubps(vreg_tmp(1), vreg_inp(y * 4 + 1), vreg_inp(y * 4 + 2));
            vsubps(vreg_stg(y * 2+1), vreg_tmp(1), vreg_inp(y * 4 + 3));
        }
        for (int x = 0; x < jcp.m; x++) {
            vaddps(vreg_tmp(0), vreg_stg(x), vreg_stg(x+2 * 1));
            vaddps(vreg_out(x), vreg_tmp(0), vreg_stg(x+2 * 2));

            vsubps(vreg_tmp(1), vreg_stg(x+2 * 1), vreg_stg(x+2 * 2));
            vsubps(vreg_out(x+2), vreg_tmp(1), vreg_stg(x+2 * 3));
        }


        if (jcp.with_bias) {
            auto bias_addr = ptr [ reg_ptr_bias ];
            vmovups(vreg_bias, bias_addr);
        }
        for (int y = 0; y < jcp.m; y++) {
            kmovw(y_mask, ptr[ reg_ptr_v_y_masks + sizeof(int16_t) * y ]);
            for (int x = 0; x < jcp.m; x++) {
                kandw(r_mask, y_mask, x_mask(x));

                int i = y * jcp.m + x;
                int offset = sizeof(float) *
                    (y * jcp.ow * jcp.oc_block + x * jcp.oc_block);
                Address addr = EVEX_compress_addr(reg_aux_ptr_dst, offset);

                Zmm zmm = vreg_out(i);
                if (jcp.with_bias)
                    vaddps(zmm, zmm, vreg_bias);
                vmulps(zmm, zmm, ptr [reg_ptr_scales]);

                if (maybe_relu(0)) {
                    vxorps(vreg_zero, vreg_zero, vreg_zero);
                    vmaxps(zmm, vreg_zero, zmm);
                }
                if (p_sum_scale) { // post_op: sum
                    vxorps(vreg_prev_dst, vreg_prev_dst, vreg_prev_dst);
                    vmovups(vreg_prev_dst | r_mask, addr);
                    if (*p_sum_scale == 1.f)
                        vaddps(zmm, vreg_prev_dst);
                    else
                        vfmadd231ps(zmm, vreg_prev_dst,
                            zword_b[reg_ptr_sum_scale]);
                }
                if (maybe_relu(1)) {
                    vxorps(vreg_zero, vreg_zero, vreg_zero);
                    vmaxps(zmm, vreg_zero, zmm);
                }

                vmovups(addr, zmm | r_mask);
            }
        }
    };

    preamble();

#define READ_PARAM(reg, field) \
        mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_aux_ptr_src, wino_dst);
    READ_PARAM(reg_aux_ptr_dst, dst);
    READ_PARAM(reg_ptr_v_y_masks, v_y_masks);
    READ_PARAM(reg_ptr_v_x_masks, v_x_masks);
    READ_PARAM(reg_ptr_bias, bias);
    READ_PARAM(reg_ptr_scales, scales);
#undef READ_PARAM

    for (int i = 0; i < jcp.alpha * jcp.alpha; i++)
        vxorps(vreg_inp(i), vreg_inp(i), vreg_inp(i));

    for (int i = 0; i < jcp.alpha; i++)
        kmovw(x_mask(i), ptr[reg_ptr_v_x_masks + sizeof(int16_t) * i]);

    int oc_blocks = 1;
    oc_blocks = jcp.oc / load_block;
    mov(reg_oc_block, oc_blocks);
    L(oc_block_label);
    {
        loop_body();
        add(reg_aux_ptr_src, sizeof(float) * load_block);
        add(reg_aux_ptr_dst, sizeof(float) * jcp.oh * jcp.ow * load_block);

        add(reg_ptr_scales, jcp.is_oc_scale * sizeof(float) * load_block);
        add(reg_ptr_bias, jcp.typesize_bia * load_block);
    }
    dec(reg_oc_block);
    cmp(reg_oc_block, 0);
    jg(oc_block_label, T_NEAR);

    sub(reg_ptr_scales, jcp.is_oc_scale * sizeof(float) * load_block);
    sub(reg_ptr_bias, oc_blocks * jcp.typesize_bia * load_block);

    postamble();

}

/// GEMM kernel ////////////////////////////////////////////////////////////////
struct jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t)
    jit_conv_conf_2x3_wino_t jcp;

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

    jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t(
            jit_conv_conf_2x3_wino_t ajcp, const primitive_attr_t &attr)
        : jcp(ajcp) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(
                const_cast<uint8_t *>(getCode()));
    }

    static status_t init_conf(
            jit_conv_conf_2x3_wino_t &jcp, const convolution_desc_t &cd,
            memory_desc_t &src_md, memory_desc_t &weights_md,
            memory_desc_t &dst_md, memory_desc_t &bias_md,
            const primitive_attr_t &attr,
            memory_desc_t& expect_wei_md);

    Zmm vreg_out(int n, int m) {
        const int id_reg_out = n * jcp.m_block + m;
        assert(id_reg_out < jcp.n2_block * jcp.m_block);
        return Zmm(31 - id_reg_out);
    }
    Zmm vreg_wei(int i) {
        assert (31 - jcp.n2_block * jcp.m_block - i > 1);
        return Zmm(31 - jcp.n2_block * jcp.m_block - i);
    }

    Zmm vreg_src = Zmm(0);
    Zmm vreg_one = Zmm(1);
    Zmm vreg_tmp = Zmm(2);

    Reg64 reg_ptr_src = r15;

    Reg64 reg_aux_dst = r12;
    Reg64 reg_aux_dst2 = r11;
    Reg64 reg_aux_wei = r10;
    Reg64 reg_aux_wei2 = r9;
    Reg64 reg_aux_src = r8;
    Reg64 reg_aux_src2 = rax;

    Reg64 reg_mb = rbx;
    Reg64 reg_nnb = rdx;
    Reg64 reg_K = rsi;

};

bool jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t::post_ops_ok(
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

void jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t::generate() {
    Label nnb_loop_label, K_loop_label, mb_loop_label;

    preamble();
#define READ_PARAM(reg, field) \
    mov(reg, ptr[abi_param1 + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src, src);
    READ_PARAM(reg_aux_dst, dst);
    READ_PARAM(reg_aux_wei, wei);
#undef READ_PARAM

    if (!jcp.small_mb) {
        mov(reg_nnb, jcp.n_chunks);
        L(nnb_loop_label);
    }
    mov(reg_aux_dst2, reg_aux_dst);
    mov(reg_aux_src, reg_ptr_src);
    mov(reg_mb, jcp.M / jcp.m_block);
    L(mb_loop_label);
    {
        int nb2 = 0;
        for (nb2 = 0; nb2 < jcp.n2_block; nb2++) {
            for (int m = 0; m < jcp.m_block; m++) {
                vxorps(vreg_out(nb2, m), vreg_out(nb2, m), vreg_out(nb2, m));
            }
        }
        mov(reg_aux_src2, reg_aux_src);
        mov(reg_aux_wei2, reg_aux_wei);

        mov(reg_K, jcp.k_chunks);
        L(K_loop_label); {
            int wei_offset = 0;
            for (int _i = 0; _i < jcp.k2_block; _i++) {
                for (int nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                    if (jcp.small_mb) {
                        int wei_offset = sizeof(float)
                                * ((nb2 * jcp.nb_ic * jcp.ic_block
                                           * jcp.oc_block)
                                          + _i * jcp.oc_block);
                        vmovups(vreg_wei(nb2),
                                EVEX_compress_addr(reg_aux_wei2, wei_offset));
                    } else {
                        vmovups(vreg_wei(nb2),
                                EVEX_compress_addr(reg_aux_wei2,
                                        sizeof(float) * wei_offset));
                        wei_offset += jcp.oc_block;
                    }
                }
                for (int m = 0; m < jcp.m_block; m++) {
                    int inp_offset = sizeof(float) * (m * jcp.K + _i);
                    if (jcp.n2_block > 1) {
                        vbroadcastss(vreg_src,
                            EVEX_compress_addr(reg_aux_src2, inp_offset));
                        for (int nb2 = 0; nb2 < jcp.n2_block; nb2++)
                            vfmadd231ps(vreg_out(nb2, m), vreg_wei(nb2),
                                vreg_src);
                    } else {
                        vfmadd231ps(vreg_out(0, m), vreg_wei(0),
                            EVEX_compress_addr(reg_aux_src2, inp_offset, true));
                    }
                }
            }
            add(reg_aux_src2, sizeof(float) * jcp.ic_block);
            if (jcp.small_mb)
                add(reg_aux_wei2, sizeof(float) * jcp.oc_block * jcp.ic_block);
            else
                add(reg_aux_wei2,
                        sizeof(float) * jcp.k2_block * jcp.n2_block
                                * jcp.oc_block);
        }
        dec(reg_K);
        cmp(reg_K, 0);
        jg(K_loop_label, T_NEAR);

        for (int m = 0; m < jcp.m_block; m++) {
            int nb2 = 0;
            for (nb2 = 0; nb2 < jcp.n2_block; nb2++) {
                int offset = sizeof(float) *
                    (m * jcp.N + nb2 * jcp.oc_block);
                vmovups(EVEX_compress_addr(reg_aux_dst2,offset),
                            vreg_out(nb2, m));
            }
        }
        add(reg_aux_src, sizeof(float) * jcp.m_block * jcp.K);
        add(reg_aux_dst2, sizeof(float) * jcp.m_block * jcp.N);
    }
    dec(reg_mb);
    cmp(reg_mb, 0);
    jg(mb_loop_label, T_NEAR);

    if (!jcp.small_mb) {
        add(reg_aux_dst, sizeof(float) * jcp.n2_block * jcp.oc_block);
        add(reg_aux_wei,
                sizeof(float) * jcp.k_chunks * jcp.ic_block * jcp.n2_block
                        * jcp.oc_block);

        dec(reg_nnb);
        cmp(reg_nnb, 0);
        jg(nnb_loop_label, T_NEAR);
    }
    postamble();
}

namespace {
bool is_winograd_faster_than_direct(const jit_conv_conf_2x3_wino_t &jcp) {
    return jcp.mb >= 4;
}
}

status_t jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t ::init_conf(
        jit_conv_conf_2x3_wino_t &jcp, const convolution_desc_t &cd,
        memory_desc_t &src_md, memory_desc_t &wei_md,
        memory_desc_t &dst_md, memory_desc_t &bias_md,
        const primitive_attr_t &attr, memory_desc_t &expect_wei_md) {
    const memory_desc_wrapper src_d(&src_md);
    const memory_desc_wrapper wei_d(&wei_md);
    const memory_desc_wrapper dst_d(&dst_md);
    const memory_desc_wrapper bias_d(&bias_md);

    const bool with_groups = wei_d.ndims() == src_d.ndims() + 1;

    jcp.nthr = mkldnn_get_max_threads();

    jcp.ngroups = with_groups ? wei_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
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

    jcp.m = 2;
    jcp.r = 3;
    jcp.alpha = jcp.m + jcp.r - 1;
    int simdw = 16;

    format_tag_t dat_tag = format_tag::nChw16c;
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    if (jcp.src_tag != dat_tag) return status::unimplemented;
    if (jcp.dst_tag != dat_tag) return status::unimplemented;

    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    bool ok_to_pad_channels = jcp.ngroups == 1;
    if (ok_to_pad_channels) {
        jcp.oc = rnd_up(jcp.oc, simdw);
        jcp.ic = rnd_up(jcp.ic, simdw);
    }

    jcp.ver = ver_avx512_core;
    if (!(mayiuse(avx512_core)))
        return status::unimplemented;

    if (!IMPLICATION(cd.alg_kind == alg_kind::convolution_auto,
               is_winograd_faster_than_direct(jcp)))
        return status::unimplemented;

    if (src_d.data_type() != data_type::f32)
        return status::unimplemented;
    if (wei_d.data_type() != data_type::f32)
        return status::unimplemented;
    if (dst_d.data_type() != data_type::f32)
        return status::unimplemented;

    jcp.ic_block = simdw;
    jcp.oc_block = simdw;

    bool ok = true && jcp.kh == 3 && jcp.kw == 3 && jcp.ngroups == 1
            && jcp.oc % jcp.oc_block == 0 && jcp.ic % jcp.ic_block == 0
            && jcp.stride_h == 1 && jcp.stride_w == 1 && jcp.dilate_h == 0
            && jcp.dilate_w == 0 && jcp.t_pad == jcp.b_pad
            && jcp.l_pad == jcp.r_pad && jcp.t_pad < 2 && jcp.t_pad >= 0
            && jcp.l_pad < 2 && jcp.l_pad >= 0;
    if (!ok)
        return status::unimplemented;

    const int L2_cap = get_cache_size(2, true) / sizeof(float);
    const int L3_capacity = get_cache_size(3, false) / sizeof(float);
    int a = jcp.alpha;
    int aa = a * a;
    int mb = jcp.mb;
    int ic = jcp.ic;
    int oc = jcp.oc;
    int ih = jcp.ih;
    int iw = jcp.iw;
    auto wei_sz = (float)aa * ic * oc;
    auto inp_sz = (float)mb * ih * iw * ic;
    auto sp_sz = (float)mb * ih * iw;

    /* Heuristics here. Numbers '28','196' is an observation from data. */
    if (wei_sz / inp_sz > 5)
        jcp.small_mb = true;
    else
        jcp.small_mb = false;

    if (mb > nstl::min(jcp.nthr, 28)
        || (!jcp.small_mb
            && (wei_sz >= 0.9f * L2_cap
                || inp_sz > L2_cap * jcp.nthr + L3_capacity))
        || (jcp.small_mb && sp_sz > 196))
        return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.typesize_bia
            = jcp.with_bias ? types::data_type_size(bias_d.data_type()) : 0;

    jcp.nb_oc = jcp.oc / jcp.oc_block;
    jcp.nb_ic = jcp.ic / jcp.ic_block;

    const int skx_free_regs = 30;

    auto find_m_n2_blocks = [=](int xb, int yb, int &M, int &m_block,
                                    int &n2_block, float &reg_eff) {
        M = (xb * yb) / jcp.alpha;
        int max_m_block = m_block = nstl::min(M, skx_free_regs);
        int max_n2_block = n2_block = nstl::min(jcp.nb_oc, skx_free_regs);
        reg_eff = 0;
        for (int im = max_m_block; im > 0; im--) {
            for (int in2 = max_n2_block; in2 > 0; in2--) {
                int used_regs = in2 * im + in2;
                float cur_reg_eff = ((float)in2 * im) / (im + in2) / 2.5f;
                if (M % im || jcp.nb_oc % in2 || used_regs > skx_free_regs
                        || cur_reg_eff <= reg_eff)
                    continue;
                reg_eff = cur_reg_eff;
                m_block = im;
                n2_block = in2;
            }
        }
    };

    int oh = jcp.oh;
    int ow = jcp.ow;
    int nb_oc = jcp.nb_oc;
    int Z = ic + oc;
    int Y = ic * oc;
    const int L3_cap_per_core = get_cache_size(3, true) / sizeof(float);

    /* Selecting xb and yb blocking */
    int min_yb = jcp.alpha;
    int min_xb = jcp.alpha;
    int max_yb = nstl::max(min_yb, rnd_up(ih, 2));
    int max_xb = nstl::max(min_xb, rnd_up(iw, 2));
    float best_eff = 0.f;
    for (int ix = max_xb; ix >= min_xb; ix -= 2) {
        if (rnd_up(ow, ix) < iw - 2)
            continue;
        for (int iy = max_yb; iy >= min_yb; iy -= 2) {
            if (rnd_up(oh, iy) < ih - 2)
                continue;
            int ex_y = rnd_up(oh, iy);
            int ex_x = rnd_up(ow, ix);
            float work_eff = (float)(ih * iw) / (ex_y * ex_x);

            int M, m_block, n2_b;
            float reg_eff, thr_eff, par_eff, mem_eff, req_mem;

            find_m_n2_blocks(ix, iy, M, m_block, n2_b, reg_eff);

            /* outer parallelization */
            int nblocks = mb * div_up(ih, iy) * div_up(iw, ix);
            thr_eff = (float)nblocks / rnd_up(nblocks, jcp.nthr);

            mem_eff = 1.f;
            req_mem = (((float)ix + 2) * (iy + 2) + aa * M) * Z + aa * Y;
            if (req_mem > L2_cap / 2) {
                if (req_mem > ((L2_cap + L3_cap_per_core) * 4) / 7)
                    mem_eff /= (n2_b + 1) / 2.f;
                else
                    mem_eff /= (n2_b + 1) / 3.f;
            }

            float outer_eff = thr_eff + work_eff + reg_eff + mem_eff;

            /* inner parallelization */
            int bsz = iy * ix / a;
            int gemmw = aa * (nb_oc / n2_b);
            int bsz_r = rnd_up(bsz, jcp.nthr);
            int gemmw_r = rnd_up(gemmw, jcp.nthr);
            thr_eff = ((float)Z * bsz / bsz_r + Y * gemmw / gemmw_r) / (Z + Y);

            req_mem = (float)ix * iy * (ic + simdw * n2_b) + simdw * n2_b * ic;
            mem_eff = nstl::min(1.f, L2_cap / req_mem);
            int M_per_thr = nstl::max(2, div_up(aa, jcp.nthr));
            int oc_per_thr =
                nstl::min(oc, div_up(aa * (nb_oc / n2_b), jcp.nthr));
            req_mem = (float)aa * oc_per_thr * ic + M_per_thr * M * Z;
            if (req_mem > L2_cap)
                mem_eff = 0.1f;
            par_eff = 1 / (2.f * nblocks);

            float inner_eff = thr_eff + work_eff + mem_eff + par_eff;

            float eff = jcp.small_mb ? inner_eff : outer_eff;
            if (eff > best_eff) {
                best_eff = eff;
                jcp.yb = iy;
                jcp.xb = ix;
                jcp.M = M;
                jcp.m_block = m_block;
                jcp.n2_block = n2_b;
            }
        }
    }

    assert(jcp.xb % 2 == 0 && jcp.yb % 2 == 0);

    jcp.inp_stride = jcp.M * jcp.ic;
    jcp.out_stride = jcp.M * jcp.oc;
    jcp.wei_stride = jcp.ic * jcp.oc;
    jcp.bia_stride = jcp.oc;

    jcp.N = jcp.oc;
    jcp.K = jcp.ic;

    jcp.n_block = jcp.oc_block;
    jcp.k_block = jcp.ic_block;

    assert(jcp.M % jcp.m_block == 0);
    assert(jcp.nb_oc % jcp.n2_block == 0);

    jcp.n_chunks = jcp.nb_oc / jcp.n2_block;
    jcp.k2_block = jcp.ic_block;
    jcp.k_chunks = jcp.K / jcp.k2_block;

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    /* re-create weights primitive descriptor
                                    and set weights wino_blocking */
    expect_wei_md.format_kind = format_kind::wino;
    expect_wei_md.data_type = data_type::f32;
    mkldnn_wino_desc_t &wd = expect_wei_md.format_desc.wino_desc;
    wd.wino_format
            = jcp.small_mb ? mkldnn_wino_wei_aaOio : mkldnn_wino_wei_aaOBiOo;
    wd.r = jcp.r;
    wd.alpha = jcp.alpha;
    wd.ic = jcp.ic;
    wd.oc = jcp.oc;
    wd.ic_block = jcp.ic_block;
    wd.oc_block = jcp.oc_block;
    wd.oc2_block = jcp.n2_block;
    wd.ic2_block = 1;
    wd.adj_scale = 1.f;
    size_t max_size = sizeof(float) * jcp.alpha * jcp.alpha * jcp.ic * jcp.oc;
    wd.size = max_size;

    return status::success;
}
////////////////////////////////////////////////////////////////////////////////

status_t jit_avx512_core_fp32_wino_conv_2x3_fwd_t
    ::pd_t::jit_conf(memory_desc_t& expect_wei_md) {
    return jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t::init_conf(
            jcp_, *this->desc(), this->src_md_, this->weights_md_,
            this->dst_md_,this->bias_md_, *this->attr(), expect_wei_md);
}

jit_avx512_core_fp32_wino_conv_2x3_fwd_t::
        jit_avx512_core_fp32_wino_conv_2x3_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd)
{
    kernel_ = new jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t(
            pd()->jcp_, *pd()->attr());
    src_trans_ = new jit_avx512_core_fp32_wino_conv_2x3_src_trans_t(
            pd()->jcp_, *pd()->attr());
    dst_trans_ = new jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t(
            pd()->jcp_, *pd()->attr());
}

jit_avx512_core_fp32_wino_conv_2x3_fwd_t
    ::~jit_avx512_core_fp32_wino_conv_2x3_fwd_t() {
    delete kernel_;
    delete src_trans_;
    delete dst_trans_;
}

void jit_avx512_core_fp32_wino_conv_2x3_fwd_t::execute_forward_mbN(
        const float *src, const float *wei, const float *bia, float *dst,
        const memory_tracking::grantor_t &scratchpad) const
{
    const auto &jcp = kernel_->jcp;
    const auto &oscales = pd()->attr()->output_scales_;

    const size_t wino_size_offset =
        (size_t)(pd()->jcp_.yb / 2) * (pd()->jcp_.xb / 2) + (pd()->jcp_.xb);
    const size_t size_wino_src = wino_size_offset * pd()->jcp_.ic * 16;
    const size_t size_wino_dst = wino_size_offset * pd()->jcp_.oc * 16;

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.get<float>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bia, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bia = padded_bias;
    }

    auto ptr_V = scratchpad.get<float>(key_wino_V);
    auto ptr_M = scratchpad.get<float>(key_wino_M);

    parallel_nd(jcp.mb, div_up(jcp.oh,jcp.yb), div_up(jcp.ow, jcp.xb),
        [&](int mb, int tile_y_b, int tile_x_b) {
        int tile_y = tile_y_b * jcp.yb;
        int tile_x = tile_x_b * jcp.xb;

        int ithr = mkldnn_get_thread_num();
        auto wino_src = ptr_V + size_wino_src * ithr;
        auto wino_dst = ptr_M + size_wino_dst * ithr;

        auto src_trans_p =
            jit_avx512_core_fp32_wino_conv_2x3_src_trans_t
                ::call_params_t();
        auto dst_trans_p =
            jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t
                ::call_params_t();
        auto gemm_p = jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t ::
                call_params_t();

        /* transformation of input tensor to winograd domain */
        for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb;
                    x_in_block += 2) {

                unsigned short v_y_masks[4], v_x_masks[4];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2)
                        + (x_in_block / 2);

                int v_ys = nstl::max(0, jcp.t_pad - y);
                int v_ye = nstl::min(jcp.alpha,
                        nstl::max(0, jcp.ih + jcp.t_pad - y));

                int v_xs = nstl::max(0, jcp.l_pad - x);
                int v_xe = nstl::min(jcp.alpha,
                        nstl::max(0, jcp.iw + jcp.l_pad - x));

#pragma unroll(4)
                for (int i = 0; i < jcp.alpha; i++) {
                    v_y_masks[i] = (i < v_ys || i >= v_ye) ? 0 : 0xffff;
                    v_x_masks[i] = (i < v_xs || i >= v_xe) ? 0 : 0xffff;
                }
                auto local_s = src
                        + mb * jcp.nb_ic * jcp.ih * jcp.iw
                                * jcp.ic_block
                        + y * jcp.iw * jcp.ic_block + x * jcp.ic_block;
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
            int offset = (tile_ij + ithr) % 16;
            gemm_p.src = wino_src + jcp.inp_stride * offset;
            gemm_p.dst = wino_dst + jcp.out_stride * offset;
            gemm_p.wei = wei + jcp.wei_stride * offset;

            kernel_->ker_(&gemm_p);
        }

        /* transformation from winograd domain to output tensor */
        for (int y_in_block = 0; y_in_block < jcp.yb; y_in_block += 2) {
            for (int x_in_block = 0; x_in_block < jcp.xb;
                    x_in_block += 2) {
                unsigned short v_y_masks[2], v_x_masks[2];

                int y = y_in_block + tile_y;
                int x = x_in_block + tile_x;
                int m = (y_in_block / 2) * (jcp.xb / 2)
                        + (x_in_block / 2);

#pragma unroll(2)
                for (int i = 0; i < jcp.m; i++) {
                    v_x_masks[i] = (x + i < jcp.ow) ? 0xffff : 0;
                    v_y_masks[i] = (y + i < jcp.oh) ? 0xffff : 0;
                }
                auto local_d = dst
                        + mb * jcp.nb_oc * jcp.oh * jcp.ow
                                * jcp.oc_block
                        + y * jcp.ow * jcp.oc_block + x * jcp.oc_block;
                auto local_w = wino_dst + m * jcp.oc;

                auto scales = oscales.scales_;
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

void jit_avx512_core_fp32_wino_conv_2x3_fwd_t::execute_forward_small_mb(
        const float *src, const float *wei, const float *bia, float *dst,
        const memory_tracking::grantor_t &scratchpad) const
{
    const auto &jcp = kernel_->jcp;
    const auto &oscales = pd()->attr()->output_scales_;

    if (pd()->wants_padded_bias()) {
        auto padded_bias = scratchpad.get<float>(key_conv_padded_bias);
        utils::array_copy(padded_bias, bia, jcp.oc_without_padding);
        utils::array_set(padded_bias + jcp.oc_without_padding, 0.f,
                jcp.oc - jcp.oc_without_padding);
        bia = padded_bias;
    }

    auto ptr_V = scratchpad.get<float>(key_wino_V);
    auto ptr_M = scratchpad.get<float>(key_wino_M);

    for (int mb = 0; mb < jcp.mb; mb++) {
    for (int tile_y = 0; tile_y < jcp.oh; tile_y += jcp.yb) {
    for (int tile_x = 0; tile_x < jcp.ow; tile_x += jcp.xb) {
        /* transformation of input tensor to winograd domain */
        parallel_nd(div_up(jcp.yb, 2), div_up(jcp.xb, 2),
            [&](int y_in_block_b, int x_in_block_b) {
            int y_in_block = y_in_block_b * 2;
            int x_in_block = x_in_block_b * 2;

            auto src_trans_p = jit_avx512_core_fp32_wino_conv_2x3_src_trans_t ::
                    call_params_t();

            unsigned short v_y_masks[4], v_x_masks[4];

            int y = y_in_block + tile_y;
            int x = x_in_block + tile_x;
            int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

            int v_ys = nstl::max(0, jcp.t_pad - y);
            int v_ye = nstl::min(
                    jcp.alpha, nstl::max(0, jcp.ih + jcp.t_pad - y));

            int v_xs = nstl::max(0, jcp.l_pad - x);
            int v_xe = nstl::min(
                    jcp.alpha, nstl::max(0, jcp.iw + jcp.l_pad - x));

#pragma unroll(4)
            for (int i = 0; i < jcp.alpha; i++) {
                v_y_masks[i] = (i < v_ys || i >= v_ye) ? 0 : 0xffff;
                v_x_masks[i] = (i < v_xs || i >= v_xe) ? 0 : 0xffff;
            }
            auto local_s = src
                    + mb * jcp.nb_ic * jcp.ih * jcp.iw * jcp.ic_block
                    + y * jcp.iw * jcp.ic_block + x * jcp.ic_block;
            auto local_w = ptr_V + m * jcp.ic;

            src_trans_p.src = local_s;
            src_trans_p.wino_src = local_w;
            src_trans_p.v_y_masks = v_y_masks;
            src_trans_p.v_x_masks = v_x_masks;

            src_trans_->ker_(&src_trans_p);
        });

        /* gemms */
        parallel_nd(16, jcp.n_chunks, [&](int tile_ij, int nnb) {
            auto gemm_p = jit_avx512_core_fp32_wino_conv_2x3_fwd_ker_t ::
                    call_params_t();

            gemm_p.src = ptr_V + jcp.inp_stride * tile_ij;
            gemm_p.dst = ptr_M + jcp.out_stride * tile_ij
                    + nnb * jcp.n2_block * jcp.n_block;
            gemm_p.wei = wei + jcp.wei_stride * tile_ij
                    + nnb * jcp.n2_block * jcp.n_block * jcp.K;

            kernel_->ker_(&gemm_p);
        });

        /* transformation from winograd domain to output tensor */

        parallel_nd(div_up(jcp.yb, 2), div_up(jcp.xb, 2),
            [&](int y_in_block_b, int x_in_block_b) {
            int y_in_block = y_in_block_b * 2;
            int x_in_block = x_in_block_b * 2;

            auto dst_trans_p = jit_avx512_core_fp32_wino_conv_2x3_dst_trans_t ::
                    call_params_t();

            unsigned short v_y_masks[2], v_x_masks[2];

            int y = y_in_block + tile_y;
            int x = x_in_block + tile_x;
            int m = (y_in_block / 2) * (jcp.xb / 2) + (x_in_block / 2);

#pragma unroll(2)
            for (int i = 0; i < jcp.m; i++) {
                v_x_masks[i] = (x + i < jcp.ow) ? 0xffff : 0;
                v_y_masks[i] = (y + i < jcp.oh) ? 0xffff : 0;
            }
            auto local_d = dst
                    + mb * jcp.nb_oc * jcp.oh * jcp.ow * jcp.oc_block
                    + y * jcp.ow * jcp.oc_block + x * jcp.oc_block;
            auto local_w = ptr_M + m * jcp.oc;

            auto scales = oscales.scales_;
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

} // namespace cpu
} // namespace impl
} // namespace mkldnn
