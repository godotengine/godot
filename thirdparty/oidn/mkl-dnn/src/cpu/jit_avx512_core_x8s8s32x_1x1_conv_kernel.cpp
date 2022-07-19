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
#include "nstl.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "cpu_memory.hpp"

#include "jit_uni_1x1_conv_utils.hpp"
#include "jit_avx512_core_x8s8s32x_1x1_conv_kernel.hpp"

#define GET_OFF(field) offsetof(jit_1x1_conv_call_s, field)

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;

using namespace Xbyak;

bool jit_avx512_core_x8s8s32x_1x1_conv_kernel::maybe_eltwise(int position)
{
    using namespace primitive_kind;
    const auto &p = attr_.post_ops_;

    if (position == 0) {
        /* eltwise before sum */
        return p.contain(eltwise, 0);
    } else if (position == 1) {
        /* eltwise after sum */
        return p.contain(sum, 0) && p.contain(eltwise, 1);
    }

    return false;
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::bcast_loop(int load_loop_blk)
{
    mov(aux1_reg_bcast_data, reg_bcast_data);
    mov(aux_reg_bcast_data, reg_bcast_data);

    mov(aux_reg_output_data, reg_output_data);
    mov(bcast_loop_iter, EVEX_compress_addr(rsp, bcast_loop_work_off));

    Label bcast_loop;
    Label bcast_loop_tail;

    cmp(bcast_loop_iter, jcp.ur);
    jl(bcast_loop_tail, T_NEAR);

    L(bcast_loop); {
        assert(jcp.bcast_block % jcp.ur == 0);
        int num_substeps = jcp.bcast_block / jcp.ur;
        assert(num_substeps > 0 && num_substeps < 10);
        for (int i = 0; i < num_substeps; i++) {
            reduce_loop(load_loop_blk, jcp.ur, i, false);
            if (i < num_substeps - 1) {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_substep);
                add(aux_reg_output_data, jcp.bcast_loop_output_substep);
            }
            else {
                add(aux1_reg_bcast_data, jcp.bcast_loop_bcast_step
                    - (num_substeps - 1) * jcp.bcast_loop_bcast_substep);
                int output_offset = jcp.bcast_loop_output_step
                    - (num_substeps - 1) * jcp.bcast_loop_output_substep;

                add(aux_reg_output_data, output_offset);
            }
        }
        sub(bcast_loop_iter, jcp.bcast_block);
        cmp(bcast_loop_iter, jcp.bcast_block);
        jge(bcast_loop, T_NEAR);
    }

    L(bcast_loop_tail);
    if (jcp.ur_tail) {
        Label bcast_loop_tail_out;
        cmp(bcast_loop_iter, 0);
        jz(bcast_loop_tail_out, T_NEAR);
        reduce_loop(load_loop_blk, jcp.ur_tail, 0, true);
        L(bcast_loop_tail_out);
    }
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::cvt2ps(data_type_t type_in,
        zmm_t zmm_in, const Xbyak::Operand &op, bool mask_flag) {
    zmm_t zmm = mask_flag ? zmm_in | ktail_mask | T_z : zmm_in;
    switch (type_in) {
    case data_type::f32:
    case data_type::s32: vmovups(zmm, op); break;
    case data_type::s8: vpmovsxbd(zmm, op); break;
    case data_type::u8: vpmovzxbd(zmm, op); break;
    default: assert(!"unsupported data type");
    }
    if (type_in != data_type::f32)
        vcvtdq2ps(zmm_in, zmm_in);
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::reduce_loop(int load_loop_blk,
         int ur, int substep, bool wraparound)
{
    auto vreg_load = [=](int i_load) {
        return Zmm(ur * load_loop_blk + i_load);
    };

    auto vreg_accum = [=](int i_load, int i_ur) {
        return Zmm(i_ur * load_loop_blk + i_load);
    };

    auto zmm_bias_alpha = [=]() {
        return Zmm(ur * load_loop_blk);
    };

    auto xmm_bias_alpha = [=]() {
        return Xmm(ur * load_loop_blk);
    };
    auto bias_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_bias_data,
                                  jcp.typesize_bia * jcp.oc_block * i_load);
    };

    auto comp_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_comp_data,
                                  sizeof(int32_t) * jcp.oc_block * i_load);
    };

    auto scale_ptr = [=](int i_load) {
        return EVEX_compress_addr(reg_ptr_scales,
                    jcp.is_oc_scale * (sizeof(float) * jcp.oc_block * i_load));
    };

    auto bcast_ptr = [=](int i_reduce, int i_ur, bool bcast) {
        assert(i_ur < jcp.ur);
        assert(i_reduce <= jcp.reduce_loop_unroll);
        assert(jcp.reduce_loop_unroll == jcp.reduce_block);

        int offt = (jcp.ic_without_padding * i_ur + i_reduce);

        return EVEX_compress_addr(aux_reg_bcast_data, jcp.typesize_in * offt,
                                bcast);
    };

    auto load_ptr = [=](int i_reduce, int i_load) {
        int u0 = i_reduce % jcp.reduce_loop_unroll;
        int u1 = i_reduce / jcp.reduce_loop_unroll;

        int offt = (i_load * jcp.reduce_dim + u0) * jcp.load_block;

        return EVEX_compress_addr(aux_reg_load_data,
                                  u1 * jcp.reduce_loop_load_step
                                  + jcp.typesize_in * offt);
    };

    auto output_ptr = [=](int i_load, int i_ur) {
        return EVEX_compress_addr(aux_reg_output_data,
            jcp.typesize_out * (jcp.oc_without_padding * i_ur
                                + i_load * jcp.load_block));
    };

    auto init = [=]() {
        for (int i_load = 0; i_load < load_loop_blk; ++i_load)
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vpxord(r, r, r);
            }
        if (jcp.signed_input) {
            xor_(reg_scratch, reg_scratch);
            Reg8 _t8 = reg_scratch.cvt8();
            mov(_t8, (int8_t)-128);
            vpbroadcastb(zmm_shift, _t8);
        }
    };

    auto store = [=](const bool mask_flag_in) {
        const auto &p = attr_.post_ops_;
        const int sum_idx = p.find(primitive_kind::sum);
        const float *p_sum_scale = (sum_idx != -1)
            ? &p.entry_[sum_idx].sum.scale
            : nullptr;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        if (p_sum_scale && *p_sum_scale != 1.f) {
            mov(EVEX_compress_addr(rsp, reg_load_data_off), reg_load_data);
            mov(reg_ptr_sum_scale, (size_t)p_sum_scale);
        }
        if (jcp.signed_input && jcp.ver != ver_vnni) {
            mov(reg_scratch, float2int(jcp.wei_adj_scale));
            vmovq(xmm_bias_alpha(), reg_scratch);
            vbroadcastss(zmm_bias_alpha(), xmm_bias_alpha());
        }
        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in && i_load == load_loop_blk - 1;
            auto zmm_bias = zmm_tmp;
            auto zmm_comp = zmm_bcast;
            if (jcp.with_bias) {
                if (jcp.signed_input)
                    mov(reg_bias_data,
                        EVEX_compress_addr(rsp,reg_bias_data_off));
                cvt2ps(jcp.bia_dt, zmm_bias, bias_ptr(i_load), mask_flag);
                if (jcp.signed_input && jcp.ver != ver_vnni)
                    vmulps(zmm_bias, zmm_bias, zmm_bias_alpha());
            }
            if (jcp.signed_input) {
                mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
                cvt2ps(data_type::s32, zmm_comp, comp_ptr(i_load), mask_flag);
            }

            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                vcvtdq2ps(r, r);
                if (jcp.signed_input)
                    vaddps(r, r, zmm_comp);
                if (jcp.with_bias)
                    vaddps(r, r, zmm_bias);

                zmm_t mask_zmm = mask_flag ? r | ktail_mask | T_z : r;
                vmulps(mask_zmm, r, scale_ptr(i_load));
            }
        }

        if (maybe_eltwise(0))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        if (p_sum_scale) { // post_op: sum
            for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                const bool mask_flag = mask_flag_in &&
                                           i_load == load_loop_blk - 1;
                for (int i_ur = 0; i_ur < ur; ++i_ur) {
                    vpxord(zmm_zero, zmm_zero, zmm_zero);
                    auto zmm_prev_dst = zmm_zero;

                    auto r = vreg_accum(i_load, i_ur);
                    cvt2ps(jcp.dst_dt, zmm_prev_dst, output_ptr(i_load, i_ur),
                        mask_flag);

                    if (*p_sum_scale == 1.f)
                        vaddps(r, zmm_prev_dst);
                    else
                        vfmadd231ps(r, zmm_prev_dst, zword_b[reg_ptr_sum_scale]);
                }
            }
        }

        if (maybe_eltwise(1))
            eltwise_injector_->compute_vector_range(0, ur * load_loop_blk);

        for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
            const bool mask_flag = mask_flag_in &&
                                       i_load == load_loop_blk - 1;
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                if (jcp.dst_dt == data_type::u8) {
                    vpxord(zmm_zero, zmm_zero, zmm_zero);
                    vmaxps(r, zmm_zero, r);
                }
                if (jcp.dst_dt != data_type::f32)
                    vcvtps2dq(r, r);
            }
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                auto r = vreg_accum(i_load, i_ur);
                zmm_t r_zmm = mask_flag ? r | ktail_mask : r;

                switch (jcp.dst_dt) {
                case data_type::f32:
                case data_type::s32:
                    vmovups(output_ptr(i_load, i_ur), r_zmm); break;
                case data_type::s8:
                    vpmovsdb(output_ptr(i_load, i_ur), r_zmm); break;
                case data_type::u8:
                    vpmovusdb(output_ptr(i_load, i_ur), r_zmm); break;
                default: assert(!"unknown dst_dt");
                }
            }
        }
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        if (p_sum_scale && *p_sum_scale != 1.f)
            mov(reg_load_data, EVEX_compress_addr(rsp, reg_load_data_off));
    };

    auto compute = [=](Zmm vreg_acc, Zmm vreg_wei, Zmm vreg_src) {
        if (jcp.ver == ver_vnni) {
            vpdpbusd(vreg_acc, vreg_src, vreg_wei);
        } else {
            vpmaddubsw(zmm_tmp, vreg_src, vreg_wei);
            vpmaddwd(zmm_tmp, zmm_tmp, zmm_one);
            vpaddd(vreg_acc, vreg_acc, zmm_tmp);
        }
    };

    auto fma_block = [=](bool last_block) {
        int reduce_step = 4;
        int tail_size = jcp.ic_without_padding % reduce_step;
        int loop_unroll = last_block && jcp.ic != jcp.ic_without_padding
            ? rnd_up(jcp.ic_without_padding % jcp.ic_block, reduce_step)
            : jcp.reduce_loop_unroll;
        for (int i_reduce = 0; i_reduce < loop_unroll;
                i_reduce += reduce_step) {
            for (int i_load = 0; i_load < load_loop_blk; ++i_load)
                vmovups(vreg_load(i_load), load_ptr(i_reduce, i_load));
            for (int i_ur = 0; i_ur < ur; ++i_ur) {
                if (last_block && tail_size != 0
                    && i_reduce == loop_unroll - reduce_step) {
                    Xmm xmm_bcast = Xmm(zmm_bcast.getIdx());
                    for (int r = 0; r < tail_size; ++r)
                        vpinsrb(xmm_bcast, xmm_bcast, ptr[aux_reg_bcast_data
                        + jcp.ic_without_padding * i_ur + i_reduce + r], r);
                    vpbroadcastd(zmm_bcast, xmm_bcast);
                } else {
                    vpbroadcastd(zmm_bcast, bcast_ptr(i_reduce, i_ur, false));
                }
                if (jcp.signed_input)
                    vpsubb(zmm_bcast, zmm_bcast, zmm_shift);
                for (int i_load = 0; i_load < load_loop_blk; ++i_load) {
                    compute(vreg_accum(i_load, i_ur),
                                vreg_load(i_load), zmm_bcast);
                }
            }
        }
    };

    Label reduce_loop;
    Label reduce_loop_tail;

    mov(aux_reg_load_data, reg_load_data);

    mov(aux_reg_bcast_data, aux1_reg_bcast_data);
    init();

    mov(reduce_loop_iter, reg_reduce_loop_work);
    sub(reduce_loop_iter, jcp.reduce_loop_unroll);
    jle(reduce_loop_tail, T_NEAR);

    L(reduce_loop); {
        fma_block(false);
        add(aux_reg_bcast_data, jcp.reduce_loop_bcast_step);
        add(aux_reg_load_data, jcp.reduce_loop_load_step);
        sub(reduce_loop_iter, jcp.reduce_loop_unroll);
        jg(reduce_loop, T_NEAR);
    }

    L(reduce_loop_tail);
    if (jcp.ic != jcp.ic_without_padding) {
        fma_block(true);
    } else {
        fma_block(false);
    }

    if (jcp.oc_without_padding != jcp.oc) {
        Label end_store, common_store;
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);

        /*Check if it is the last load_loop_blk*/
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
        cmp(reg_load_loop_work, 0);
        jg(common_store, T_NEAR);

        /*Check if it is the last ocb*/
        test(reg_reduce_pos_flag, FLAG_OC_LAST);
        jz(common_store, T_NEAR);

        store(true);
        jmp(end_store, T_NEAR);

        L(common_store);
        store(false);

        L(end_store);

        add(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    } else {
        store(false);
    }
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::generate()
{
    preamble();

    xor_(reg_scratch, reg_scratch);
    Reg16 _t = reg_scratch.cvt16();
    mov(_t, 0x1);
    vpbroadcastw(zmm_one, _t);

    sub(rsp, stack_space_needed);

    if (jcp.oc_without_padding != jcp.oc) {
        int tail_size = jcp.oc_without_padding % jcp.oc_block;
        int mask = (1 << tail_size) - 1;
        Reg32 regw_tmp = reg_last_load.cvt32();
        mov(regw_tmp, mask);
        kmovw(ktail_mask, regw_tmp);
    }

    if (jcp.with_bias)
        mov(reg_bias_data, ptr[param1 + GET_OFF(bias_data)]);
    if (jcp.signed_input) {
        mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        mov(reg_comp_data, ptr[param1 + GET_OFF(compensation)]);
        mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
    }
    mov(reg_ptr_scales, ptr[param1 + GET_OFF(scales)]);
    mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
    mov(reg_bcast_data, ptr[param1 + GET_OFF(bcast_data)]);
    mov(reg_load_data, ptr[param1 + GET_OFF(load_data)]);
    mov(reg_output_data, ptr[param1 + GET_OFF(output_data)]);

    mov(reg_load_loop_work, ptr[param1 + GET_OFF(load_dim)]);
    mov(reg_bcast_loop_work, ptr[param1 + GET_OFF(bcast_dim)]);
    mov(EVEX_compress_addr(rsp, bcast_loop_work_off), reg_bcast_loop_work);
    mov(reg_reduce_loop_work, ptr[param1 + GET_OFF(reduce_dim)]);
    mov(reg_reduce_pos_flag, ptr[param1 + GET_OFF(first_last_flag)]);


    auto load_loop_body = [=](int load_loop_blk) {
        bcast_loop(load_loop_blk);
        add(reg_load_data, load_loop_blk * jcp.load_loop_load_step);
        if (jcp.with_bias) {
            if (jcp.signed_input)
                mov(reg_bias_data, EVEX_compress_addr(rsp, reg_bias_data_off));
            add(reg_bias_data,
                load_loop_blk * jcp.load_block * jcp.typesize_bia);
            if (jcp.signed_input)
                mov(EVEX_compress_addr(rsp, reg_bias_data_off), reg_bias_data);
        }
        if (jcp.signed_input) {
            mov(reg_comp_data, EVEX_compress_addr(rsp, reg_comp_data_off));
            add(reg_comp_data,
                load_loop_blk * jcp.load_block * sizeof(int32_t));
            mov(EVEX_compress_addr(rsp, reg_comp_data_off), reg_comp_data);
        }
        mov(EVEX_compress_addr(rsp, reg_bcast_data_off), reg_bcast_data);
        mov(reg_ptr_scales, EVEX_compress_addr(rsp, reg_ptr_sum_scale_off));
        add(reg_ptr_scales,
            jcp.is_oc_scale * load_loop_blk * jcp.load_block * sizeof(float));
        mov(EVEX_compress_addr(rsp, reg_ptr_sum_scale_off), reg_ptr_scales);
        mov(reg_bcast_data, EVEX_compress_addr(rsp, reg_bcast_data_off));
        add(reg_output_data,
            load_loop_blk * jcp.load_block * jcp.typesize_out);
        sub(reg_load_loop_work, load_loop_blk * jcp.load_loop_iter_step);
    };

    const int simd_w = 16;

    Label load_loop_blk[7];

    static const int ur_cases_fma_expl_bcast[] = { 2, 5, 6, 9, 14, 32 };
    const int size_ur_cases_fma = sizeof(ur_cases_fma_expl_bcast);
    const int *ur_cases_fma = ur_cases_fma_expl_bcast;
    const int *ur_cases = ur_cases_fma;
    const int num_ur_cases = (size_ur_cases_fma) / sizeof(*ur_cases);

    for (int ur_idx = num_ur_cases - 1; ur_idx > 0; ur_idx--) {
        int label_idx = num_ur_cases - ur_idx - 1;
        if (jcp.ur <= ur_cases[ur_idx]) {
            cmp(reg_load_loop_work, simd_w * (label_idx + 1));
            jle(load_loop_blk[label_idx], T_NEAR);
        }
    }

    for (int ur_idx = 0; ur_idx < num_ur_cases; ur_idx++) {
        if (jcp.ur <= ur_cases[ur_idx]) {
            int label_idx = num_ur_cases - ur_idx - 1;
            L(load_loop_blk[label_idx]);
            {
                if (label_idx == 0) {
                    cmp(reg_load_loop_work, 0);
                    je(load_loop_blk[num_ur_cases], T_NEAR);
                }

                for (int _i = 1; _i <= label_idx + 1; _i++) {
                    prefetcht0(ptr [ reg_load_data + _i * jcp.ic * jcp.oc_block ]);
                    prefetcht1(ptr [ reg_output_data + _i * jcp.oc_block ]);
                }

                load_loop_body(label_idx + 1);
                if (label_idx - 1 > 0) {
                    cmp(reg_load_loop_work, 2 * label_idx * simd_w);
                    je(load_loop_blk[label_idx - 1], T_NEAR);
                }
                cmp(reg_load_loop_work, (label_idx + 1) * simd_w);
                jge(load_loop_blk[label_idx]);
            }
            for (int idx = label_idx - 1; idx > 0; --idx) {
                cmp(reg_load_loop_work, simd_w * (idx + 1));
                je(load_loop_blk[idx], T_NEAR);
            }
            if (ur_idx < num_ur_cases - 2) {
                cmp(reg_load_loop_work, simd_w);
                jle(load_loop_blk[0], T_NEAR);
            }
        }
    }
    L(load_loop_blk[num_ur_cases]);

    add(rsp, stack_space_needed);

    postamble();

    if (jcp.with_eltwise)
        eltwise_injector_->prepare_table();
}

bool jit_avx512_core_x8s8s32x_1x1_conv_kernel::post_ops_ok(
        jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace primitive_kind;
    const auto &p = attr.post_ops_;

    auto is_eltwise = [&](int idx) { return p.entry_[idx].is_eltwise(); };

    switch (p.len_) {
    case 0: return true;
    case 1: return is_eltwise(0) || p.contain(sum, 0);
    case 2: return (p.contain(sum, 0) && is_eltwise(1))
                       || (p.contain(sum, 1) && is_eltwise(0));
    default: return false;
    }

    return false;
}

status_t jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_conf(
        jit_1x1_conv_conf_t &jcp, const convolution_desc_t &cd,
        const memory_desc_wrapper &src_d, const memory_desc_wrapper &weights_d,
        const memory_desc_wrapper &dst_d, const memory_desc_wrapper &bias_d,
        const primitive_attr_t &attr, int nthreads, bool reduce_src) {
    if (!mayiuse(avx512_core)) return status::unimplemented;

    const bool with_groups = weights_d.ndims() == src_d.ndims() + 1;
    if (!one_of(src_d.data_type(), data_type::u8, data_type::s8)
        || weights_d.data_type() != data_type::s8
        || !one_of(dst_d.data_type(),
            data_type::f32, data_type::s32, data_type::s8, data_type::u8))
        return status::unimplemented;
    jcp.ver = ver_avx512_core;
    if (mayiuse(avx512_core_vnni))
        jcp.ver = ver_vnni;

    jcp.ngroups = with_groups ? weights_d.dims()[0] : 1;
    jcp.mb = src_d.dims()[0];
    jcp.oc = dst_d.dims()[1] / jcp.ngroups;
    jcp.oc_without_padding = jcp.oc;
    jcp.ic = src_d.dims()[1] / jcp.ngroups;
    jcp.ic_without_padding = jcp.ic;
    jcp.ih = src_d.dims()[2];
    jcp.iw = src_d.dims()[3];
    jcp.oh = dst_d.dims()[2];
    jcp.ow = dst_d.dims()[3];
    jcp.kh = weights_d.dims()[with_groups + 2];
    jcp.kw = weights_d.dims()[with_groups + 3];
    jcp.t_pad = cd.padding[0][0];
    jcp.l_pad = cd.padding[0][1];
    jcp.stride_h = cd.strides[0];
    jcp.stride_w = cd.strides[1];
    jcp.with_bias = cd.bias_desc.format_kind != format_kind::undef;

    jcp.signed_input = (src_d.data_type() == data_type::s8) ? true : false;

    jcp.os = jcp.oh * jcp.ow;
    jcp.is = jcp.ih * jcp.iw;
    jcp.tr_is = rnd_up(jcp.is, 4);

    if (!post_ops_ok(jcp, attr))
        return status::unimplemented;

    const auto &p = attr.post_ops_;
    const int eltwise_ind = p.find(primitive_kind::eltwise);
    jcp.with_eltwise = eltwise_ind != -1;
    if (jcp.with_eltwise)
        jcp.eltwise = p.entry_[eltwise_ind].eltwise;

    format_tag_t dat_tag = format_tag::nhwc;
    jcp.src_tag = src_d.matches_one_of_tag(dat_tag);
    jcp.dst_tag = dst_d.matches_one_of_tag(dat_tag);

    bool args_ok = true
        && jcp.ngroups == 1
        && jcp.src_tag == dat_tag
        && jcp.dst_tag == dat_tag;
    if (!args_ok) return status::unimplemented;

    const int simd_w = 16;

    jcp.oc = rnd_up(jcp.oc, simd_w);
    jcp.ic = rnd_up(jcp.ic, simd_w);

    args_ok = true
        && jcp.oc % simd_w == 0 && jcp.ic % simd_w == 0
        && jcp.t_pad == 0 && jcp.l_pad == 0
        && jcp.stride_w == 1 && jcp.stride_h == 1 // TODO: support some strides
        && jcp.kh == 1 && jcp.kw == 1;
    if (!args_ok) return status::unimplemented;

    jcp.bia_dt = jcp.with_bias ? cd.bias_desc.data_type : data_type::undef;
    jcp.dst_dt = cd.dst_desc.data_type;

    jcp.ic_block = jcp.oc_block = simd_w;

    jcp.typesize_in = types::data_type_size(src_d.data_type());
    jcp.typesize_out = types::data_type_size(dst_d.data_type());
    jcp.typesize_bia = jcp.with_bias
        ? types::data_type_size(bias_d.data_type())
        : 0;

    const int SMALL_SPATIAL = 7 * 7;
    const int BIG_REDUCE_DIM = 1024;

    int load_blocking = 0;
    int load_blocking_max = 0;
    int bcast_blocking = 0;
    int bcast_blocking_max = 0;
    int reduce_blocking = 0;
    int reduce_blocking_max = 0;
    jcp.load_grp_count = 1;
    jcp.use_vmovntps = false;

    const int L2_size = get_cache_size(2, true) / sizeof(jcp.typesize_in);
    const int L2_capacity = (L2_size * 3) / 4;

    int size_treshold = 28;
    int max_regs = 0;
    int min_regs = 6;
    if (jcp.ver == ver_vnni)
        max_regs = ((jcp.oh > size_treshold && jcp.ow > size_treshold)
                    && (jcp.oc < 128 || jcp.ic < 128)) ?  min_regs : 9;
    else
        max_regs = 8;
    jcp.expl_bcast = true;

    if (jcp.mb == 1 && jcp.ic > 128
        && (jcp.oh <= size_treshold && jcp.ow <= size_treshold)) {
        if (jcp.os <= SMALL_SPATIAL && jcp.oc * jcp.ic < L2_size)
            max_regs = min_regs; // mobilenet_v2 performance improvement
        jcp.ur = nstl::min(max_regs, jcp.os);
    } else {
        const int spatial = jcp.oh;
        jcp.ur = 1;
        for (int ur_w = max_regs; ur_w >= min_regs; ur_w--) {
            if ((spatial >= size_treshold && spatial % ur_w == 0)
                    || (spatial < size_treshold && jcp.os % ur_w == 0)) {
                jcp.ur = ur_w;
                break;
            }
        }
        if (jcp.ur == 1) {
            jcp.ur = nstl::min(max_regs, jcp.os);
            int os_tail = jcp.os % max_regs;
            for (int i = max_regs; i >= min_regs; i--) {
                int i_tail = jcp.os % i;
                if (i_tail > os_tail || i_tail == 0) {
                    jcp.ur = i;
                    os_tail = i_tail;
                    if (i_tail == 0)
                        break;
                }
            }
        }
    }

    jcp.reduce_dim = jcp.ic;
    jcp.reduce_block = jcp.ic_block;

    jcp.load_dim = jcp.oc;
    jcp.load_block = jcp.oc_block;

    jcp.bcast_dim = jcp.is;

    jcp.bcast_block = jcp.ur;

    jcp.reduce_loop_unroll = jcp.reduce_block;
    jcp.reduce_loop_bcast_step
            = jcp.reduce_loop_unroll * jcp.typesize_in;

    jcp.reduce_loop_load_step
            = jcp.reduce_loop_unroll * jcp.load_block * jcp.typesize_in;

    jcp.bcast_loop_output_step = jcp.ur * jcp.oc_without_padding * jcp.typesize_out;
    jcp.bcast_loop_output_substep = -1; // unused
    jcp.bcast_loop_bcast_step = jcp.ur * jcp.ic_without_padding * jcp.typesize_in;
    jcp.bcast_loop_bcast_substep = -1; // unused

    jcp.load_loop_load_step
            = jcp.reduce_dim * jcp.load_block * jcp.typesize_in;

    jcp.load_loop_iter_step = jcp.load_block;

    jcp.loop_order = reduce_src ? loop_blr : loop_lbr;

    int nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    int nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    reduce_blocking = nb_reduce;
    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 64;
    else if (jcp.bcast_dim > SMALL_SPATIAL && jcp.reduce_dim >= BIG_REDUCE_DIM)
        reduce_blocking = 16;
    reduce_blocking = best_divider(nb_reduce, 1, reduce_blocking, true);
    reduce_blocking *= jcp.reduce_block;

    bool cmp_reduce = reduce_blocking <= jcp.reduce_dim;
    if (cmp_reduce)
        jcp.loop_order = reduce_src ? loop_rbl : loop_rlb;
    load_blocking = jcp.load_dim;

    jcp.load_grp_count = div_up(nthreads, jcp.mb * jcp.ngroups * nb_bcast);
    jcp.load_grp_count = best_divider(
            nthreads, jcp.load_grp_count, 2 * jcp.load_grp_count, false);

    if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.load_dim * jcp.reduce_dim >= L2_size) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 4);
    } else if (jcp.bcast_dim <= SMALL_SPATIAL && jcp.mb <= nthreads
            && jcp.load_dim > 512 && jcp.load_dim / jcp.reduce_dim >= 4) {
        jcp.load_grp_count = nstl::max(jcp.load_grp_count, 2); //
        load_blocking = jcp.load_block;
    }

    bcast_blocking = div_up(jcp.mb * jcp.ngroups * nb_bcast,
                             div_up(nthreads, jcp.load_grp_count)) * jcp.bcast_block;
    bcast_blocking = nstl::min(jcp.bcast_dim, bcast_blocking);
    bcast_blocking = rnd_up(bcast_blocking, jcp.bcast_block);

    int space_for_bcast
            = (L2_capacity - /* kernel_size - */
                2 * jcp.load_block * reduce_blocking
                    - jcp.ur * reduce_blocking - 3 * 1024);
    if (jcp.reduce_dim * jcp.bcast_dim > L2_capacity)
        space_for_bcast /= 2;

    int bcast_in_cache
            = nstl::max(jcp.bcast_block, space_for_bcast / reduce_blocking);
    bcast_blocking = nstl::min(
            bcast_blocking, rnd_dn(bcast_in_cache, jcp.bcast_block));

    load_blocking_max = load_blocking;
    bcast_blocking_max = bcast_blocking * 3 / 2;
    reduce_blocking_max = reduce_blocking;

    assert(load_blocking);
    assert(load_blocking_max);
    assert(bcast_blocking);
    assert(bcast_blocking_max);
    assert(reduce_blocking);
    assert(reduce_blocking_max);
    assert(load_blocking % jcp.load_block == 0);
    assert(reduce_blocking % jcp.reduce_block == 0);
    assert(load_blocking_max % jcp.load_block == 0);
    assert(reduce_blocking_max % jcp.reduce_block == 0);

    assert(jcp.reduce_loop_unroll % 4 == 0);
    assert(jcp.reduce_dim % jcp.reduce_loop_unroll == 0);

    assert(jcp.bcast_block % jcp.ur == 0);
    assert(jcp.reduce_dim % jcp.reduce_block == 0);

    jcp.ur_tail = jcp.bcast_dim % jcp.ur;

    jcp.nb_bcast_blocking = bcast_blocking / jcp.bcast_block;
    jcp.nb_bcast_blocking_max = bcast_blocking_max / jcp.bcast_block;
    jcp.nb_load_blocking = load_blocking / jcp.load_block;
    jcp.nb_load_blocking_max = load_blocking_max / jcp.load_block;
    jcp.nb_reduce_blocking = reduce_blocking / jcp.reduce_block;
    jcp.nb_reduce_blocking_max = reduce_blocking_max / jcp.reduce_block;

    jcp.nb_bcast = div_up(jcp.bcast_dim, jcp.bcast_block);
    jcp.nb_load = div_up(jcp.load_dim, jcp.load_block);
    jcp.nb_reduce = div_up(jcp.reduce_dim, jcp.reduce_block);

    // miniumum size of load dim chunk for work distribution within threads
    jcp.nb_load_chunk = 1;
    // peformance improvements for googlenet_v3, mb=1;
    // TODO: generalize this condition and rewrite it in appropriate manner
    if (jcp.mb == 1 && jcp.nb_load % 4 == 0 && jcp.ic / jcp.oc >= 4
            && jcp.ic * jcp.oc <= L2_size) {
        jcp.nb_load_chunk = 4;
        jcp.load_grp_count = nstl::max(jcp.nb_load / 4, jcp.load_grp_count);
    }

    const auto &oscales = attr.output_scales_;
    jcp.is_oc_scale = oscales.mask_ == 1 << 1;
    assert(IMPLICATION(!jcp.is_oc_scale, oscales.mask_ == 0));

    jcp.wei_adj_scale =
        (weights_d.extra().flags | memory_extra_flags::scale_adjust)
        ? weights_d.extra().scale_adjust : 1.f;

    return status::success;
}

void jit_avx512_core_x8s8s32x_1x1_conv_kernel::init_scratchpad(
        memory_tracking::registrar_t &scratchpad,
        const jit_1x1_conv_conf_t &jcp, const primitive_attr_t &attr) {
    using namespace mkldnn::impl::memory_tracking::names;

    if (jcp.signed_input && jcp.ver != ver_vnni) {
        dim_t count = nstl::max<dim_t>(attr.output_scales_.count_, 16);
        scratchpad.book(key_conv_adjusted_scales, sizeof(float) * count);
    }
}

}
}
}
