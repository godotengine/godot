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

#include "c_types_map.hpp"
#include "utils.hpp"
#include "type_helpers.hpp"
#include "mkldnn_thread.hpp"
#include "math_utils.hpp"

#include "simple_q10n.hpp"

#include "gemm/gemm.hpp"
#include "gemm_x8s8s32x_convolution.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::math;
using namespace mkldnn::impl::memory_tracking::names;

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::
execute_forward(const exec_ctx_t &ctx) const {
    auto src_base = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst_base = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    auto scratchpad = this->scratchpad(ctx);

    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    assert(IMPLICATION(
            jcp.id != 1, jcp.oh_block == jcp.oh && jcp.ow_block == jcp.ow));
    assert(IMPLICATION(jcp.ow_block != jcp.ow, jcp.oh_block == 1));

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_forward_thr(ithr, nthr, src_base, wei_base, bia_base, dst_base,
                scratchpad);
    });
}

template <data_type_t src_type, data_type_t dst_type>
_gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::pp_ker_t::pp_ker_t(
    const pd_t *pd)
    : ker_(nullptr)
    , jcp_(pd->jcp_)
    , OC_(pd->jcp_.oc)
    , OS_(pd->jcp_.os)
    , bias_data_type_(data_type::undef)
    , bias_data_type_size_(0)
    , scale_idx_mult_(0)
    , do_bias_(false)
    , do_relu_(false)
    , do_sum_(false)
{
    using namespace types;

    const auto dst_md = memory_desc_wrapper(pd->dst_md());
    dst_os_stride_ = dst_md.blk_off(0, 0, 0, 1);

    scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));

    auto &post_ops = pd->attr()->post_ops_;

    int entry_idx = -1;
    for (int idx = 0; idx < post_ops.len_; ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            entry_idx = idx;
            break;
        }
    }
    do_relu_ = entry_idx >= 0;

    do_signed_scaling_ = jcp_.signed_input;

    do_sum_ = post_ops.contain(primitive_kind::sum, 0);
    do_bias_ = pd->with_bias();
    bias_data_type_ = pd->desc()->bias_desc.data_type;
    if (do_bias_) {
        assert(bias_data_type_ != data_type::undef);
        bias_data_type_size_ = data_type_size(bias_data_type_);
    }
    const size_t vlen_start
            = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    for (size_t i = vlen_start; i > 0; i--) {
        if (OC_ % i == 0) {
            vlen_ = i;
            break;
        }
    }

    if (!mayiuse(avx512_core))
        // use fallback code for older CPUs
        return;
    else
        generate();
}

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::pp_ker_t::generate()
{
    using namespace Xbyak;
    using namespace utils;

    // TODO: clean-up
    Reg64 reg_param = abi_param1;
    Reg64 reg_dst = rdx;
    Reg64 reg_acc = rax;
    Reg64 reg_bias = rbx;
    Reg64 reg_scales = rsi;

    Reg64 reg_len = r8;
    Reg64 reg_tmp = rcx; // intentional for shifting purposes
    Reg64 reg_oc_offset = r9;
    Reg64 reg_rem_mask_short = r10;
    Reg64 reg_rem_mask_vlen = r11;
    Opmask kreg_rem_mask_short = k1;
    Opmask kreg_rem_mask_vlen = k3;
    Opmask kreg_relu_cmp = k2;

    const size_t vlen = vlen_;

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_scale = Zmm(1);
    Zmm vreg_nslope = Zmm(2);
    Zmm vreg_sum_scale = Zmm(3);
    Zmm vreg_signed_scale = Zmm(4);

    size_t def_unroll = 4;
    size_t max_unroll = 12;
    size_t zmm_step = 2;
    if (do_sum_) {
        max_unroll = 8;
        zmm_step = 3;
    }

    auto vreg_dst = [&](int idx) {
        return Zmm(5 + idx * zmm_step + 0);
    };
    auto vreg_bias = [&](int idx) {
        return Zmm(5 + idx * zmm_step + 1);
    };
    auto vreg_prev_dst = [&](int idx) {
        return Zmm(5 + idx * zmm_step + 2);
    };

    preamble();

#define PARAM_OFF(x) offsetof(ker_args, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    vbroadcastss(vreg_nslope, ptr[reg_param + PARAM_OFF(nslope)]);
    vbroadcastss(vreg_sum_scale, ptr[reg_param + PARAM_OFF(sum_scale)]);
    vbroadcastss(vreg_signed_scale, ptr[reg_param + PARAM_OFF(signed_scale)]);
    if (scale_idx_mult_ == 0)
        vbroadcastss(vreg_scale, dword[reg_scales]);

#undef PARAM_OFF

    mov(reg_rem_mask_vlen, 1);
    shl(reg_rem_mask_vlen, vlen);
    sub(reg_rem_mask_vlen, 1);
    kmovq(kreg_rem_mask_vlen, reg_rem_mask_vlen);

    if (do_relu_ || dst_type == data_type::u8)
        vxorps(vreg_zero, vreg_zero, vreg_zero);

    // Load accumulated value, convert to float, apply sum (if any),
    // bias (if any), scaling, and relu (if any);
    // then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        if (scale_idx_mult_ > 0) {
            assert(scale_idx_mult_ == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            if (apply_mask)
                vreg_scale_ = vreg_scale_ | kreg_rem_mask_short;
            else
                vreg_scale_ = vreg_scale_ | kreg_rem_mask_vlen;
            vmovups(vreg_scale_, scale_addr);
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (apply_mask)
            vreg_dst_ = vreg_dst_ | kreg_rem_mask_short;
        else
            vreg_dst_ = vreg_dst_ | kreg_rem_mask_vlen;
        vcvtdq2ps(vreg_dst_, acc_addr);

        if (do_signed_scaling_)
            vmulps(vreg_dst(idx), vreg_dst(idx), vreg_signed_scale);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_short;
            else
                vreg_bias_ = vreg_bias_ | kreg_rem_mask_vlen;

            switch (bias_data_type_) {
            case data_type::s8:
                vpmovsxbd(vreg_bias_, bias_addr);
                break;
            case data_type::u8:
                vpmovzxbd(vreg_bias_, bias_addr);
                break;
            case data_type::s32:
            case data_type::f32:
                vmovups(vreg_bias_, bias_addr);
                break;
            default: assert(!"unimplemented");
            }
            if (bias_data_type_ != data_type::f32)
                vcvtdq2ps(vreg_bias(idx), vreg_bias(idx));
            vaddps(vreg_dst(idx), vreg_dst(idx), vreg_bias(idx));
        }

        vmulps(vreg_dst(idx), vreg_dst(idx), vreg_scale);

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];

        if (do_sum_)
        {
            auto vreg_prev_dst_ = vreg_prev_dst(idx);
            if (apply_mask)
                vreg_prev_dst_ = vreg_prev_dst_ | kreg_rem_mask_short;
            else
                vreg_prev_dst_ = vreg_prev_dst_ | kreg_rem_mask_vlen;

            switch (dst_type) {
            case data_type::f32:
            case data_type::s32: vmovups(vreg_prev_dst_, dst_addr); break;
            case data_type::s8: vpmovsxbd(vreg_prev_dst_, dst_addr); break;
            case data_type::u8: vpmovzxbd(vreg_prev_dst_, dst_addr); break;
            default: assert(!"unsupported data type");
            }
            if (dst_type != data_type::f32)
                vcvtdq2ps(vreg_prev_dst(idx), vreg_prev_dst(idx));

            vfmadd231ps(vreg_dst(idx), vreg_prev_dst(idx), vreg_sum_scale);
        }

        if (do_relu_) {
            vcmpps(kreg_relu_cmp, vreg_dst(idx), vreg_zero, _cmp_lt_os);
            vmulps(vreg_dst(idx) | kreg_relu_cmp, vreg_dst(idx), vreg_nslope);
        }

        if (dst_type != data_type::f32) {
            vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
        }

        if (dst_type == data_type::u8)
            vpmaxsd(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        switch (dst_type) {
        case data_type::s8:
            vpmovsdb(dst_addr, vreg_dst_);
            break;
        case data_type::u8:
            vpmovusdb(dst_addr, vreg_dst_);
            break;
        case data_type::f32:
        case data_type::s32:
            vmovups(dst_addr, vreg_dst_);
            break;
        default: assert(!"unimplemented");
        }
    };

    // Advance all pointers by an immediate
    auto advance_ptrs_imm = [&](size_t offset) {
        add(reg_dst, offset * sizeof(dst_data_t));
        add(reg_acc, offset * sizeof(acc_data_t));
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            add(reg_scales, offset * sizeof(float));
        }
        if (do_bias_)
            add(reg_bias, offset * bias_data_type_size_);
    };

    // Advance all pointers by a value stored in a register
    auto advance_ptrs_reg = [&](Reg64 offset) {
        lea(reg_dst, ptr[reg_dst + offset * sizeof(dst_data_t)]);
        lea(reg_acc, ptr[reg_acc + offset * sizeof(acc_data_t)]);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            lea(reg_scales, ptr[reg_scales + offset * sizeof(float)]);
        }
        if (do_bias_)
            lea(reg_bias, ptr[reg_bias + offset * bias_data_type_size_]);
    };

    // Rewind pointers that point to data that is indexed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
        add(reg_dst, (dst_os_stride_ - OC_) * sizeof(dst_data_t));
    };

    //                    <--------- OC --------------->
    //
    // ^  ................+..............+-------------+.......................
    // |  .               : not accessed |Prologue loop|                      .
    // |  .               +--------------+-------------+                      .
    //    .               |                            |                      .
    // O  .               |  Main loop (unrolled)      |                      .
    // S  .               |                            |                      .
    //    .               +--------------+-------------+                      .
    // |  .               | Epilogue loop|not accessed :                      .
    // v  ................+--------------+.............+.......................

    Label prologue_end;
    cmp(reg_oc_offset, 0);
    je(prologue_end, T_NEAR);

    // Prologue loop
    {
        mov(reg_tmp, OC_);
        sub(reg_tmp, reg_oc_offset);
        cmp(reg_tmp, reg_len);
        cmovg(reg_tmp, reg_len);
        sub(reg_len, reg_tmp);

        Label prologue_loop, prologue_loop_tail, prologue_loop_end;
        cmp(reg_tmp, vlen);
        jle(prologue_loop_tail, T_NEAR);
        L(prologue_loop); {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        mov(reg_rem_mask_short, 1);
        // cl == reg_tmp because reg_tmp <= vlen here
        shl(reg_rem_mask_short, cl);
        sub(reg_rem_mask_short, 1);
        jz(prologue_loop_end, T_NEAR);

        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
        advance_ptrs_reg(reg_tmp);

        L(prologue_loop_end);
        rewind_ptrs();
    }
    L(prologue_end);

    // Main loop
    Label main_loop_end;
    {
        cmp(reg_len, OC_);
        jle(main_loop_end, T_NEAR);

        Label main_loop;
        L(main_loop); {
            size_t OC_loop, OC_tail;
            if (OC_ < max_unroll * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = OC_;
            }
            else {
                OC_loop = vlen * def_unroll;
                OC_tail = OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            if (OC_tail % vlen) {
                int vlen_tail = OC_tail % vlen;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask_short, reg_tmp);
            }

            if (OC_loop) {
                mov(reg_tmp, rnd_dn(OC_, OC_loop));
                Label oc_loop;
                L(oc_loop); {
                    for (size_t offset = 0; offset < OC_loop; offset += vlen)
                        compute(offset, offset / vlen, false);
                    advance_ptrs_imm(OC_loop);
                    sub(reg_tmp, OC_loop);
                    jnz(oc_loop);
                }
            }

            if (OC_tail) {
                for (size_t offset = 0; offset < OC_tail; offset += vlen) {
                    bool use_mask = (offset + vlen) > OC_tail;
                    compute(offset, offset / vlen, use_mask);
                }
                advance_ptrs_imm(OC_tail);
            }

            rewind_ptrs();
            sub(reg_len, OC_);
            cmp(reg_len, OC_);
            jge(main_loop, T_NEAR);
        }
    }
    L(main_loop_end);

    // Epilogue loop
    Label epilogue_end;
    {
        cmp(reg_len, 0);
        je(epilogue_end, T_NEAR);

        Label epilogue_loop, epilogue_loop_tail;
        cmp(reg_len, vlen);
        jle(epilogue_loop_tail, T_NEAR);
        L(epilogue_loop); {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        mov(reg_rem_mask_short, 1);
        shl(reg_rem_mask_short, cl); // reg_tmp == rcx and reg_tail < vlen
        sub(reg_rem_mask_short, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask_short, reg_rem_mask_short);
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    ker_ = getCode<decltype(ker_)>();
}

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::pp_ker_t::operator ()
    (dst_data_t *dst, const acc_data_t *acc, const char *bias,
        const float *scales, float nslope, float sum_scale, float signed_scale,
        int g, size_t start, size_t end)
{
    using math::get_bias;

    if (end <= start)
        return;

    if (ker_) {
        // JIT
        ker_args args;
        size_t oc_offset = start % OC_;
        size_t os_offset = start / OC_;
        args.acc = acc + start;
        args.dst = dst + os_offset * dst_os_stride_ + oc_offset;
        args.bias = bias + (g * jcp_.oc + oc_offset) * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * (g * jcp_.oc + oc_offset);
        args.nslope = nslope;
        args.sum_scale = sum_scale;
        args.signed_scale = signed_scale;
        args.len = end - start;
        args.oc_offset = oc_offset;
        ker_(&args);
    }
    else {
        // Fallback
        const size_t first_oc = start % OC_;
        const size_t last_oc = (end - 1) % OC_;
        const size_t first_os = start / OC_;
        const size_t last_os = (end - 1) / OC_;
        for (size_t os = first_os; os <= last_os; os++) {
            const size_t start_oc = (os == first_os) ? first_oc : 0;
            const size_t end_oc = (os == last_os) ? last_oc : OC_ - 1;
            for (size_t oc = start_oc; oc <= end_oc; oc++) {
                const size_t acc_off = os * jcp_.oc + oc;
                const size_t dst_off = os * dst_os_stride_ + oc;

                float d = (float)(acc[acc_off]);
                if (jcp_.signed_input)
                    d *= signed_scale;

                if (do_bias_)
                    d += get_bias(bias, g * jcp_.oc + oc,
                        bias_data_type_);

                d *= scales[(g * jcp_.oc + oc) * scale_idx_mult_];
                if (do_sum_)
                    d += sum_scale * dst[dst_off];
                if (do_relu_ && d < 0)
                    d *= nslope;
                dst[dst_off] = qz_a1b0<float, dst_data_t>()(d);
            }
        }
    }
};

template <data_type_t src_type, data_type_t dst_type>
void _gemm_x8s8s32x_convolution_fwd_t<src_type, dst_type>::
execute_forward_thr(const int ithr, const int nthr, const src_data_t *src_base,
        const wei_data_t *wei_base, const char *bia_base, dst_data_t *dst_base,
        const memory_tracking::grantor_t &scratchpad) const {
    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const auto src_md = memory_desc_wrapper(pd()->src_md());
    const size_t src_mb_stride = src_md.blk_off(1);
    const size_t src_g_stride = src_md.blk_off(0, 1) * jcp.ic;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto dst_md = memory_desc_wrapper(pd()->dst_md());
    const size_t dst_mb_stride = dst_md.blk_off(1);
    const size_t dst_g_stride = dst_md.blk_off(0, 1) * jcp.oc;

    const float *scales = pd()->attr()->output_scales_.scales_;

    const auto &post_ops = pd()->attr()->post_ops_;
    const bool do_sum = post_ops.contain(primitive_kind::sum, 0);
    const float sum_scale = do_sum ? post_ops.entry_[0].sum.scale : 0;

    float nslope = 0;
    for (int idx = 0; idx < post_ops.len_; ++idx) {
        const auto &e = post_ops.entry_[idx];
        if (e.is_relu(true, false)) {
            nslope = e.eltwise.alpha;
            break;
        }
    }

    auto col = scratchpad.get<uint8_t>(key_conv_gemm_col)
        + (ptrdiff_t)ithr * jcp.im2col_sz;
    src_data_t *__restrict imtr = scratchpad.get<src_data_t>(key_conv_gemm_imtr)
        + (ptrdiff_t)ithr * jcp.is * jcp.ic;
    auto acc = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
        + (ptrdiff_t)ithr * jcp.oh_block * jcp.ow_block * jcp.oc;

    const ptrdiff_t offset = (ptrdiff_t)jcp.ngroups * jcp.ks * jcp.ic * jcp.oc;
    const int32_t *_wei_comp = (const int32_t *)(wei_base + offset);

    int g{ 0 }, n{ 0 }, ohb{ 0 }, owb{ 0 };
    size_t start = 0, end = 0;

    const int nb_oh = div_up(jcp.oh, jcp.oh_block);
    const int nb_ow = div_up(jcp.ow, jcp.ow_block);
    const size_t work_amount = jcp.ngroups * jcp.mb * nb_oh * nb_ow;
    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups, ohb,
                nb_oh, owb, nb_ow);

    for (size_t iwork = start; iwork < end; ++iwork) {
        int oh = ohb * jcp.oh_block;
        int ow = owb * jcp.ow_block;
        const src_data_t *__restrict src = src_base + n * src_mb_stride
            + g * src_g_stride;
        const wei_data_t *__restrict wei = wei_base + g * wei_g_stride;
        dst_data_t *__restrict dst =
                dst_base + n * dst_mb_stride + g * dst_g_stride;
        const int32_t *wei_comp = _wei_comp + g * jcp.oc;
        const int h_step = nstl::min(jcp.oh_block, jcp.oh - oh);
        const int w_step = nstl::min(jcp.ow_block, jcp.ow - ow);

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::im2col_u8<src_data_t>(
                    jcp, src, imtr, col, oh, h_step, ow, w_step);

        const int M = jcp.oc;
        const int K = jcp.ks * jcp.ic;
        const int N = h_step * w_step;
        const int LDA = M * jcp.ngroups;
        const int LDB = jcp.im2col_sz ? N : K;
        const char *BT = jcp.im2col_sz ? "T" : "N";
        const int8_t off_a = 0, off_b = 0;
        const int32_t off_c = 0;
        const float onef = 1.0, zerof = 0.0;
        gemm_s8x8s32("N", BT, jcp.signed_input ? "C" : "F",
            &M, &N, &K, &onef, wei, &LDA, &off_a,
            jcp.im2col_sz ? col : (uint8_t *)src, &LDB, &off_b,
            &zerof, acc, &M, jcp.signed_input ? wei_comp : &off_c);

        auto wei_adj_scale =
            (wei_md.extra().flags | memory_extra_flags::scale_adjust)
            ? wei_md.extra().scale_adjust : 1.f;

        parallel(0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)N * jcp.oc, nthr, ithr, start, end);
            (*pp_ker_)(dst + (oh * jcp.ow + ow) * pp_ker_->dst_os_stride_,
                    acc, bia_base, scales, nslope, sum_scale,
                    1.f / wei_adj_scale, g, start, end);
        });

        nd_iterator_step(n, jcp.mb, g, jcp.ngroups, ohb, nb_oh,
                    owb, nb_ow);
    }
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::
execute_backward_data(const exec_ctx_t &ctx) const {
    auto diff_dst_base = CTX_IN_MEM(const diff_dst_data_t *, MKLDNN_ARG_DIFF_DST);
    auto wei_base = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bia_base = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto diff_src_base = CTX_OUT_MEM(diff_src_data_t *, MKLDNN_ARG_DIFF_SRC);

    auto scratchpad = this->scratchpad(ctx);

    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    parallel(jcp.nthr, [&](const int ithr, const int nthr) {
        execute_backward_data_thr(ithr, nthr, diff_dst_base, wei_base,
                bia_base, diff_src_base, scratchpad);
    });
}

template <data_type_t dst_type>
void _gemm_u8s8s32x_convolution_bwd_data_t<dst_type>::
execute_backward_data_thr(const int ithr, const int nthr,
        const diff_dst_data_t *diff_dst_base, const wei_data_t *wei_base,
        const char *bia_base, diff_src_data_t *diff_src_base,
        const memory_tracking::grantor_t &scratchpad) const
{
    const jit_gemm_conv_conf_t &jcp = this->pd()->jcp_;

    const auto diff_dst_md = memory_desc_wrapper(pd()->diff_dst_md());
    const size_t diff_dst_mb_stride = diff_dst_md.blk_off(1);
    const size_t diff_dst_g_stride = diff_dst_md.blk_off(0, 1) * jcp.oc;

    const auto wei_md = memory_desc_wrapper(pd()->weights_md(0));
    const size_t wei_g_stride = pd()->with_groups() ? wei_md.blk_off(1) : 0;

    const auto diff_src_md = memory_desc_wrapper(pd()->diff_src_md());
    const size_t diff_src_mb_stride = diff_src_md.blk_off(1);
    const size_t diff_src_g_stride = diff_src_md.blk_off(0, 1) * jcp.ic;
    const size_t diff_src_os_stride = diff_src_md.blk_off(0, 0, 0, 1);

    /* scale_idx_mult = 1 for per_oc scales and 0, otherwise */
    const int scale_idx_mult = pd()->attr()->output_scales_.mask_ == (1 << 1);
    const float *scales = pd()->attr()->output_scales_.scales_;
    const size_t work_amount = jcp.ngroups * jcp.mb;

    auto col = scratchpad.get<acc_data_t>(key_conv_gemm_col)
        + (ptrdiff_t)ithr * jcp.im2col_sz;
    auto acc = scratchpad.get<acc_data_t>(key_conv_int_dat_in_acc_dt)
        + (ptrdiff_t)ithr * jcp.is * jcp.ic;

    int n{0}, g{0};
    size_t start = 0, end = 0;

    balance211(work_amount, nthr, ithr, start, end);
    nd_iterator_init(start, n, jcp.mb, g, jcp.ngroups);

    for (size_t iwork = start; iwork < end; ++iwork) {
        const diff_dst_data_t *diff_dst = diff_dst_base
            + n * diff_dst_mb_stride + g * diff_dst_g_stride;
        const wei_data_t *wei = wei_base + g * wei_g_stride;
        diff_src_data_t *diff_src = diff_src_base + n * diff_src_mb_stride
            + g * diff_src_g_stride;

        const int M = jcp.ks * jcp.ic;
        const int N = jcp.os;
        const int K = jcp.oc;
        const int8_t off_a = 0, off_b = 0;
        const int32_t off_c = 0;
        const float onef = 1.0, zerof = 0.0;
        const int LD = K * jcp.ngroups;

        gemm_s8x8s32("T", "N", "F", &M, &N, &K, &onef,
                wei, &LD, &off_a, diff_dst, &LD, &off_b,
                &zerof, jcp.im2col_sz ? col : acc, &M, &off_c);

        if (jcp.im2col_sz)
            jit_gemm_convolution_utils::col2im_s32(jcp, col, acc);

        parallel_nd(jcp.is, jcp.ic, [&](int is, int ic) {
            float d = (float)acc[is * jcp.ic + ic];
            if (jcp.with_bias)
                d += get_bias(bia_base, g * jcp.ic + ic,
                        pd()->desc()->bias_desc.data_type);
            d *= scales[(g * jcp.ic + ic) * scale_idx_mult];
            const size_t diff_src_off = is * diff_src_os_stride + ic;
            diff_src[diff_src_off] =
                qz_a1b0<float, diff_src_data_t>()(d);
        });
        nd_iterator_step(n, jcp.mb, g, jcp.ngroups);
    }
}

using namespace data_type;

template struct _gemm_x8s8s32x_convolution_fwd_t<u8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<u8, u8>;

template struct _gemm_x8s8s32x_convolution_fwd_t<s8, f32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s32>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, s8>;
template struct _gemm_x8s8s32x_convolution_fwd_t<s8, u8>;

template struct _gemm_u8s8s32x_convolution_bwd_data_t<f32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s32>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<s8>;
template struct _gemm_u8s8s32x_convolution_bwd_data_t<u8>;
}
}
}
