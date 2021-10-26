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

#include "math_utils.hpp"
#include "mkldnn_thread.hpp"
#include "simple_q10n.hpp"

#include "gemm/gemm.hpp"
#include "gemm_x8s8s32x_inner_product.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace math;
using namespace format_tag;
using namespace memory_tracking::names;

template<data_type_t src_type, data_type_t dst_type>
gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type>::pp_kernel_t::pp_kernel_t(
        const pd_t *pd, bool dst_is_acc)
    : ker_(nullptr), OC_(pd->OC())
    , bias_data_type_(data_type::undef), bias_data_type_size_(0)
    , scale_idx_mult_(0), do_bias_(false), do_relu_(false)
{
    using namespace types;

    scale_idx_mult_ = (pd->attr()->output_scales_.mask_ == (1 << 1));

    auto &post_ops = pd->attr()->post_ops_;
    do_relu_ = post_ops.len_ == 1;
    do_bias_ = pd->with_bias();
    bias_data_type_ = pd->desc()->bias_desc.data_type;
    if (do_bias_) {
        assert(bias_data_type_ != data_type::undef);
        bias_data_type_size_ = data_type_size(bias_data_type_);
    }

    if (!mayiuse(avx512_core))
        // use fallback code for older CPUs since they do not have optimized
        // x8s8s32 GEMM anyways. The configuration variables above are used by
        // the fallback code.
        return;
    else
        generate();
}

template<data_type_t src_type, data_type_t dst_type>
void gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type>::pp_kernel_t::generate()
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
    Reg64 reg_rem_mask = r10;
    Opmask kreg_rem_mask = k1;
    Opmask kreg_relu_cmp = k2;

    const size_t vlen = cpu_isa_traits<avx512_common>::vlen / sizeof(float);

    Zmm vreg_zero = Zmm(0);
    Zmm vreg_scale = Zmm(1);
    Zmm vreg_nslope = Zmm(2);

    auto vreg_dst = [&](int idx) { return Zmm(3 + idx * 2 + 0); };
    auto vreg_bias = [&](int idx) { return Zmm(3 + idx * 2 + 1); };

    preamble();

#define PARAM_OFF(x) offsetof(ker_args, x)
    mov(reg_dst, ptr[reg_param + PARAM_OFF(dst)]);
    mov(reg_acc, ptr[reg_param + PARAM_OFF(acc)]);
    mov(reg_bias, ptr[reg_param + PARAM_OFF(bias)]);
    mov(reg_scales, ptr[reg_param + PARAM_OFF(scales)]);
    mov(reg_len, ptr[reg_param + PARAM_OFF(len)]);
    mov(reg_oc_offset, ptr[reg_param + PARAM_OFF(oc_offset)]);
    vbroadcastss(vreg_nslope, ptr[reg_param + PARAM_OFF(nslope)]);
    if (scale_idx_mult_ == 0)
        vbroadcastss(vreg_scale, dword[reg_scales]);
#undef PARAM_OFF

    if (do_relu_ || dst_type == data_type::u8)
        vxorps(vreg_zero, vreg_zero, vreg_zero);

    // Load accumulated value, convert to float, apply bias (if any), scaling,
    // and relu (if any); then convert to destination type and store
    auto compute = [&](size_t offset, int idx, bool apply_mask) {
        auto acc_addr = ptr[reg_acc + offset * sizeof(acc_data_t)];

        if (scale_idx_mult_ > 0) {
            assert(scale_idx_mult_ == 1);
            auto scale_addr = ptr[reg_scales + offset * sizeof(float)];
            auto vreg_scale_ = vreg_scale;
            if (apply_mask)
                vreg_scale_ = vreg_scale_ | kreg_rem_mask;
            vmovups(vreg_scale, scale_addr);
        }

        auto vreg_dst_ = vreg_dst(idx);
        if (apply_mask)
            vreg_dst_ = vreg_dst_ | kreg_rem_mask;
        vcvtdq2ps(vreg_dst_, acc_addr);

        if (do_bias_) {
            auto bias_addr = ptr[reg_bias + offset * bias_data_type_size_];
            auto vreg_bias_ = vreg_bias(idx);
            if (apply_mask)
                vreg_bias_ = vreg_bias_ | kreg_rem_mask;

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
        if (do_relu_) {
            vcmpps(kreg_relu_cmp, vreg_dst(idx), vreg_zero, _cmp_lt_os);
            vmulps(vreg_dst(idx) | kreg_relu_cmp, vreg_dst(idx), vreg_nslope);
        }

        if (dst_type == data_type::u8)
            vmaxps(vreg_dst(idx), vreg_dst(idx), vreg_zero);

        if (dst_type != data_type::f32) {
            vcvtps2dq(vreg_dst(idx), vreg_dst(idx));
        }

        auto dst_addr = ptr[reg_dst + offset * sizeof(dst_data_t)];
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

    // Rewind pointers that point to data that is indixed by output channel
    // (bias or per-oc scaling factors)
    auto rewind_ptrs = [&]() {
        if (do_bias_)
            sub(reg_bias, OC_ * bias_data_type_size_);
        if (scale_idx_mult_) {
            assert(scale_idx_mult_ == 1);
            sub(reg_scales, OC_ * sizeof(float));
        }
    };

    //      <-------------------- OC ------------------------------->
    //
    // ^    +....................+----------------------------------+
    // |    :   not accessed     |          Prologue loop           |
    // |    +--------------------+----------------------------------+
    //      |                                                       |
    // M    |                 Main loop (unrolled)                  |
    // B    |                                                       |
    //      +--------------------------------+----------------------+
    // |    |       Epilogue loop            |      not accessed    :
    // v    +--------------------------------+......................+

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
        jle(prologue_loop_tail, T_NEAR); // Skips for reg_tmp == 16 too (?)
        L(prologue_loop); {
            compute(0, 0, false);
            advance_ptrs_imm(vlen);
            sub(reg_tmp, vlen);
            cmp(reg_tmp, vlen);
            jge(prologue_loop, T_NEAR);
        }

        L(prologue_loop_tail);
        mov(reg_rem_mask, 1);
        shl(reg_rem_mask, cl); // cl == reg_tmp because reg_tmp <= vlen here
        sub(reg_rem_mask, 1);
        jz(prologue_loop_end, T_NEAR);

        kmovq(kreg_rem_mask, reg_rem_mask);
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
            size_t def_unroll = 4;
            size_t max_unroll = 13;

            size_t OC_loop, OC_tail;
            if (OC_ < max_unroll * vlen) {
                // Fully unroll small loops
                OC_loop = 0;
                OC_tail = OC_;
            } else {
                OC_loop = vlen * def_unroll;
                OC_tail = OC_ % OC_loop;
            }

            assert(!!OC_loop || !!OC_tail);

            if (OC_tail % vlen) {
                int vlen_tail = OC_tail % vlen;
                unsigned tail_mask = (1 << vlen_tail) - 1;
                mov(reg_tmp, tail_mask);
                kmovq(kreg_rem_mask, reg_tmp);
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
        jle(epilogue_loop_tail, T_NEAR); // Skips for reg_len == 16 (?)
        L(epilogue_loop); {
            compute(0, 0, false);
            sub(reg_len, vlen);
            advance_ptrs_imm(vlen);
            cmp(reg_len, vlen);
            jge(epilogue_loop, T_NEAR);
        }

        L(epilogue_loop_tail);
        mov(reg_tmp, reg_len); // reg_tmp is rcx, and we need cl for the shift
        mov(reg_rem_mask, 1);
        shl(reg_rem_mask, cl); // reg_tmp == rcx and reg_tail < vlen == 16
        sub(reg_rem_mask, 1);
        jz(epilogue_end, T_NEAR);
        kmovq(kreg_rem_mask, reg_rem_mask);
        compute(0, 0, true);
    }

    L(epilogue_end);

    postamble();

    ker_ = getCode<decltype(ker_)>();
}

template<data_type_t src_type, data_type_t dst_type>
void gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type>::pp_kernel_t::operator ()(
        dst_data_t *dst, const acc_data_t *acc,
        const char *bias, const float *scales, float nslope,
        size_t start, size_t end)
{
    using math::get_bias;

    if (end <= start)
        return;

    if (ker_) {
        // JIT
        ker_args args;
        size_t oc_offset = start % OC_;
        args.dst = dst + start;
        args.acc = acc + start;
        args.bias = bias + oc_offset * bias_data_type_size_;
        args.scales = scales + scale_idx_mult_ * oc_offset;
        args.nslope = nslope;
        args.len = end - start;
        args.oc_offset = oc_offset;
        ker_(&args);
    } else {
        // Fallback
        size_t oc = start % OC_;
        for (size_t i = start; i < end; i++) {
            float d = (float)acc[i];
            float b = get_bias(bias, oc, bias_data_type_);
            d = d + b;
            d *= scales[oc * scale_idx_mult_];
            if (do_relu_ && d < 0)
                d *= nslope;
            dst[i] = qz_a1b0<float, dst_data_t>()(d);
            oc = (oc == OC_ - 1) ? 0 : oc + 1;
        }
    }
};

template <data_type_t src_type, data_type_t dst_type>
void gemm_x8s8s32x_inner_product_fwd_t<src_type, dst_type>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src = CTX_IN_MEM(const src_data_t *, MKLDNN_ARG_SRC);
    auto weights = CTX_IN_MEM(const wei_data_t *, MKLDNN_ARG_WEIGHTS);
    auto bias = CTX_IN_MEM(const char *, MKLDNN_ARG_BIAS);
    auto dst = CTX_OUT_MEM(dst_data_t *, MKLDNN_ARG_DST);

    const int MB = pd()->MB();
    const int OC = pd()->OC();

    bool wei_tr = memory_desc_matches_one_of_tag(
            *pd()->weights_md(), oiw, oihw, oidhw, oi);

    const int M = OC;
    const int N = MB;
    const int K = pd()->IC_total_padded();
    const int8_t off_a = 0, off_b = 0;
    const int32_t off_c = 0;

    const float *scales = pd()->attr()->output_scales_.scales_;

    const auto &post_ops = pd()->attr()->post_ops_;
    const bool do_relu = post_ops.len_ == 1;
    const float nslope = do_relu ? post_ops.entry_[0].eltwise.alpha : 0.f;

    acc_data_t *acc = pd()->dst_is_acc_
        ? (acc_data_t *)dst
        : scratchpad(ctx).template get<acc_data_t>(key_iprod_int_dat_in_acc_dt);

    const float onef = 1.0, zerof = 0.0;
    gemm_s8x8s32(wei_tr ? "T" : "N", "N", "F", &M, &N, &K, &onef, weights,
            wei_tr ? &K : &M, &off_a, src, &K, &off_b, &zerof, acc, &M, &off_c);

    if (!pd()->attr()->has_default_values() || !pd()->dst_is_acc_
            || pd()->with_bias()) {
        const bool force_sequential = MB * OC < 2000;
        parallel(force_sequential ? 1 : 0, [&](int ithr, int nthr) {
            size_t start, end;
            balance211((size_t)OC * MB, nthr, ithr, start, end);
            (*pp_kernel_)(dst, acc, bias, scales, nslope, start, end);
        });
    }
}

using namespace data_type;

template struct gemm_x8s8s32x_inner_product_fwd_t<u8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<u8, u8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, f32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s32>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, s8>;
template struct gemm_x8s8s32x_inner_product_fwd_t<s8, u8>;

}
}
}
