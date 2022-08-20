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

#include "c_types_map.hpp"
#include "nstl.hpp"
#include "utils.hpp"

#include "jit_uni_lrn_kernel_f32.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

//////////////////////////////////////////////////////////////////////////////
// forward kernel
template<cpu_isa_t isa>
void jit_uni_lrn_fwd_kernel_f32<isa>::within_body(
        int hoff, int Hoff, int woff, int Woff, int stride,
        Xbyak::Ymm ysum, Xbyak::Ymm ydst, Xbyak::Ymm ytmp, Xbyak::Ymm ysum2,
        prop_kind_t pk)
{
    vxorps(ysum, ysum, ysum);
    for (int i = hoff; i <= Hoff; ++i)
    {
        for (int j = woff; j <= Woff; ++j)
        {
            if (i == 0 && j == 0)
            {
                vmovups(ydst, ptr[src]);
                vfmadd231ps(ysum, ydst, ydst);
            }
            else
            {
                vmovups(ytmp, ptr[src + (i*stride + j)*VECTOR_LENGTH*4]);
                vfmadd231ps(ysum, ytmp, ytmp);
            }
        }
    }
    vfmadd132ps(ysum, yk, yalpha); // ysum <- ysum*yalpha+yk
    vmovaps(ytmp, ysum);
    if (pk != prop_kind::forward_inference)
        vmovups(ptr[scratch], ytmp);
    vmulps(ysum2, ysum, ysum);
    vmulps(ysum, ysum, ysum2); // ysum = (ysum*yalpha+yk)^3;
    vsqrtps(ysum, ysum);
    vsqrtps(ysum, ysum); // ysum = (ysum*yalpha+yk)^0.75
    vdivps(ydst, ydst, ysum); // ydst <- ydst / ysum
    vmovups(ptr[dst], ydst);
    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);
}

template<cpu_isa_t isa>
void jit_uni_lrn_fwd_kernel_f32<isa>::within_body_sse42(
    int hoff, int Hoff, int woff, int Woff, int stride, prop_kind_t pk)
{
    Xbyak::Xmm xtmp_lo = xmm12;
    Xbyak::Xmm xtmp_hi = xmm13;
    Xbyak::Xmm xsum_lo = xmm8;
    Xbyak::Xmm xsum_hi = xmm9;
    Xbyak::Xmm xdst_lo = xmm10;
    Xbyak::Xmm xdst_hi = xmm11;
    Xbyak::Xmm xsum2_lo = xmm14;
    Xbyak::Xmm xsum2_hi = xmm15;

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);
    for (int i = hoff; i <= Hoff; ++i)
    {
        for (int j = woff; j <= Woff; ++j)
        {
            if (i == 0 && j == 0)
            {
                movups(xdst_lo, ptr[src]);
                movups(xdst_hi, ptr[src + 4 * sizeof(float)]);
                mulps(xdst_lo, xdst_lo);
                mulps(xdst_hi, xdst_hi);
                addps(xsum_lo, xdst_lo);
                addps(xsum_hi, xdst_hi);
            }
            else
            {
                movups(xtmp_lo, ptr[src + (i*stride + j)*VECTOR_LENGTH * 4]);
                movups(xtmp_hi, ptr[src + (i*stride + j)*VECTOR_LENGTH * 4 + 4 * sizeof(float)]);
                mulps(xtmp_lo, xtmp_lo);
                mulps(xtmp_hi, xtmp_hi);
                addps(xsum_lo, xtmp_lo);
                addps(xsum_hi, xtmp_hi);
            }
        }
    }
    mulps(xsum_lo, xalpha);
    mulps(xsum_hi, xalpha);
    addps(xsum_lo, xk);
    addps(xsum_hi, xk); // xsum <- xsum*xalpha+xk
    movaps(xtmp_lo, xsum_lo);
    movaps(xtmp_hi, xsum_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch], xtmp_lo);
        movups(ptr[scratch + 4 * sizeof(float)], xtmp_hi);
    }
    movaps(xsum2_lo, xsum_lo);
    movaps(xsum2_hi, xsum_hi);
    mulps(xsum2_lo, xsum_lo);
    mulps(xsum2_hi, xsum_hi);
    mulps(xsum_lo, xsum2_lo);
    mulps(xsum_hi, xsum2_hi); // xsum = (xsum*xalpha+xk)^3;

    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi);
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi); // xsum = (xsum*xalpha+xk)^0.75

    movups(xdst_lo, ptr[src]);
    movups(xdst_hi, ptr[src + 4 * sizeof(float)]);
    divps(xdst_lo, xsum_lo);
    divps(xdst_hi, xsum_hi); // xdst <- xdst / xsum

    movups(ptr[dst], xdst_lo);
    movups(ptr[dst + 4 * sizeof(float)], xdst_hi);
    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);
}

template <cpu_isa_t isa>
jit_uni_lrn_fwd_kernel_f32<isa>::jit_uni_lrn_fwd_kernel_f32(
        const struct nchw8c_within &J,
        float A,
        float K,
        prop_kind_t pk,
        void *code_ptr,
        size_t code_size)
        : jit_generator(code_ptr, code_size)
        , alpha(A), k(K)
{
    Xbyak::Reg64 h = r9;
    Xbyak::Reg64 w = r10;
    Vmm ysum = Vmm(isa == avx2 ? 9 : 9);
    Vmm ysum2 = Vmm(isa == avx2 ? 10 : 10);
    Vmm ydst = Vmm(isa == avx2 ? 11 : 11);
    Vmm ytmp = Vmm(isa == avx2 ? 12 : 12);

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);

    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    if (isa == avx2) {
        vbroadcastss(yalpha, xalpha);
    } else {
        shufps(xalpha, xalpha, 0);
    }

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    if (isa == avx2) {
        vbroadcastss(yk, xk);
    } else {
        shufps(xk, xk, 0);
    }

    int s2 = (J.size - 1) / 2, S2 = J.size - s2 - 1;

    for (int i = 0; i < s2; ++i)
    {
        Label label_t;
        for (int j = 0; j < s2; ++j) {
            if (isa == avx2) {
                within_body(-i, S2, -j, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
            }
            else {
                within_body_sse42(-i, S2, -j, S2, J.W, pk);
            }
        }
        mov(w, J.W - J.size + 1);
        L(label_t);
        if (isa == avx2) {
            within_body(-i, S2, -s2, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
        } else {
            within_body_sse42(-i, S2, -s2, S2, J.W, pk);
        }
        dec(w);
        cmp(w, 0);
        jne(label_t, T_NEAR);
        for (int j = J.W - S2; j < J.W; ++j) {
            if (isa == avx2) {
                within_body(-i, S2, -s2, J.W - 1 - j, J.W,
                    ysum, ydst, ytmp, ysum2, pk);
            } else {
                within_body_sse42(-i, S2, -s2, J.W - 1 - j, J.W, pk);
            }
        }
    }

    mov(h, J.H - J.size + 1);
    Label lrn_loop_h;
    L(lrn_loop_h);
    for (int j = 0; j < s2; ++j) {
        if (isa == avx2) {
            within_body(-s2, S2, -j, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
        } else {
            within_body_sse42(-s2, S2, -j, S2, J.W, pk);
        }
    }
    mov(w, J.W - J.size + 1);
    Label lrn_loop_w;
    L(lrn_loop_w);
    if (isa == avx2) {
        within_body(-s2, S2, -s2, S2, J.W, ysum, ydst, ytmp, ysum2, pk);
    } else {
        within_body_sse42(-s2, S2, -s2, S2, J.W, pk);
    }
    dec(w);
    cmp(w, 0);
    jne(lrn_loop_w, T_NEAR);
    for (int j = J.W - S2; j < J.W; ++j) {
        if (isa == avx2) {
            within_body(-s2, S2, -s2, J.W - 1 - j, J.W,
                ysum, ydst, ytmp, ysum2, pk);
        } else {
            within_body_sse42(-s2, S2, -s2, J.W - 1 - j, J.W, pk);
        }
    }
    dec(h);
    cmp(h, 0);
    jne(lrn_loop_h, T_NEAR);

    for (int i = J.H - S2; i < J.H; ++i)
    {
        for (int j = 0; j < s2; ++j) {
            if (isa == avx2) {
                within_body(-s2, J.H - 1 - i, -j, S2, J.W,
                    ysum, ydst, ytmp, ysum2, pk);
            } else {
                within_body_sse42(-s2, J.H - 1 - i, -j, S2, J.W, pk);
            }
        }

        mov(w, J.W - J.size + 1);
        Label label_b;
        L(label_b);
        if (isa == avx2) {
            within_body(-s2, J.H - 1 - i, -s2, S2, J.W,
                ysum, ydst, ytmp, ysum2, pk);
        } else {
            within_body_sse42(-s2, J.H - 1 - i, -s2, S2, J.W, pk);
        }
        dec(w);
        cmp(w, 0);
        jne(label_b, T_NEAR);

        for (int j = J.W - S2; j < J.W; ++j) {
            if (isa == avx2) {
                within_body(-s2, J.H - 1 - i, -s2, J.W - 1 - j, J.W,
                    ysum, ydst, ytmp, ysum2, pk);
            } else {
                within_body_sse42(-s2, J.H - 1 - i, -s2, J.W - 1 - j, J.W, pk);
            }
        }
    }

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                this->getCode()));
}

template<>
jit_uni_lrn_fwd_kernel_f32<avx2>::jit_uni_lrn_fwd_kernel_f32(
        const struct nchw8c_across &J,
        float A,
        float K,
        prop_kind_t pk,
        void *code_ptr,
        size_t code_size)
        : jit_generator(code_ptr, code_size)
        , alpha(A), k(K)
{
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r9;
    Xbyak::Xmm xsrc_prev = xmm2;
    Xbyak::Ymm ysrc = ymm3;
    Xbyak::Ymm yc = ymm3;
    Xbyak::Xmm xsrc_next = xmm4;
    Xbyak::Ymm ya = ymm5;
    Xbyak::Ymm yb = ymm6;
    Xbyak::Ymm yd = ymm7;
    Xbyak::Ymm ye = ymm8;
    Xbyak::Ymm ysum = ymm9;
    Xbyak::Ymm ysum2 = ymm10;
    Xbyak::Ymm ydst = ymm11;
    Xbyak::Ymm ybase = ymm12;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);
    sub(t, 64);
    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    vbroadcastss(yalpha, xalpha);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    vbroadcastss(yk, xk);

    if (J.version == -1)
    {
        vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        vmovups(ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1)
    {
        vxorps(xsrc_next, xsrc_next, xsrc_next);
        vmovups(ptr[t + 48], xsrc_next);
    }

    mov(hw, J.H*J.W);

    Label lrn_loop;
    L(lrn_loop);

    if (J.version != -1) vmovups(xsrc_prev, ptr[src - J.H*J.W * 32 + 16]);
    vmovups(ysrc, ptr[src]);
    if (J.version != +1) vmovups(xsrc_next, ptr[src + J.H*J.W * 32]);

    if (J.version != -1) vmovups(ptr[t + 0], xsrc_prev);
    vmovups(ptr[t + 16], ysrc);
    if (J.version != +1) vmovups(ptr[t + 48], xsrc_next);

    vmovups(ya, ptr[t + 16 - 8]);
    vmovups(yb, ptr[t + 16 - 4]);
    vmovups(yd, ptr[t + 16 + 4]);
    vmovups(ye, ptr[t + 16 + 8]);
    vmulps(ysum, yc, yc);
    vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya*ya
    vfmadd231ps(ysum, yb, yb);
    vfmadd231ps(ysum, yd, yd);
    vfmadd231ps(ysum, ye, ye);
    vfmadd132ps(ysum, yk, yalpha); // ysum <- ysum*yalpha+yk

    vmovaps(ybase, ysum);
    if (pk != prop_kind::forward_inference)
        vmovups(ptr[scratch], ybase);
    vmulps(ysum2, ysum, ysum);
    vmulps(ysum, ysum, ysum2); // ysum = ybase^3;
    vsqrtps(ysum, ysum);
    vsqrtps(ysum, ysum); // ysum = ybase^0.75
    vdivps(ydst, ysrc, ysum); // ydst = ysrc / ysum
    vmovups(ptr[dst], ydst);

    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);
    dec(hw);
    cmp(hw, 0);
    jne(lrn_loop, T_NEAR);

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                this->getCode()));
}

template<>
jit_uni_lrn_fwd_kernel_f32<sse42>::jit_uni_lrn_fwd_kernel_f32(
    const struct nchw8c_across &J,
    float A,
    float K,
    prop_kind_t pk,
    void *code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha(A), k(K)
{
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r9;

    Xbyak::Xmm xsrc_lo = xmm2;
    Xbyak::Xmm xsrc_hi = xmm3;
    Xbyak::Xmm xc_lo = xmm4;
    Xbyak::Xmm xc_hi = xmm5;
    Xbyak::Xmm xsum_lo = xc_lo;
    Xbyak::Xmm xsum_hi = xc_hi;
    Xbyak::Xmm xsrc_prev = xmm6;
    Xbyak::Xmm xsrc_next = xmm7;
    Xbyak::Xmm xa_lo = xmm8;
    Xbyak::Xmm xa_hi = xmm9;
    Xbyak::Xmm xb_lo = xmm10;
    Xbyak::Xmm xb_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;
    Xbyak::Xmm xe_lo = xmm14;
    Xbyak::Xmm xe_hi = xmm15;
    Xbyak::Xmm xbase_lo = xmm14;
    Xbyak::Xmm xbase_hi = xmm15;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);
    sub(t, 64);
    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    shufps(xalpha, xalpha, 0);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    shufps(xk, xk, 0);

    if (J.version == -1)
    {
        xorps(xsrc_prev, xsrc_prev);
        movups(ptr[t + 0], xsrc_prev);
    }
    if (J.version == +1)
    {
        xorps(xsrc_next, xsrc_next);
        movups(ptr[t + 48], xsrc_next);
    }

    mov(hw, J.H*J.W);
    Label lrn_loop;
    L(lrn_loop);

    if (J.version != -1) movups(xsrc_prev, ptr[src - J.H*J.W * 32 + 16]);
    movups(xsrc_lo, ptr[src]);
    movups(xsrc_hi, ptr[src + 4 * sizeof(float)]);
    if (J.version != +1) movups(xsrc_next, ptr[src + J.H*J.W * 32]);

    if (J.version != -1) movups(ptr[t + 0], xsrc_prev);
    movups(ptr[t + 16], xsrc_lo);
    movups(ptr[t + 16 + 4 * sizeof(float)], xsrc_hi);
    if (J.version != +1) movups(ptr[t + 48], xsrc_next);

    movups(xa_lo, ptr[t + 16 - 8]);
    movups(xa_hi, ptr[t + 16 - 8 + 4 * sizeof(float)]);
    movups(xb_lo, ptr[t + 16 - 4]);
    movups(xb_hi, ptr[t + 16 - 4 + 4 * sizeof(float)]);
    movups(xd_lo, ptr[t + 16 + 4]);
    movups(xd_hi, ptr[t + 16 + 4 + 4 * sizeof(float)]);
    movups(xe_lo, ptr[t + 16 + 8]);
    movups(xe_hi, ptr[t + 16 + 8 + 4 * sizeof(float)]);
    movaps(xc_lo, xsrc_lo);
    movaps(xc_hi, xsrc_hi);
    mulps(xsum_lo, xc_lo);
    mulps(xsum_hi, xc_hi);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi); // xsum <- xsum + xa*xa
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi);
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    mulps(xsum_lo, xalpha);
    mulps(xsum_hi, xalpha);
    addps(xsum_lo, xk);
    addps(xsum_hi, xk); // xsum <- xsum*xalpha+xk

    movaps(xbase_lo, xsum_lo);
    movaps(xbase_hi, xsum_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch], xbase_lo);
        movups(ptr[scratch + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xsum_lo, xsum_lo);
    mulps(xsum_hi, xsum_hi);
    mulps(xsum_lo, xbase_lo);
    mulps(xsum_hi, xbase_hi); // xsum = xbase^3;
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi);
    sqrtps(xsum_lo, xsum_lo);
    sqrtps(xsum_hi, xsum_hi); // xsum = xbase^0.75
    divps(xsrc_lo, xsum_lo);
    divps(xsrc_hi, xsum_hi); // xdst = xsrc / xsum
    movups(ptr[dst], xsrc_lo);
    movups(ptr[dst + 4 * sizeof(float)], xsrc_hi);

    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);
    dec(hw);
    cmp(hw, 0);
    jne(lrn_loop, T_NEAR);

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
        this->getCode()));
}

template<>
jit_uni_lrn_fwd_kernel_f32<avx2>::jit_uni_lrn_fwd_kernel_f32(
    const struct nhwc_across &J,
    float A,
    float K,
    prop_kind_t pk,
    void *code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha(A), k(K)
{
    static const uint32_t mask[] = {
        0, 0, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0x80000000, 0, 0
    };

    Xbyak::Reg64 c = r9;
    Xbyak::Ymm ya = ymm2;
    Xbyak::Ymm yb = ymm3;
    Xbyak::Ymm yc = ymm4;
    Xbyak::Ymm yd = ymm5;
    Xbyak::Ymm ye = ymm6;
    Xbyak::Ymm ysum = ymm7;
    Xbyak::Ymm ydst = ymm8;
    Xbyak::Ymm ybase = ymm9;
    Xbyak::Ymm ymask = ymm10;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);
    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    vbroadcastss(yalpha, xalpha);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    vbroadcastss(yk, xk);

    vxorps(ysum, ysum, ysum);

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[0]));
    vmovups(ymask, ptr[imm_addr64]);
    vmaskmovps(ya, ymask, ptr[src - 8]);
    vfmadd231ps(ysum, ya, ya); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[1]));
    vmovups(ymask, ptr[imm_addr64]);
    vmaskmovps(yb, ymask, ptr[src - 4]);
    vfmadd231ps(ysum, yb, yb);

    mov(c, J.C / 8 - 1);
    Label lrn_loop;
    L(lrn_loop);

    vmovups(yc, ptr[src]);
    vmovups(yd, ptr[src + 4]);
    vmovups(ye, ptr[src + 8]);
    vfmadd231ps(ysum, yc, yc);
    vfmadd231ps(ysum, yd, yd);
    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk, yalpha); // ydst <- ysum*yalpha+yk

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference)
        vmovups(ptr[scratch], ybase);
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yk)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yk)^0.75

    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yk)^0.75
    vmovups(ptr[dst], ydst);

    vxorps(ysum, ysum, ysum);

    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);

    vmovups(ya, ptr[src - 8]);
    vfmadd231ps(ysum, ya, ya);
    vmovups(yb, ptr[src - 4]);
    vfmadd231ps(ysum, yb, yb);

    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    vmovups(yc, ptr[src]);
    vfmadd231ps(ysum, yc, yc);

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[2]));
    vmovups(ymask, ptr[imm_addr64]);
    vmaskmovps(yd, ymask, ptr[src + 4]);
    vfmadd231ps(ysum, yd, yd); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[3]));
    vmovups(ymask, ptr[imm_addr64]);
    vmaskmovps(ye, ymask, ptr[src + 8]);
    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk, yalpha); // ydst <- ysum*yalpha+yk

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference)
        vmovups(ptr[scratch], ybase);
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yk)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yk)^0.75
    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yk)^0.75

    vmovups(ptr[dst], ydst);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                this->getCode()));
}

template<>
jit_uni_lrn_fwd_kernel_f32<sse42>::jit_uni_lrn_fwd_kernel_f32(
    const struct nhwc_across &J,
    float A,
    float K,
    prop_kind_t pk,
    void *code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha(A), k(K)
{
    static const uint32_t mask[] = {
        0, 0, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0xffffffff, 0, 0
    };

    static uint32_t store[] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    };
    Xbyak::Reg64 c = r9;

    Xbyak::Xmm xdst_lo = xmm0;
    Xbyak::Xmm xdst_hi = xmm1;
    Xbyak::Xmm xa_lo = xmm2;
    Xbyak::Xmm xa_hi = xmm3;
    Xbyak::Xmm xb_lo = xmm2;
    Xbyak::Xmm xb_hi = xmm3;
    Xbyak::Xmm xc_lo = xmm4;
    Xbyak::Xmm xc_hi = xmm5;
    Xbyak::Xmm xd_lo = xmm6;
    Xbyak::Xmm xd_hi = xmm7;
    Xbyak::Xmm xe_lo = xmm8;
    Xbyak::Xmm xe_hi = xmm9;
    Xbyak::Xmm xsum_lo = xmm10;
    Xbyak::Xmm xsum_hi = xmm11;
    Xbyak::Xmm xmask_lo = xmm12;
    Xbyak::Xmm xmask_hi = xmm13;
    Xbyak::Xmm xbase_lo = xmm14;
    Xbyak::Xmm xbase_hi = xmm15;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);
    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    shufps(xalpha, xalpha, 0);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    shufps(xk, xk, 0);

    mov(store_addr, reinterpret_cast<size_t>(&store[0]));
    and_(store_addr, -15);
    movups(ptr[store_addr], xalpha);
    movups(ptr[store_addr + 4 * sizeof(float)], xk);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[0]));
    movups(xmask_lo, ptr[imm_addr64]);
    movups(xmask_hi, ptr[imm_addr64 + 4 * sizeof(float)]);
    movups(xa_lo, ptr[src - 8]);
    movups(xa_hi, ptr[src - 8 + 4 * sizeof(float)]);
    andps(xa_lo, xmask_lo);
    andps(xa_hi, xmask_hi);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[1]));
    movups(xmask_lo, ptr[imm_addr64]);
    movups(xmask_hi, ptr[imm_addr64 + 4 * sizeof(float)]);
    movups(xb_lo, ptr[src - 4]);
    movups(xb_hi, ptr[src - 4 + 4 * sizeof(float)]);
    andps(xb_lo, xmask_lo);
    andps(xb_hi, xmask_hi);
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);

    mov(c, J.C / 8 - 1);
    Label lrn_loop;
    L(lrn_loop);

    movups(xc_lo, ptr[src]);
    movups(xc_hi, ptr[src + 4 * sizeof(float)]);
    movups(xd_lo, ptr[src + 4]);
    movups(xd_hi, ptr[src + 4 + 4 * sizeof(float)]);
    movups(xe_lo, ptr[src + 8]);
    movups(xe_hi, ptr[src + 8 + 4 * sizeof(float)]);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi);
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    movaps(xdst_lo, xsum_lo);
    movaps(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha+xk
    mulps(xdst_lo, ptr[store_addr]);
    mulps(xdst_hi, ptr[store_addr]);
    addps(xdst_lo, ptr[store_addr + 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr + 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch], xbase_lo);
        movups(ptr[scratch + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha+xk)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha+xk)^0.75

    movups(xc_lo, ptr[src]);
    movups(xc_hi, ptr[src + 4 * sizeof(float)]);
    divps(xc_lo, xdst_lo);
    divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha+xk)^0.75
    movups(ptr[dst], xc_lo);
    movups(ptr[dst + 4 * sizeof(float)], xc_hi);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);

    add(src, 32);
    add(dst, 32);
    if (pk != prop_kind::forward_inference)
        add(scratch, 32);

    movups(xa_lo, ptr[src - 8]);
    movups(xa_hi, ptr[src - 8 + 4 * sizeof(float)]);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    addps(xsum_lo, xa_lo);
    addps(xsum_hi, xa_hi);
    movups(xb_lo, ptr[src - 4]);
    movups(xb_hi, ptr[src - 4 + 4 * sizeof(float)]);
    mulps(xb_lo, xb_lo);
    mulps(xb_hi, xb_hi);
    addps(xsum_lo, xb_lo);
    addps(xsum_hi, xb_hi);

    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    movups(xc_lo, ptr[src]);
    movups(xc_hi, ptr[src + 4 * sizeof(float)]);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[2]));
    movups(xmask_lo, ptr[imm_addr64]);
    movups(xmask_hi, ptr[imm_addr64 + 4 * sizeof(float)]);
    movups(xd_lo, ptr[src + 4]);
    movups(xd_hi, ptr[src + 4 + 4 * sizeof(float)]);
    andps(xd_lo, xmask_lo);
    andps(xd_hi, xmask_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    mov(imm_addr64, reinterpret_cast<size_t>(&mask[3]));
    movups(xmask_lo, ptr[imm_addr64]);
    movups(xmask_hi, ptr[imm_addr64 + 4 * sizeof(float)]);
    movups(xe_lo, ptr[src + 8]);
    movups(xe_hi, ptr[src + 8 + 4 * sizeof(float)]);
    andps(xe_lo, xmask_lo);
    andps(xe_hi, xmask_hi);
    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    movups(xdst_lo, xsum_lo);
    movups(xdst_hi, xsum_hi);
    // xdst <- xsum*xalpha+xk
    mulps(xdst_lo, ptr[store_addr]);
    mulps(xdst_hi, ptr[store_addr]);
    addps(xdst_lo, ptr[store_addr + 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr + 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference) {
        movups(ptr[scratch], xbase_lo);
        movups(ptr[scratch + 4 * sizeof(float)], xbase_hi);
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha+xk)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha+xk)^0.75
    movups(xc_lo, ptr[src]);
    movups(xc_hi, ptr[src + 4 * sizeof(float)]);
    divps(xc_lo, xdst_lo);
    divps(xc_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha+xk)^0.75

    movups(ptr[dst], xc_lo);
    movups(ptr[dst + 4 * sizeof(float)], xc_hi);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
        this->getCode()));
}

template<>
void jit_uni_lrn_fwd_kernel_f32<sse42>::nchw_body(
    int tail, int HW, prop_kind_t pk,
    Xbyak::Ymm ymask,
    Xbyak::Ymm ya,
    Xbyak::Ymm yb,
    Xbyak::Ymm yc,
    Xbyak::Ymm yd,
    Xbyak::Ymm ye,
    Xbyak::Ymm ysum) {}

template<>
void jit_uni_lrn_fwd_kernel_f32<avx2>::nchw_body(
    int tail, int HW, prop_kind_t pk,
    Xbyak::Ymm ymask,
    Xbyak::Ymm ya,
    Xbyak::Ymm yb,
    Xbyak::Ymm yc,
    Xbyak::Ymm yd,
    Xbyak::Ymm ye,
    Xbyak::Ymm ysum)
{
    Xbyak::Ymm ydst = ymm14;
    Xbyak::Ymm ybase = ymm15;

    vfmadd231ps(ysum, ye, ye);

    vmovups(ydst, ysum);
    vfmadd132ps(ydst, yk, yalpha); // ydst <- ysum*yalpha+yk

    vmovaps(ybase, ydst);
    if (pk != prop_kind::forward_inference)
    {
        if (tail != 0)
            vmaskmovps(ptr[scratch], ymask, ybase);
        else
            vmovups(ptr[scratch], ybase);
    }
    vmulps(ydst, ydst, ydst);
    vmulps(ydst, ydst, ybase); // ydst = (ysum*yalpha+yk)^3;
    vsqrtps(ydst, ydst);
    vsqrtps(ydst, ydst); // ydst = (ysum*yalpha+yk)^0.75
    vdivps(ydst, yc, ydst); // ydst = ysrc / (ysum*yalpha+yk)^0.75

    if (tail != 0)
        vmaskmovps(ptr[dst], ymask, ydst);
    else
        vmovups(ptr[dst], ydst);


    vfnmadd231ps(ysum, ya, ya);
    vmovups(ya, yb);
    vmovups(yb, yc);
    vmovups(yc, yd);
    vmovups(yd, ye);
}

template<>
void jit_uni_lrn_fwd_kernel_f32<avx2>::nchw_tail_sse42(
    int tail, Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi)
{}

template<>
void jit_uni_lrn_fwd_kernel_f32<sse42>::nchw_tail_sse42(
    int tail, Xbyak::Reg64 reg_dst, Xbyak::Xmm xtail_lo, Xbyak::Xmm xtail_hi)
{
    Xbyak::Xmm xmm_tmp = xmm10;
    movaps(xmm_tmp, xtail_lo);
    size_t offset = 0;

    if (tail > 4) {
        movups(ptr[reg_dst], xtail_lo);
        movaps(xmm_tmp, xtail_hi);
        offset += 4 * sizeof(float);
        tail -= 4;
    }
    movss(ptr[reg_dst + offset], xmm_tmp);
    for (int i = 1; i < tail; i++)
    {
        psrldq(xmm_tmp, 4);
        movss(ptr[reg_dst + offset + i * sizeof(float)], xmm_tmp);
    }
}

template<>
void jit_uni_lrn_fwd_kernel_f32<sse42>::nchw_body_sse42(
    int tail, int HW, prop_kind_t pk,
    Xbyak::Xmm xmask_lo, Xbyak::Xmm xmask_hi,
    Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi,
    Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi)
{
    Xbyak::Xmm xdst_lo = xmm0;
    Xbyak::Xmm xdst_hi = xmm1;
    Xbyak::Xmm xbase_lo = xmm6;
    Xbyak::Xmm xbase_hi = xmm7;
    Xbyak::Xmm xtmp_lo = xmm8;
    Xbyak::Xmm xtmp_hi = xmm9;
    Xbyak::Xmm xa_lo = xmm6;
    Xbyak::Xmm xa_hi = xmm7;
    Xbyak::Xmm xb_lo = xmm8;
    Xbyak::Xmm xb_hi = xmm9;
    Xbyak::Xmm xc_lo = xmm10;
    Xbyak::Xmm xc_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;

    // store xe
    movaps(ptr[store_addr + 10 * 4 * sizeof(float)], xe_lo);
    movaps(ptr[store_addr + 11 * 4 * sizeof(float)], xe_hi);

    mulps(xe_lo, xe_lo);
    mulps(xe_hi, xe_hi);
    addps(xsum_lo, xe_lo);
    addps(xsum_hi, xe_hi);

    // xdst <- xsum*xalpha+xk
    movaps(xdst_lo, xsum_lo);
    movaps(xdst_hi, xsum_hi);
    mulps(xdst_lo, ptr[store_addr + 0 * 4 * sizeof(float)]);
    mulps(xdst_hi, ptr[store_addr + 0 * 4 * sizeof(float)]);
    addps(xdst_lo, ptr[store_addr + 1 * 4 * sizeof(float)]);
    addps(xdst_hi, ptr[store_addr + 1 * 4 * sizeof(float)]);

    movaps(xbase_lo, xdst_lo);
    movaps(xbase_hi, xdst_hi);
    if (pk != prop_kind::forward_inference)
    {
        if (tail != 0) {
            nchw_tail_sse42(tail, scratch, xbase_lo, xbase_hi);
        }
        else {
            movups(ptr[scratch], xbase_lo);
            movups(ptr[scratch + 4 * sizeof(float)], xbase_hi);
        }
    }
    mulps(xdst_lo, xdst_lo);
    mulps(xdst_hi, xdst_hi);
    mulps(xdst_lo, xbase_lo);
    mulps(xdst_hi, xbase_hi); // xdst = (xsum*xalpha+xk)^3;
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi);
    sqrtps(xdst_lo, xdst_lo);
    sqrtps(xdst_hi, xdst_hi); // xdst = (xsum*xalpha+xk)^0.75
    movaps(xtmp_lo, ptr[store_addr + 6 * 4 * sizeof(float)]);
    movaps(xtmp_hi, ptr[store_addr + 7 * 4 * sizeof(float)]);
    divps(xtmp_lo, xdst_lo);
    divps(xtmp_hi, xdst_hi); // xdst = xsrc / (xsum*xalpha+xk)^0.75
    movaps(xdst_lo, xtmp_lo);
    movaps(xdst_hi, xtmp_hi);

    if (tail != 0) {
        nchw_tail_sse42(tail, dst, xdst_lo, xdst_hi);
    }
    else {
        movups(ptr[dst], xdst_lo);
        movups(ptr[dst + 4 * sizeof(float)], xdst_hi);
    }

    movaps(xa_lo, ptr[store_addr + 2 * 4 * sizeof(float)]);
    movaps(xa_hi, ptr[store_addr + 3 * 4 * sizeof(float)]);
    mulps(xa_lo, xa_lo);
    mulps(xa_hi, xa_hi);
    subps(xsum_lo, xa_lo);
    subps(xsum_hi, xa_hi);

    // xa <- xb
    movaps(xb_lo, ptr[store_addr + 4 * 4 * sizeof(float)]);
    movaps(xb_hi, ptr[store_addr + 5 * 4 * sizeof(float)]);
    movaps(ptr[store_addr + 2 * 4 * sizeof(float)], xb_lo);
    movaps(ptr[store_addr + 3 * 4 * sizeof(float)], xb_hi);

    // xb <- xc
    movaps(xc_lo, ptr[store_addr + 6 * 4 * sizeof(float)]);
    movaps(xc_hi, ptr[store_addr + 7 * 4 * sizeof(float)]);
    movaps(ptr[store_addr + 4 * 4 * sizeof(float)], xc_lo);
    movaps(ptr[store_addr + 5 * 4 * sizeof(float)], xc_hi);

    // xc <- xd
    movaps(xd_lo, ptr[store_addr + 8 * 4 * sizeof(float)]);
    movaps(xd_hi, ptr[store_addr + 9 * 4 * sizeof(float)]);
    movaps(ptr[store_addr + 6 * 4 * sizeof(float)], xd_lo);
    movaps(ptr[store_addr + 7 * 4 * sizeof(float)], xd_hi);

    // xd <- xe
    movaps(xe_lo, ptr[store_addr + 10 * 4 * sizeof(float)]);
    movaps(xe_hi, ptr[store_addr + 11 * 4 * sizeof(float)]);
    movaps(ptr[store_addr + 8 * 4 * sizeof(float)], xe_lo);
    movaps(ptr[store_addr + 9 * 4 * sizeof(float)], xe_hi);
}

template<>
void jit_uni_lrn_fwd_kernel_f32<avx2>::nchw_body_sse42(
    int tail, int HW, prop_kind_t pk,
    Xbyak::Xmm xmask_lo, Xbyak::Xmm xmask_hi,
    Xbyak::Xmm xe_lo, Xbyak::Xmm xe_hi,
    Xbyak::Xmm xsum_lo, Xbyak::Xmm xsum_hi) {}

template<>
jit_uni_lrn_fwd_kernel_f32<avx2>::jit_uni_lrn_fwd_kernel_f32(
    struct nchw_across J,
    float A,
    float K,
    prop_kind_t pk,
    void* code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha(A), k(K)
{
    static const uint32_t mask[] = {
        0x80000000, 0x80000000, 0x80000000, 0x80000000, 0x80000000,
        0x80000000, 0x80000000, 0, 0, 0, 0, 0, 0, 0
    };
    Xbyak::Reg64 c = r10;
    Xbyak::Ymm ymask = ymm2;
    Xbyak::Ymm ye = ymm3;
    Xbyak::Ymm ya = ymm4;
    Xbyak::Ymm yb = ymm5;
    Xbyak::Ymm yc = ymm6;
    Xbyak::Ymm yd = ymm7;
    Xbyak::Ymm ysum = ymm8;

    this->preamble();

    if (J.tail != 0)
    {
        mov(imm_addr64, reinterpret_cast<size_t>(&mask[7 - J.tail]));
        vmovups(ymask, ptr[imm_addr64]);
    }
    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    vbroadcastss(yalpha, xalpha);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    vbroadcastss(yk, xk);

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);

    vxorps(ya, ya, ya);
    vxorps(yb, yb, yb);
    if (J.tail != 0)
        vmaskmovps(yc, ymask, ptr[src + J.HW * 0]);
    else
        vmovups(yc, ptr[src + J.HW * 0]);
    if (J.tail != 0)
        vmaskmovps(yd, ymask, ptr[src + J.HW * 4]);
    else
        vmovups(yd, ptr[src + J.HW * 4]);

    vxorps(ysum, ysum, ysum);
    vfmadd231ps(ysum, yc, yc); // ysum <- ysum + ya^2+yb^2+yc^2+yd^2+ye^2
    vfmadd231ps(ysum, yd, yd);

    mov(c, J.C - 2);
    Label lrn_loop;
    L(lrn_loop);

    if (J.tail != 0)
        vmaskmovps(ye, ymask, ptr[src + J.HW * 8]);
    else
        vmovups(ye, ptr[src + J.HW * 8]);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

    add(src, J.HW * 4);
    add(dst, J.HW * 4);
    if (pk != prop_kind::forward_inference)
        add(scratch, J.HW * 4);
    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    vxorps(ye, ye, ye);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);
    add(src, J.HW * 4);
    add(dst, J.HW * 4);
    if (pk != prop_kind::forward_inference)
        add(scratch, J.HW * 4);

    nchw_body(J.tail, J.HW, pk, ymask, ya, yb, yc, yd, ye, ysum);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                this->getCode()));
}

template<>
jit_uni_lrn_fwd_kernel_f32<sse42>::jit_uni_lrn_fwd_kernel_f32(
    struct nchw_across J,
    float A,
    float K,
    prop_kind_t pk,
    void* code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , alpha(A), k(K)
{
    static const uint32_t mask[] = {
        0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff, 0xffffffff,
        0xffffffff, 0xffffffff, 0, 0, 0, 0, 0, 0, 0
    };

    Xbyak::Reg64 c = r10;

    Xbyak::Xmm xmask_lo = xmm2;
    Xbyak::Xmm xmask_hi = xmm3;
    Xbyak::Xmm xsum_lo = xmm4;
    Xbyak::Xmm xsum_hi = xmm5;
    Xbyak::Xmm xa_lo = xmm6;
    Xbyak::Xmm xa_hi = xmm7;
    Xbyak::Xmm xb_lo = xmm8;
    Xbyak::Xmm xb_hi = xmm9;
    Xbyak::Xmm xc_lo = xmm10;
    Xbyak::Xmm xc_hi = xmm11;
    Xbyak::Xmm xd_lo = xmm12;
    Xbyak::Xmm xd_hi = xmm13;
    Xbyak::Xmm xe_lo = xmm14;
    Xbyak::Xmm xe_hi = xmm15;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(dst, ptr[this->param1 + 8]);
    if (pk != prop_kind::forward_inference)
        mov(scratch, ptr[this->param1 + 16]);

    sub(rsp, stack_space_needed);
    mov(store_addr, rsp);
    and_(store_addr, -15);

    mov(imm_addr64, float2int(this->alpha));
    movq(xalpha, imm_addr64);
    shufps(xalpha, xalpha, 0);

    mov(imm_addr64, float2int(this->k));
    movq(xk, imm_addr64);
    shufps(xk, xk, 0);

    // put alpha and k into store (free up regs)
    movaps(ptr[store_addr + 0 * 4 * sizeof(float)], xalpha);
    movaps(ptr[store_addr + 1 * 4 * sizeof(float)], xk);

    if (J.tail != 0)
    {
        mov(imm_addr64, reinterpret_cast<size_t>(&mask[7 - J.tail]));
        movups(xmask_lo, ptr[imm_addr64]);
        movups(xmask_hi, ptr[imm_addr64 + 4 * sizeof(float)]);
    }
    // init xa, xb
    xorps(xa_lo, xa_lo);
    xorps(xa_hi, xa_hi);
    xorps(xb_lo, xb_lo);
    xorps(xb_hi, xb_hi);

    // read xc, xd
    if (J.tail != 0) {
        movups(xc_lo, ptr[src + J.HW * 0]);
        movups(xc_hi, ptr[src + J.HW * 0 + 4 * sizeof(float)]);
        andps(xc_lo, xmask_lo);
        andps(xc_hi, xmask_hi);
    }
    else {
        movups(xc_lo, ptr[src + J.HW * 0]);
        movups(xc_hi, ptr[src + J.HW * 0 + 4 * sizeof(float)]);
    }
    if (J.tail != 0) {
        movups(xd_lo, ptr[src + J.HW * 4]);
        movups(xd_hi, ptr[src + J.HW * 4 + 4 * sizeof(float)]);
        andps(xd_lo, xmask_lo);
        andps(xd_hi, xmask_hi);
    }
    else {
        movups(xd_lo, ptr[src + J.HW * 4]);
        movups(xd_hi, ptr[src + J.HW * 4 + 4 * sizeof(float)]);
    }

    // put xa, xb, xc, xd into store to free-up regs
    movaps(ptr[store_addr + 2 * 4 * sizeof(float)], xa_lo);
    movaps(ptr[store_addr + 3 * 4 * sizeof(float)], xa_hi);
    movaps(ptr[store_addr + 4 * 4 * sizeof(float)], xb_lo);
    movaps(ptr[store_addr + 5 * 4 * sizeof(float)], xb_hi);
    movaps(ptr[store_addr + 6 * 4 * sizeof(float)], xc_lo);
    movaps(ptr[store_addr + 7 * 4 * sizeof(float)], xc_hi);
    movaps(ptr[store_addr + 8 * 4 * sizeof(float)], xd_lo);
    movaps(ptr[store_addr + 9 * 4 * sizeof(float)], xd_hi);

    xorps(xsum_lo, xsum_lo);
    xorps(xsum_hi, xsum_hi);
    mulps(xc_lo, xc_lo);
    mulps(xc_hi, xc_hi);
    addps(xsum_lo, xc_lo);
    addps(xsum_hi, xc_hi);
    mulps(xd_lo, xd_lo);
    mulps(xd_hi, xd_hi);
    addps(xsum_lo, xd_lo);
    addps(xsum_hi, xd_hi); // xsum <- xsum + xa^2+xb^2+xc^2+xd^2+xe^2

    mov(c, J.C - 2);
    Label lrn_loop;
    L(lrn_loop);

    if (J.tail != 0) {
        movups(xe_lo, ptr[src + J.HW * 8]);
        movups(xe_hi, ptr[src + J.HW * 8 + 4 * sizeof(float)]);
        andps(xe_lo, xmask_lo);
        andps(xe_hi, xmask_hi);
    }
    else {
        movups(xe_lo, ptr[src + J.HW * 8]);
        movups(xe_hi, ptr[src + J.HW * 8 + 4 * sizeof(float)]);
    }

    nchw_body_sse42(J.tail, J.HW, pk, xmask_lo, xmask_hi,
        xe_lo, xe_hi,
        xsum_lo, xsum_hi);

    add(src, J.HW * 4);
    add(dst, J.HW * 4);
    if (pk != prop_kind::forward_inference)
        add(scratch, J.HW * 4);
    dec(c);
    cmp(c, 0);
    jne(lrn_loop, T_NEAR);

    xorps(xe_lo, xe_lo);
    xorps(xe_hi, xe_hi);

    nchw_body_sse42(J.tail, J.HW, pk, xmask_lo, xmask_hi,
        xe_lo, xe_hi,
        xsum_lo, xsum_hi);
    add(src, J.HW * 4);
    add(dst, J.HW * 4);
    if (pk != prop_kind::forward_inference)
        add(scratch, J.HW * 4);

    nchw_body_sse42(J.tail, J.HW, pk, xmask_lo, xmask_hi,
        xe_lo, xe_hi,
        xsum_lo, xsum_hi);

    add(rsp, stack_space_needed);

    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
        this->getCode()));
}

//////////////////////////////////////////////////////////////////////////////
// backward kernel
template <cpu_isa_t isa>
jit_uni_lrn_bwd_kernel_f32<isa>::jit_uni_lrn_bwd_kernel_f32(
    const struct nchw8c_across &J,
    float A,
    float B,
    int use_h_parallel,
    void *code_ptr,
    size_t code_size)
    : jit_generator(code_ptr, code_size)
    , nalphabeta(-2 * A*B)
    , use_h_parallelizm(use_h_parallel)
{
    Xbyak::Reg64 t = rsp;
    Xbyak::Reg64 hw = r10;

    Xbyak::Xmm xsrc_prev = xmm1;
    Xbyak::Xmm xws_prev = xmm2;
    Xbyak::Xmm xdiffdst_prev = xmm3;
    Xbyak::Ymm ysrc = ymm4;
    Xbyak::Ymm yws = ymm5;
    Xbyak::Ymm ydiffdst = ymm6;
    Xbyak::Xmm xsrc_next = xmm7;
    Xbyak::Xmm xws_next = xmm8;
    Xbyak::Xmm xdiffdst_next = xmm9;
    Xbyak::Ymm ya = ymm10;
    Xbyak::Xmm xa = xmm10;
    Xbyak::Ymm yb = ymm11;
    Xbyak::Ymm yd = ymm12;
    Xbyak::Ymm ye = ymm13;
    Xbyak::Ymm ysum = ymm14;
    Xbyak::Ymm ydiffsrc = ymm15;

    this->preamble();

    mov(src, ptr[this->param1 + 0]);
    mov(diffdst, ptr[this->param1 + 8]);
    mov(workspace, ptr[this->param1 + 16]);
    mov(diffsrc, ptr[this->param1 + 24]);

    sub(t, 64);
    mov(imm_addr64, float2int(this->nalphabeta));
    movq(xnalphabeta, imm_addr64);
    vbroadcastss(ynalphabeta, xnalphabeta);

    bool is_single = J.version == 3;
    bool is_first = J.version == -1 || J.version == -2;
    bool is_last = J.version == +1 || J.version == -2;

    if (is_first || is_single) {
        vxorps(xsrc_prev, xsrc_prev, xsrc_prev);
        vmovups(ptr[t + 0], xsrc_prev);
    }
    if (is_last || is_single) {
        vxorps(xsrc_next, xsrc_next, xsrc_next);
        vmovups(ptr[t + 48], xsrc_next);
    }
    mov(hw, this->use_h_parallelizm ? J.W : J.H*J.W);
    Label lrn_loop;
    L(lrn_loop);
    {
        if (!is_first && !is_single) {
            vmovups(xws_prev, ptr[workspace - J.H*J.W * 32 + 16]);
            vmovups(xsrc_prev, ptr[src - J.H*J.W * 32 + 16]);
            vmovups(xdiffdst_prev, ptr[diffdst - J.H*J.W * 32 + 16]);
            vmulps(xa, xws_prev, xws_prev);
            vmulps(xa, xa, xws_prev);
            vsqrtps(xa, xa);
            vsqrtps(xa, xa);
            vmulps(xa, xa, xws_prev);
            vdivps(xsrc_prev, xsrc_prev, xa);
            vmulps(xdiffdst_prev, xdiffdst_prev, xsrc_prev);
        }

        vmovups(ysrc, ptr[src]);
        vmovups(yws, ptr[workspace]);
        vmovups(ydiffdst, ptr[diffdst]);
        vmulps(ya, yws, yws);
        vmulps(ya, ya, yws);
        vsqrtps(ya, ya);
        vsqrtps(ya, ya);
        vdivps(ydiffsrc, ydiffdst, ya);
        vdivps(ysum, ydiffsrc, yws);
        vmulps(ysum, ysum, ysrc);

        if (!is_last && !is_single) {
            vmovups(xws_next, ptr[workspace + J.H*J.W * 32]);
            vmovups(xsrc_next, ptr[src + J.H*J.W * 32]);
            vmovups(xdiffdst_next, ptr[diffdst + J.H*J.W * 32]);
            vmulps(xa, xws_next, xws_next);
            vmulps(xa, xa, xws_next);
            vsqrtps(xa, xa);
            vsqrtps(xa, xa);
            vmulps(xa, xa, xws_next);
            vdivps(xsrc_next, xsrc_next, xa);
            vdivps(xsrc_next, xsrc_next, xws_next);
            vmulps(xdiffdst_next, xdiffdst_next, xsrc_next);
        }

        if (!is_first && !is_single) vmovups(ptr[t + 0], xdiffdst_prev);
        vmovups(ptr[t + 16], ysum);
        if (!is_last && !is_single) vmovups(ptr[t + 48], xdiffdst_next);

        vmovups(ya, ptr[t + 16 - 8]);
        vmovups(yb, ptr[t + 16 - 4]);
        vaddps(ysum, ysum, ya);
        vmulps(ysrc, ysrc, ynalphabeta);
        vaddps(ysum, ysum, yb);

        vmovups(yd, ptr[t + 16 + 4]);
        vmovups(ye, ptr[t + 16 + 8]);
        vaddps(ysum, ysum, yd);
        vaddps(ysum, ysum, ye);

        vfmadd231ps(ydiffsrc, ysum, ysrc);

        vmovups(ptr[diffsrc], ydiffsrc);

        add(src, 32);
        add(diffsrc, 32);
        add(diffdst, 32);
        add(workspace, 32);

        dec(hw);
        cmp(hw, 0);
        jne(lrn_loop, T_NEAR);
    }

    add(t, 64);
    this->postamble();

    ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
        this->getCode()));
}

template struct jit_uni_lrn_fwd_kernel_f32<sse42>;
template struct jit_uni_lrn_fwd_kernel_f32<avx2>;
template struct jit_uni_lrn_bwd_kernel_f32<avx2>;

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
