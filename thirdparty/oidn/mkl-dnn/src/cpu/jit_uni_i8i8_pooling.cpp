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

#include "jit_uni_i8i8_pooling.hpp"

#include <math.h>

#include "mkldnn_thread.hpp"
#include "utils.hpp"

#include "jit_generator.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace Xbyak;

using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::utils;
using namespace mkldnn::impl::types;
using namespace alg_kind;

template <cpu_isa_t isa>
struct jit_uni_i8i8_pooling_fwd_ker_t: public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_uni_i8i8_pooling_fwd_ker_t)

    struct call_params_t {
        const char *src_i8;
        const char *dst_i8;
        size_t kw_range;
        size_t kh_range;
        float idivider;
    };

    using Vmm = typename cpu_isa_traits<isa>::Vmm;
    Xmm xreg(int idx) const { return Xmm(idx); }
    Ymm yreg(int idx) const { return Ymm(xreg(idx).getIdx()); }
    Vmm vreg(int idx) const { return Vmm(xreg(idx).getIdx()); }

    // In case of avx2 with data type i8 we need to use
    // maskmovdqu instruction which has its destination hardcoded in rdi.
    // Windows ABI: abi_param1 is rcx - nothing to do else
    // Unix ABI: abi_param1 is rdi - copy it to rcx and use it as abi_param1
    Reg64 reg_param      = rcx; // Our "unified abi_param1"
    Reg64 reg_ptr_src_i8 = r8;
    Reg64 reg_ptr_dst_i8 = r9;
    Reg64 reg_ptr_maskmovdqu_dst = rdi; // store destination - must be rdi

    Reg64 ki = r10;
    Reg64 kj = r11;
    Reg64 reg_kw = r12;
    Reg64 reg_kh = r13;
    Reg64 c_iter = r14;

    Reg64 aux_reg_src_h = rax;
    Reg64 aux_reg_src_w = rbx;

    Reg64 reg_tmp = rdx;

    Reg64 reg_mask = r15;

    Opmask k_cmp_mask = Opmask(7);

    Opmask mask(int idx) {
        return Opmask(6 - idx);
    }

    // ref to any of XYZ-regs via xreg/yreg/vreg functions
    Xmm xmm_tmp = xreg(0);     // temp to init vreg_tmp
    Vmm vreg_tmp = vreg(0);    // max pooling : holds minimum values for data_type
    Vmm vreg_zeros = vreg(1);

    // only in case of <isa> == avx2
    Vmm vreg_mask    = vreg(2); // full byte-mask
    Xmm xreg_mask_lo = xreg(2); // low 128-bits part of byte-mask (alias for xmm part of vreg_mask)
    Xmm xreg_mask_hi = xreg(3); // "max" - high 128-bits part of byte-mask (stored separately)
    Xmm xreg_mask_q  = xreg(3); // "avg" - 1/4 part of the mask for s8/u8 operations
    Vmm vreg_mask_q  = vreg(3); // "avg" - 1/4 part for non-zero tails

    enum:int {vidx_base = isa == avx2 ? 4 : 2};
    Vmm base_vr(int idx) const { return vreg(vidx_base + idx); }

    size_t sizeof_src_dt() const { return data_type_size(jpp.src_dt); }
    size_t sizeof_dst_dt() const { return data_type_size(jpp.dst_dt); }

    /* max pooling */
    Vmm vreg_src(int idx) const { return base_vr(idx); }            // [0    .. ur_c-1]
    Vmm vreg_dst(int idx) const { return base_vr(jpp.ur_c + idx); } // [ur_c .. 2*ur_c-1]

    /* avg pooling */
    // s32 used for processing of s8/u8 data
    // thus we need to take into account ratio of sizes s32/i8 = 4
    static constexpr data_type_t avg_proc_dt = data_type::s32;
    enum:int {
        s32_to_i8_ratio = sizeof(typename prec_traits<avg_proc_dt>::type)
                / sizeof(typename prec_traits<data_type::u8>::type),
        max_num_ll =  s32_to_i8_ratio
    };
    Vmm vreg_src_s32(int jj, int ll) { return base_vr(3*max_num_ll*jj + ll + 0*max_num_ll); }  // ll: 0..4 [0..3]
    Vmm vreg_dst_s32(int jj, int ll) { return base_vr(3*max_num_ll*jj + ll + 1*max_num_ll); }  // ll: 0..4 [4..7]
    Vmm vreg_dst_f32(int jj, int ll) { return base_vr(3*max_num_ll*jj + ll + 2*max_num_ll); }  // ll: 0..4 [8..11]

    void (*ker_)(const call_params_t *);
    jit_pool_conf_t jpp;

    void init_tmp_reg();
    void init_mask();

    void load_vreg_mask_q(int ll) {};

    void load_src_max_op(int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src_avg_op(int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void load_src(int jj, int ll, int c_tail);

    void store_dst_max_op(int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst_avg_op(int jj, int ll, size_t offset, bool masked, uint64_t msk);
    void store_dst(int jj, int ll, int c_tail);

    void compute_avg_step(int ur_c, int c_tail);
    void compute_max_op(const int jj);
    void compute_max_step(int ur_c, int c_tail);
    void compute_step(int ur_c, int c_tail);

    void compute_c_block();
    void generate();

    static status_t init_conf(jit_pool_conf_t &jpp, const pooling_pd_t *ppd);

    jit_uni_i8i8_pooling_fwd_ker_t(const jit_pool_conf_t &jpp_)
           : jpp(jpp_) {
        generate();
        ker_ = reinterpret_cast<decltype(ker_)>(const_cast<uint8_t*>(
                       getCode()));
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_vreg_mask_q(int ll) {

    // extract ll-th part of mask (ll-th QWORD)
    vpblendd(vreg_mask_q, vreg_zeros, vreg_mask, 0x3 << ll); // 0x3 - mask for 2 x DWORD

    // Move mask from ll-th pos to 0-th pos
    if (ll>0)
        vpermq(vreg_mask_q, vreg_mask_q, ll);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_src_max_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32) {
            vpblendd(vreg_src(jj), vreg_tmp, ptr[aux_reg_src_w + offset], static_cast<uint8_t>(msk));
        } else {
            vpblendvb(vreg_src(jj), vreg_tmp, ptr[aux_reg_src_w + offset], vreg_mask);
        }
    } else
        vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::load_src_max_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        if (jpp.src_dt == s32)
            vmovups(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
        else
            vmovdqu8(vreg_src(jj) | mask(0), ptr[aux_reg_src_w + offset]);
    } else
        vmovups(vreg_src(jj), ptr[aux_reg_src_w + offset]);
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::load_src_avg_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk)
        return;

    auto load_i8 = [&](bool is_signed, const Vmm& vr_src) {

        // Need to use mask of tail?
        if (masked) {

            // load ll-th part of mask into vreg_mask_q
            load_vreg_mask_q(ll);

            // Load by mask from mem into register vr_src
            vpblendvb(vr_src, vreg_zeros, ptr[aux_reg_src_w + offset], vreg_mask_q);

            // Conversion s8/u8 -> s32
            if (is_signed)
                vpmovsxbd(vr_src, vr_src);
            else
                vpmovzxbd(vr_src, vr_src);
        } else {

            // Load from mem into vr_src with conversion
            if (is_signed)
                vpmovsxbd(vr_src, ptr[aux_reg_src_w + offset]);
            else
                vpmovzxbd(vr_src, ptr[aux_reg_src_w + offset]);
        }
    };

    switch (jpp.src_dt) {
        case s32:
            if (masked)
                vpblendd(vreg_src_s32(jj, ll), vreg_zeros, ptr[aux_reg_src_w + offset],
                    static_cast<uint8_t>(msk));
            else
                vmovups(vreg_src_s32(jj, ll), ptr[aux_reg_src_w + offset]);
            break;
        case s8:
                load_i8(true, vreg_src_s32(jj, ll));
            break;
        case u8:
                load_i8(false, vreg_src_s32(jj, ll));
            break;
        default: assert(!"unsupported src data type");
    }
};

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::load_src_avg_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk)
        return;

    const Vmm& vr_src = masked ?
            vreg_src_s32(jj, ll) | mask(ll) :
            vreg_src_s32(jj, ll);

    switch (jpp.src_dt) {
        case s32:
            vmovups(vr_src, ptr[aux_reg_src_w + offset]);
            break;
        case s8:
            vpmovsxbd(vr_src, ptr[aux_reg_src_w + offset]);
            break;
        case u8:
            vpmovzxbd(vr_src, ptr[aux_reg_src_w + offset]);
            break;
        default: assert(!"unsupported src data type");
    }
};

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::load_src(int jj, int ll, int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch (jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_max_op(jj, ll, offset, masked, jpp.tail[0]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/max_num_ll) + jj*c_block)*sizeof_src_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            load_src_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::store_dst_max_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    int c_block = jpp.c_block;

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                vpmaskmovd(ptr[reg_ptr_dst_i8 + offset], vreg_mask, vreg_dst(jj));
                break;
            case s8:
            case u8: {
                // Store low half by mask (bytes 0...15)
                lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
                maskmovdqu(vreg_dst(jj), xreg_mask_lo);

                // Do we need to store high half (bytes 16...31) ?
                const uint64_t low_mask = (1ULL << (c_block/2))-1;
                if (msk & ~low_mask) {
                    vextracti128(Xmm(vreg_dst(jj).getIdx()), vreg_dst(jj), 1);
                    add(reg_ptr_maskmovdqu_dst, c_block / 2);
                    maskmovdqu(vreg_dst(jj), xreg_mask_hi);
                }
            } break;
            default: assert(!"unsupported src data type");
        }
    } else
        vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::store_dst_max_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    if (masked) {
        switch (jpp.src_dt) {
            case s32:
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                break;
            case s8:
            case u8:
                vmovdqu8(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj) | mask(0));
                break;
            default: assert(!"unsupported src data type");
        }
    } else
        vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst(jj));
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::store_dst_avg_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk){
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk)
        return;

    auto s32_to_i8 = [&](bool is_signed, const Vmm& vr_dst) {

        // conversion: s32 -> s16/u16 : {8 x s32}{8 x 0} -> {16 x s16/u16}
        // Result QWORDs (qw0, qw1) permuted: {qw0, 0, qw1, 0}
        if (is_signed)
            vpackssdw(vr_dst, vr_dst, vreg_zeros);
        else
            vpackusdw(vr_dst, vr_dst, vreg_zeros);

        // Permute qwords to restore original order
        // {qw0, 0, qw1, 0} -> {qw0, qw1, 0, 0}
        vpermq(vr_dst, vr_dst, 0x58);

        // conversion: s16/u16 -> s8/u8 : {16 x s16/u16}{16 x 0} -> {32 x s8/u8}
        // Target QWORD qw = {8 x s8/u8} has proper position: {qw, xx, xx, xx}
        if (is_signed)
            vpacksswb(vr_dst, vr_dst, vreg_zeros);
        else
            vpackuswb(vr_dst, vr_dst, vreg_zeros);

    };

    auto store_i8 = [&](bool is_signed, bool is_masked, const Vmm& vr_dst) {

        // Conversion s32 -> s8/u8
        s32_to_i8(is_signed, vr_dst);

        // Need to use mask of tail?
        if (is_masked) {
            // load ll-th part of mask into vreg_mask_q
            load_vreg_mask_q(ll);
        }

        // store 8 bytes
        lea(reg_ptr_maskmovdqu_dst, ptr[reg_ptr_dst_i8 + offset]);
        maskmovdqu(vr_dst, xreg_mask_q);
    };

    switch (jpp.dst_dt) {
        case s32:
            if (masked) {
                vpmaskmovd(ptr[reg_ptr_dst_i8 + offset], vreg_mask, vreg_dst_s32(jj, ll));
            } else
                vmovups(ptr[reg_ptr_dst_i8 + offset], vreg_dst_s32(jj, ll));
            break;
        case s8:
            store_i8(true, masked, vreg_dst_s32(jj, ll));
            break;
        case u8:
            store_i8(false, masked, vreg_dst_s32(jj, ll));
            break;
        default: assert(!"unsuppotred dst data_type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::store_dst_avg_op(int jj, int ll,
        size_t offset, bool masked, uint64_t msk) {
    using namespace data_type;

    // Don't generate useless code
    if (masked && !msk)
        return;

    const Vmm& vr_dst = masked ?
            vreg_dst_s32(jj, ll) | mask(ll) :
            vreg_dst_s32(jj, ll);

    switch (jpp.dst_dt) {
        case s32:
            vmovups(ptr[reg_ptr_dst_i8 + offset], vr_dst);
            break;
        case s8:
            vpmovdb(ptr[reg_ptr_dst_i8 + offset], vr_dst);
            break;
        case u8:
            vpmovusdb(ptr[reg_ptr_dst_i8 + offset], vr_dst);
            break;
        default: assert(!"unsupported dst data_type");
    }
}


template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::store_dst(int jj, int ll,
        int c_tail) {
    using namespace data_type;

    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;

    switch(jpp.alg) {
        case pooling_max: {
            auto offset = jj*c_block*sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_max_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            auto offset = (ll*(c_block/max_num_ll) + jj*c_block)*sizeof_dst_dt();
            bool masked = jj == ur_c - 1 && c_tail;
            store_dst_avg_op(jj, ll, offset, masked, jpp.tail[ll]);
            break;
        }
        default: assert(!"unsupported pooling algorithm");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::compute_max_op(const int jj)
{
    using namespace data_type;
    switch (jpp.src_dt) {
        case s32:
            vpmaxsd(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
            break;
        case s8:
            vpmaxsb(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
            break;
        case u8:
            vpmaxub(vreg_dst(jj), vreg_dst(jj), vreg_src(jj));
            break;
        default: assert(!"unsupported src data type");
    }
}

template <>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::compute_max_op(const int jj)
{
    using namespace data_type;

    // Compare
    switch (jpp.src_dt) {
        case s32:
            vpcmpd(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        case s8:
            vpcmpb(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        case u8:
            vpcmpub(k_cmp_mask, vreg_dst(jj), vreg_src(jj), _cmp_lt_os);
            break;
        default: assert(!"unsupported src data type");
    }

    // move max values into vreg_dst
    if (jpp.src_dt == s32)
        vpblendmd(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj), vreg_src(jj));
    else
        vpblendmb(vreg_dst(jj) | k_cmp_mask, vreg_dst(jj), vreg_src(jj));
}


template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_max_step(int ur_c, int c_tail)
{
    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    for (int jj = 0; jj < ur_c; jj++)
        vmovups(vreg_dst(jj), vreg_tmp);

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                load_src(jj, 0, c_tail);
                compute_max_op(jj);
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++)
        store_dst(jj, 0, c_tail);
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_avg_step(int ur_c, int c_tail)
{
    using namespace data_type;

    Label l_kw, l_kh;

    int iw = jpp.iw;
    int c = jpp.c;

    const int num_ll = data_type_size(avg_proc_dt)/data_type_size(jpp.src_dt);

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                uni_vpxor(vreg_src_s32(jj, ll), vreg_src_s32(jj, ll), vreg_src_s32(jj, ll));
                uni_vpxor(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll));
            }
        }
    }

    mov(aux_reg_src_h, reg_ptr_src_i8);

    xor_(kj, kj);
    L(l_kh);
    {
        mov(aux_reg_src_w, aux_reg_src_h);
        xor_(ki, ki);
        L(l_kw);
        {
            for (int jj = 0; jj < ur_c; jj++) {
                for (int ll = 0; ll < num_ll; ll++) {
                    bool masked = jj == ur_c - 1 && c_tail;
                    size_t msk = jpp.tail[ll];
                    if (!(masked && !msk)) {
                        load_src(jj, ll, c_tail);
                        vpaddd(vreg_dst_s32(jj, ll), vreg_dst_s32(jj, ll),
                                vreg_src_s32(jj, ll));
                    }
                }
            }
            add(aux_reg_src_w, c * sizeof_src_dt());
            inc(ki);
            cmp(ki, reg_kw);
            jl(l_kw, T_NEAR);
        }
        add(aux_reg_src_h, iw * c * sizeof_src_dt());
        inc(kj);
        cmp(kj, reg_kh);
        jl(l_kh, T_NEAR);
    }

    for (int jj = 0; jj < ur_c; jj++) {
        for (int ll = 0; ll < num_ll; ll++) {
            bool masked = jj == ur_c - 1 && c_tail;
            size_t msk = jpp.tail[ll];
            if (!(masked && !msk)) {
                vcvtdq2ps(vreg_dst_f32(jj, ll), vreg_dst_s32(jj, ll));
                vfmadd132ps(vreg_dst_f32(jj, ll), vreg_zeros, vreg_tmp);
                vcvtps2dq(vreg_dst_s32(jj, ll), vreg_dst_f32(jj, ll));
                store_dst(jj, ll, c_tail);
            }
        }
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_step(int ur_c, int c_tail) {
    switch (jpp.alg) {
        case pooling_max:
            compute_max_step(ur_c, c_tail); break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            compute_avg_step(ur_c, c_tail); break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::compute_c_block(){
    Label l_main_loop;

    int nb_c = jpp.nb_c;
    int c_block = jpp.c_block;
    int ur_c = jpp.ur_c;
    int ur_c_tail = jpp.ur_c_tail;
    int c_steps = nb_c / ur_c;
    int c_tail = jpp.c_tail;

    xor_(c_iter, c_iter);
    if (c_steps > 0) {
        L(l_main_loop); {
            compute_step(ur_c, 0);
            add(reg_ptr_src_i8, ur_c*c_block*sizeof_src_dt());
            add(reg_ptr_dst_i8, ur_c*c_block*sizeof_dst_dt());
            inc(c_iter);
            cmp(c_iter, c_steps);
            jl(l_main_loop, T_NEAR);
        }
    }

    if (ur_c_tail != 0) {
        compute_step(ur_c_tail, c_tail);
    }
}

template<>
void jit_uni_i8i8_pooling_fwd_ker_t<avx2>::init_mask() {
    using namespace data_type;
    using cpu_isa = cpu_isa_traits<avx2>;

    // AVX2 mask initialization: mask stored in Ymm-regs
    auto init = [&](uint64_t bit_mask, bool init_mask_q) {
        const size_t QW_PER_VREG = cpu_isa::vlen / sizeof(uint64_t);

        uint64_t vmask[QW_PER_VREG];
        for (size_t i = 0; i < QW_PER_VREG; i++){

            uint64_t qw_vmask=0ULL;
            const size_t DBITS = 8*sizeof_src_dt();
            const uint64_t VMSK = 1ULL << (DBITS-1);
            const size_t D_PER_QW = (8*sizeof(qw_vmask))/DBITS;
            for (size_t j = 0; j < D_PER_QW; j++) {
                if (bit_mask & 1)
                    qw_vmask |= VMSK << DBITS * j;
                bit_mask >>= 1;
            }
            vmask[i] = qw_vmask;
        }

        // Put QWORDS with target mask into xmm regs
        const int xdst_i[QW_PER_VREG] = {
                xreg_mask_lo.getIdx(),
                xreg_mask_lo.getIdx(),
                xreg_mask_hi.getIdx(),
                xreg_mask_hi.getIdx()
        };
        const int xsrc_i[QW_PER_VREG] = {
                vreg_zeros.getIdx(),   // 0-th qword insert in zeros -> {qw0,  0}
                xreg_mask_lo.getIdx(), // 1-st and 0-th merge        -> {qw0,qw1}
                vreg_zeros.getIdx(),
                xreg_mask_hi.getIdx()
        };
        const uint8 qw_dst_idx[QW_PER_VREG] = {0, 1, 0, 1}; // qword index in 128-bit xreg

        for (size_t i = 0; i < QW_PER_VREG; i++) {
            mov(reg_mask, vmask[i]);
            vpinsrq(Xmm(xdst_i[i]), Xmm(xsrc_i[i]), reg_mask, qw_dst_idx[i]);
        }

        // Merge Low (xreg_mask_lo alias for vreg_mask.xreg)
        // and High (xreg_mask_hi) into full vreg_mask
        // vreg_mask -> {xreg_mask_hi, vreg_mask.xreg}
        vinserti128(vreg_mask, vreg_mask, xreg_mask_hi, 1);

        // Keep only low qword of mask in xreg_mask_q
        if (init_mask_q) {
            mov(reg_mask, vmask[0]);
            vpinsrq(xreg_mask_q, Xmm(vreg_zeros.getIdx()), reg_mask, 0);
        }
    };

    uint64_t tail_mask = (1ULL << jpp.c_tail) - 1;
    switch (jpp.alg) {
        case pooling_max:
            // For "max" we need mask only in case of non-zero tail
            if (tail_mask)
                init(tail_mask, false);
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            // For "avg" we need mask:
            // - s32   - in case of the non-zero tail
            // - s8/u8 - irrespective of the tail
            switch (jpp.src_dt) {
                case s32:
                    if (tail_mask)
                        init(tail_mask, false);
                    break;
                case s8:
                case u8:
                    init(tail_mask ? tail_mask : ~0ULL, tail_mask == 0);
                    break;
                default: assert(!"unsupported src data type");
            }
            break;
        default: assert(!"unsupported pooling algorithm");
    }
}

template<>
void jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>::init_mask() {

    for (int ll = 0; ll < max_num_ll; ll++) {
        mov(reg_mask, jpp.tail[ll]);
        kmovq(mask(ll), reg_mask);
    }
}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_tmp_reg() {
    using namespace data_type;

    switch (jpp.alg) {
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding:
            mov(reg_tmp, ptr[reg_param + offsetof(call_params_t, idivider)]);
            movq(xmm_tmp, reg_tmp);
            vpbroadcastd(vreg_tmp, xmm_tmp);
            break;
        case pooling_max:
            switch (jpp.src_dt) {
                case s32:
                    mov(reg_tmp, nstl::numeric_limits<int32_t>::lowest());
                    break;
                case s8:
                    mov(reg_tmp, nstl::numeric_limits<int8_t>::lowest());
                    break;
                case u8:
                    mov(reg_tmp, nstl::numeric_limits<uint8_t>::lowest());
                    break;
                default: assert(!"unsupported src data_type");
            }

            movq(xmm_tmp, reg_tmp);
            if (jpp.src_dt == s32)
                vpbroadcastd(vreg_tmp, xmm_tmp);
            else
                vpbroadcastb(vreg_tmp, xmm_tmp);
            break;
        default: assert(!"unsupported pooling algorithm");
    }

}

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_ker_t<isa>::generate() {
    preamble();

#if !defined(_WIN32)
    // Always use rcx as abi_param1 -
    // see the note about maskmovdqu near reg_param.
    mov(rcx, rdi);
#endif

#   define READ_PARAM(reg, field) \
        mov(reg, ptr[reg_param + offsetof(call_params_t, field)])
    READ_PARAM(reg_ptr_src_i8, src_i8);
    READ_PARAM(reg_ptr_dst_i8, dst_i8);
    READ_PARAM(reg_kw, kw_range);
    READ_PARAM(reg_kh, kh_range);

#   undef READ_PARAM

    uni_vpxor(vreg_zeros, vreg_zeros, vreg_zeros);

    init_mask();

    init_tmp_reg();

    compute_c_block();

    postamble();
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jit_pool_conf_t &jpp,
        const pooling_pd_t *ppd) {
    if (!mayiuse(isa))
        return status::unimplemented;

    const auto &pd = *ppd->desc();
    const memory_desc_wrapper src_d(ppd->src_md());
    const memory_desc_wrapper dst_d(ppd->dst_md());

    jpp.mb = src_d.dims()[0];
    jpp.c = src_d.dims()[1];
    jpp.ih = src_d.dims()[2];
    jpp.iw = src_d.dims()[3];
    jpp.oh = dst_d.dims()[2];
    jpp.ow = dst_d.dims()[3];

    jpp.stride_h = pd.strides[0];
    jpp.stride_w = pd.strides[1];
    jpp.kh = pd.kernel[0];
    jpp.kw = pd.kernel[1];

    jpp.t_pad = pd.padding[0][0];
    jpp.l_pad = pd.padding[0][1];

    jpp.alg = pd.alg_kind;

    jpp.src_dt = pd.src_desc.data_type;
    jpp.dst_dt = pd.dst_desc.data_type;

    // data_type items per one vreg on the <isa>
    //     isa == avx2    : 32 bytes -> 32 for s8/u8, 8 for s32
    //     isa == avx512* : 64 bytes -> 64 for s8/u8, 16 for s32
    int simd_w = cpu_isa_traits<isa>::vlen / data_type_size(jpp.src_dt);

    jpp.c_block = simd_w;
    jpp.c_tail = jpp.c % jpp.c_block;
    jpp.nb_c = jpp.c / jpp.c_block;
    jpp.ur_c = 1;
    jpp.ur_c_tail = jpp.nb_c - (jpp.nb_c / jpp.ur_c)*jpp.ur_c +
            (jpp.c_tail != 0);

    size_t tail_mask = (1ULL << jpp.c_tail) - 1;

    switch (jpp.alg) {
        case pooling_max:
            jpp.tail[0] = tail_mask;
            jpp.tail[1] = 0;
            jpp.tail[2] = 0;
            jpp.tail[3] = 0;
            break;
        case pooling_avg_include_padding:
        case pooling_avg_exclude_padding: {
            // avg_proc_dt (s32) defines granularity (because u8/s8 processed as s32)
            // avx2 : 8, avx512 : 16
            const size_t msk_gran = cpu_isa_traits<isa>::vlen / data_type_size(avg_proc_dt);
            const size_t msk_msk = (1ULL << msk_gran) - 1;
            size_t m = tail_mask;
            for (size_t ll = 0; ll < max_num_ll; ll++) {
                jpp.tail[ll] = m & msk_msk;
                m = m >> msk_gran;
            }
            break;
        }
        default: return status::unimplemented;
    }

    return status::success;
}

template <cpu_isa_t isa>
status_t jit_uni_i8i8_pooling_fwd_t<isa>::pd_t::jit_conf() {
    return jit_uni_i8i8_pooling_fwd_ker_t<isa>::init_conf(jpp_, this);
}

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::
jit_uni_i8i8_pooling_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd), ker_(nullptr)
{ ker_ = new jit_uni_i8i8_pooling_fwd_ker_t<isa>(pd()->jpp_); }

template <cpu_isa_t isa>
jit_uni_i8i8_pooling_fwd_t<isa>::
~jit_uni_i8i8_pooling_fwd_t() { delete ker_; }

template <cpu_isa_t isa>
void jit_uni_i8i8_pooling_fwd_t<isa>::execute_forward(
        const exec_ctx_t &ctx) const {
    auto src_i8 = CTX_IN_MEM(const char *, MKLDNN_ARG_SRC);
    auto dst_i8 = CTX_OUT_MEM(char *, MKLDNN_ARG_DST);

    const memory_desc_wrapper src_d(pd()->src_md());
    const memory_desc_wrapper dst_d(pd()->dst_md());

    const auto &jpp = pd()->jpp_;

    parallel_nd(jpp.mb, jpp.oh, jpp.ow,
            [&](int n, int oh, int ow) {
        const int ih = nstl::max(oh*jpp.stride_h - jpp.t_pad, 0);
        const int iw = nstl::max(ow*jpp.stride_w - jpp.l_pad, 0);

        const int kh_start = nstl::max(0, jpp.t_pad - oh * jpp.stride_h);
        const int kh_end = nstl::min(jpp.kh,
                jpp.ih + jpp.t_pad - oh * jpp.stride_h);
        const int kw_start = nstl::max(0, jpp.l_pad - ow * jpp.stride_w);
        const int kw_end = nstl::min(jpp.kw,
                jpp.iw + jpp.l_pad - ow * jpp.stride_w);

        auto p = typename jit_uni_i8i8_pooling_fwd_ker_t<isa>::call_params_t();
        p.src_i8 = &src_i8[
            src_d.blk_off(n, 0, ih, iw) * src_d.data_type_size()];
        p.dst_i8 = &dst_i8[
            dst_d.blk_off(n, 0, oh, ow) * dst_d.data_type_size()];
        p.kw_range = (size_t)(kw_end - kw_start);
        p.kh_range = (size_t)(kh_end - kh_start);
        p.idivider = 1.0f / ((jpp.alg == pooling_avg_exclude_padding) ?
            p.kh_range*p.kw_range : jpp.kw*jpp.kh);

        ker_->ker_(&p);
    });
}

// Explicit instantiation only for supported <isa> values.
//
template struct jit_uni_i8i8_pooling_fwd_ker_t<avx512_core>;
template struct jit_uni_i8i8_pooling_fwd_t<avx512_core>;

template struct jit_uni_i8i8_pooling_fwd_ker_t<avx2>;
template struct jit_uni_i8i8_pooling_fwd_t<avx2>;

}
}
}
