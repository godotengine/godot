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
#include "mkldnn_thread.hpp"
#include "type_helpers.hpp"
#include "utils.hpp"

#include "jit_avx512_common_lrn.hpp"

#include "jit_generator.hpp"

#define FWD_RBC 4
#define BWD_RBC 3

#define XMM_SIZE (4*sizeof(float))
#define ZMM_SIZE (vlen)
#define BUFFER_BLOCK (XMM_SIZE + ZMM_SIZE + XMM_SIZE)
#define BUFFER_NEXT_OFFSET (XMM_SIZE + ZMM_SIZE)
#define SRC_PREV_OFFSET (vlen - XMM_SIZE)

#define IRB_LOOP(statement) for(int irb = 0; irb < loop_size; irb++) { \
    statement;\
}

namespace mkldnn {
namespace impl {
namespace cpu {

using namespace mkldnn::impl::status;
using namespace mkldnn::impl::utils;

using namespace Xbyak;

enum params { vsize = 16, vlen = 64};

typedef struct {
    const float *src;
    float *dst, *ws0, *ws1;
} jit_args_fwd_t;

typedef struct {
    const float *src, *diff_dst, *ws0, *ws1;
    float *diff_src;
} jit_args_bwd_t;

struct nChw16c_across {
/*  version:
 *  -1: channels 0..15,
 *   1: channels C-16 .. C-1,
 *   0: other channels
 *   3: channels only for this kernel(without prev and next)
 */
    int H, W, version;
    nChw16c_across(int h, int w, int v) : H(h), W(w), version(v) {}
};

struct jit_avx512_common_lrn_fwd_t::jit_avx512_common_lrn_kernel_f32:
       public jit_generator {
    int HW, W;
    bool is_first;
    bool is_last;
    bool is_single;

    Reg64 src = rax;
    Reg64 dst = r8;
    Reg64 scratch0 = rdx;
    Reg64 scratch1 = rsi;
    Reg64 imm_addr64 = rbx;

    Zmm zalpha = zmm0;
    Xmm xalpha = xmm0;
    Zmm zk = zmm1;
    Xmm xk = xmm1;

    Reg64 param = abi_param1;
    Reg64 t = rsp;
    Reg64 hw = r9;

    int xsrc_prev = 2;
    int zsrc = 7;
    int xsrc_next = 3;
    int zc = 7;

    int za = 2;
    int zb = 3;
    int zd = 5;
    int ze = 6;
    int zsum = 4;
    int zdst = 2;
    int zbase = 3;
    int zsum2 = 5;

    prop_kind_t pk;
    int use_h_parallelism;

    float alpha, k;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_f32)

    void (*ker)(jit_args_fwd_t *);
    void operator()(jit_args_fwd_t *arg) { ker(arg); }

    enum {
        prf0_offt = 1*FWD_RBC,
        prf2_offt = 8*FWD_RBC
    };

    inline void compute_loop(int loop_size_param)
    {
        // loop_size - param for IRB_LOOP macro
        int loop_size = FWD_RBC;

        auto xreg = [=](int irb, int i) {
            return Xmm(irb*3 + i);
        };

        auto zreg = [=](int irb, int i) {
            return Zmm(irb*7 + i);
        };

        if (!is_first && !is_single) {
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt - HW)*vlen]));
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt - HW)*vlen]));
        }
        IRB_LOOP(mic_prefetcht0(EVEX_compress_addr(src, (irb + prf0_offt)*vlen)));
        IRB_LOOP(mic_prefetcht2(EVEX_compress_addr(src, (irb + prf2_offt)*vlen)));
        if (!is_last && !is_single) {
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt + HW)*vlen]));
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt + HW)*vlen]));
        }
        if (pk != prop_kind::forward_inference) {
            IRB_LOOP(mic_prefetcht0(EVEX_compress_addr(scratch0,
                       (irb + prf0_offt)*vlen)));
            IRB_LOOP(mic_prefetcht2(EVEX_compress_addr(scratch0,
                       (irb + prf2_offt)*vlen)));
        }
        IRB_LOOP(mic_prefetcht0(EVEX_compress_addr(dst, (irb + prf0_offt)*vlen)));
        IRB_LOOP(mic_prefetcht2(EVEX_compress_addr(dst, (irb + prf2_offt)*vlen)));
        if (pk != prop_kind::forward_inference) {
            IRB_LOOP(mic_prefetcht0(EVEX_compress_addr(scratch1,
                         (irb + prf0_offt) * vlen)));
            IRB_LOOP(mic_prefetcht2(EVEX_compress_addr(scratch1,
                         (irb + prf2_offt) * vlen)));
        }

        loop_size = loop_size_param;
        if (loop_size == 0)
            return;
        if (!is_first && !is_single) {
            IRB_LOOP(vmovups(xreg(irb, xsrc_prev),
                        ptr[src + (irb - HW) * vlen + SRC_PREV_OFFSET]));
        }
        IRB_LOOP(vmovups(zreg(irb, zsrc), EVEX_compress_addr(src,irb*vlen)));
        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(xreg(irb, xsrc_next),
                        ptr[src + (irb + HW) * vlen]));
        }

        if (!is_first && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb*BUFFER_BLOCK],
                        xreg(irb, xsrc_prev)));
        }
        IRB_LOOP(vmovups(EVEX_compress_addr(t, irb*BUFFER_BLOCK + XMM_SIZE),
                    zreg(irb, zsrc)));
        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb*BUFFER_BLOCK + BUFFER_NEXT_OFFSET],
                    xreg(irb, xsrc_next)));
        }

        IRB_LOOP(vmovups(zreg(irb, za), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                        + XMM_SIZE - 2*sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, zb), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                        + XMM_SIZE - sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, zd), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                        + XMM_SIZE + sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, ze), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                        + XMM_SIZE + 2*sizeof(float))));

        assert(zc == zsrc);
        IRB_LOOP(vmulps(zreg(irb, zsum), zreg(irb, zc), zreg(irb, zc)));

        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, za), zreg(irb, za)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, zb), zreg(irb, zb)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, zd), zreg(irb, zd)));
        IRB_LOOP(vfmadd231ps(zreg(irb, zsum), zreg(irb, ze), zreg(irb, ze)));

        IRB_LOOP(vfmadd132ps(zreg(irb, zsum), zk, zalpha));

        IRB_LOOP(vmovaps(zreg(irb, zbase), zreg(irb, zsum)));

        IRB_LOOP(vmulps(zreg(irb, zsum2), zreg(irb, zsum), zreg(irb, zsum)));
        IRB_LOOP(vmulps(zreg(irb, zsum), zreg(irb, zsum), zreg(irb, zsum2)));

        IRB_LOOP(vsqrtps(zreg(irb, zsum), zreg(irb, zsum)));
        IRB_LOOP(vsqrtps(zreg(irb, zsum), zreg(irb, zsum)));

        if (pk != prop_kind::forward_inference) {
            IRB_LOOP(vmovups(EVEX_compress_addr(scratch0, irb*vlen),
                        zreg(irb, zsum)));
        }
        IRB_LOOP(vdivps(zreg(irb, zdst), zreg(irb, zsrc), zreg(irb, zsum)));
        IRB_LOOP(vmovups(EVEX_compress_addr(dst, irb*vlen), zreg(irb, zdst)));
        if (pk != prop_kind::forward_inference) {
            /* ws1 = zdst / zbase = zsrc / (zbase^1.75) */
            IRB_LOOP(vdivps(zreg(irb, zsum), zreg(irb, zdst), zreg(irb, zbase)));
            IRB_LOOP(vmovups(EVEX_compress_addr(scratch1, irb*vlen),
                        zreg(irb, zsum)));
        }
    }

    jit_avx512_common_lrn_kernel_f32(
        const struct nChw16c_across &J,
        prop_kind_t prop_kind,
        int use_h_parallel,
        float A,
        float K,
        void *code_ptr = nullptr,
        size_t code_size = 2 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , pk(prop_kind)
        , use_h_parallelism(use_h_parallel)
        , alpha(A)
        , k(K)
    {
        this->preamble();

        mov(src, ptr[param + 0]);
        mov(dst, ptr[param + 8]);
        if (pk != prop_kind::forward_inference)
        {
            mov(scratch0, ptr[param + 16]);
            mov(scratch1, ptr[param + 24]);
        }
        is_first = J.version == -1 || J.version == -2;
        is_last  = J.version == +1 || J.version == -2;
        is_single = J.version == 3;

        W = J.W;
        HW = J.W*J.H;
        int LSB = use_h_parallelism ? W : HW;

        sub(t, FWD_RBC*BUFFER_BLOCK);
        mov(imm_addr64, float2int(this->alpha));
        movq(xalpha, imm_addr64);
        vbroadcastss(zalpha, xalpha);

        mov(imm_addr64, float2int(this->k));
        movq(xk, imm_addr64);
        vbroadcastss(zk, xk);

        if (is_first || is_single) {
            vxorps(xmm2, xmm2, xmm2);
            for(int irb = 0; irb < FWD_RBC; irb++) {
                vmovups(ptr[t + irb*BUFFER_BLOCK], xmm2);
            }
        }
        if (is_last || is_single) {
            vxorps(xmm2, xmm2, xmm2);
            for(int irb = 0; irb < FWD_RBC; irb++) {
                vmovups(ptr[t + irb*BUFFER_BLOCK + BUFFER_NEXT_OFFSET],
                    xmm2);
            }
        }

        int LSREST = LSB % FWD_RBC;
        int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            mov(hw, LS);

            L(lrn_loop);
            {
                compute_loop(FWD_RBC);

                add(src, FWD_RBC*vlen);
                add(dst, FWD_RBC*vlen);
                if (pk != prop_kind::forward_inference)
                {
                    add(scratch0, FWD_RBC*vlen);
                    add(scratch1, FWD_RBC*vlen);
                }

                for(int irb = 0; irb < FWD_RBC; irb++)
                    dec(hw);
                cmp(hw, 0);
                jne(lrn_loop, T_NEAR);
            }
        }

        compute_loop(LSREST);

        add(t, FWD_RBC*BUFFER_BLOCK);
        this->postamble();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                    this->getCode()));
    }
};

status_t jit_avx512_common_lrn_fwd_t::pd_t::init() {
    using namespace prop_kind;
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    bool ok = true
        && mayiuse(avx512_common)
        && is_fwd()
        && !has_zero_dim_memory()
        && everyone_is(data_type::f32, data_d.data_type())
        && data_d.ndims() == 4
        && data_d.dims()[1] % vsize == 0
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    if (desc()->prop_kind == forward_training) {
        dims_t ws_dims = { MB(), C(), H(), 2*W() };
        mkldnn_memory_desc_init_by_tag(&ws_md_, 4, ws_dims, data_type::f32,
                format_tag::nChw16c);
    }

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && desc()->lrn_beta == 0.75
        && data_d.matches_tag(format_tag::nChw16c);

    return args_ok_across ? success : unimplemented;
}

jit_avx512_common_lrn_fwd_t::jit_avx512_common_lrn_fwd_t(const pd_t *apd)
    : cpu_primitive_t(apd)
    , use_h_parallelism(0), ker_(nullptr), ker_first_(nullptr)
    , ker_last_(nullptr) {
    using namespace alg_kind;
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float k = pd()->desc()->lrn_k;

    auto pk = pd()->desc()->prop_kind;

    use_h_parallelism = H > 28 ? 1 : 0;

    if (C / vsize == 1) {
        ker_ = new jit_avx512_common_lrn_kernel_f32(nChw16c_across(H, W, 3), pk,
            use_h_parallelism, alpha, k);
    } else {
        ker_ = new jit_avx512_common_lrn_kernel_f32(nChw16c_across(H, W, 0), pk,
            use_h_parallelism, alpha, k);
        ker_first_ = new jit_avx512_common_lrn_kernel_f32(
            nChw16c_across(H, W, -1), pk, use_h_parallelism, alpha, k);
        ker_last_ = new jit_avx512_common_lrn_kernel_f32(
            nChw16c_across(H, W, +1), pk, use_h_parallelism, alpha, k);
    }
}

jit_avx512_common_lrn_fwd_t::~jit_avx512_common_lrn_fwd_t()
{ delete ker_; delete ker_first_; delete ker_last_; }

void jit_avx512_common_lrn_fwd_t::execute_forward(const exec_ctx_t &ctx) const
{
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto dst = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DST);
    auto ws = CTX_OUT_MEM(data_t *, MKLDNN_ARG_WORKSPACE);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        const int C16 = C / vsize;
        const size_t work_amount = use_h_parallelism ? N*C16*H : N*C16;

        balance211(work_amount, nthr, ithr, start, end);
        if (use_h_parallelism) {
            int n{0}, c16{0}, h{0};
            nd_iterator_init(start, n, N, c16, C16, h, H);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize
                    + h*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize
                    + h*2*W*vsize;
                auto ws_offset1 = ws_offset0 + W*vsize;

                jit_args_fwd_t args;
                args.src = &src[offset];
                args.dst = &dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);
                nd_iterator_step(n, N, c16, C16, h, H);
            }
        } else {
            int n{0}, c16{0};
            nd_iterator_init(start, n, N, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize;
                auto ws_offset1 = ws_offset0 + H*W*vsize;

                jit_args_fwd_t args;
                args.src = &src[offset];
                args.dst = &dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);

                nd_iterator_step(n, N, c16, C16);
            }
        }
    });
}

struct jit_avx512_common_lrn_bwd_t::jit_avx512_common_lrn_kernel_f32:
    public jit_generator {
    int HW, W;
    bool is_first;
    bool is_last;
    bool is_single;

    Reg64 src = rax;
    Reg64 diffsrc = r8;
    Reg64 diffdst = r9;
    Reg64 workspace0 = rdx;
    Reg64 workspace1 = rsi;
    Reg64 imm_addr64 = rbx;

    Zmm znalphabeta = zmm0;
    Xmm xnalphabeta = xmm0;

    Reg64 param = abi_param1;
    Reg64 t = rsp;
    Reg64 hw = r10;

    int xws1_prev = 1;
    int xdiffdst_prev = 2;
    int zws1 = 1;

    int zsrc = 1;
    int zdiffdst = 5;
    int zdiffsrc = 6;

    int xws1_next = 1;
    int xdiffdst_next = 3;

    int za = 1;
    int zb = 2;
    int zd = 3;
    int ze = 4;
    int zws0 = 2;

    float nalphabeta;

    int use_h_parallelism;

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_avx512_common_lrn_kernel_f32)

    void (*ker)(jit_args_bwd_t *);
    void operator()(jit_args_bwd_t *arg) { ker(arg); }

    enum {
        prf0_offt = 1*BWD_RBC,
        prf2_offt = 8*BWD_RBC
    };

    inline void compute_loop(int loop_size_param, int prefetchL1,
            int prefetchL2)
    {
        // loop_size - param for IRB_LOOP macro
        int loop_size = loop_size_param;

        auto xreg = [=](int irb, int i) {
            return Xmm(irb*6 + i);
        };

        auto zreg = [=](int irb, int i) {
            return Zmm(irb*6 + i);
        };

// ---- prefetching -------------------------------------------
        if (!is_first && !is_single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt
                        - 2 * HW) * vlen]));
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(ptr[diffdst    + (irb + prf0_offt
                        - HW) * vlen]));
        }

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[src + (irb + prf0_offt)*vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(ptr[src + (irb + prf2_offt)*vlen]));

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt)*vlen]));

        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[diffdst + (irb + prf0_offt)*vlen]));

        if (!is_last && !is_single) {
            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(ptr[workspace1 + (irb + prf0_offt
                        + 2 * HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(ptr[workspace1 + (irb + prf2_offt
                        + 2 * HW) * vlen]));

            if (prefetchL1)
                IRB_LOOP(mic_prefetcht0(ptr[diffdst +  (irb + prf0_offt
                          + HW) * vlen]));
            if (prefetchL2)
                IRB_LOOP(mic_prefetcht2(ptr[diffdst +  (irb + prf2_offt
                        + HW) * vlen]));
        }
        if (prefetchL1)
            IRB_LOOP(mic_prefetcht0(ptr[workspace0 + (irb + prf0_offt)*vlen]));
        if (prefetchL2)
            IRB_LOOP(mic_prefetcht2(ptr[workspace0 + (irb + prf2_offt)*vlen]));
// -----------------------------------------------------------

        if (loop_size_param == 0)
            return;

        if (!is_first && !is_single) {
            IRB_LOOP(vmovups(xreg(irb, xws1_prev), ptr[workspace1 + (irb
                    - 2 * HW) * vlen + SRC_PREV_OFFSET]));
            IRB_LOOP(vmovups(xreg(irb, xdiffdst_prev), ptr[diffdst + (irb
                    - HW) * vlen + SRC_PREV_OFFSET]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_prev), xreg(irb, xdiffdst_prev),
                    xreg(irb, xws1_prev)));
        }

        IRB_LOOP(vmovups(zreg(irb, zws1),
                EVEX_compress_addr(workspace1, irb*vlen)));
        IRB_LOOP(vmovups(zreg(irb, zdiffdst),
                EVEX_compress_addr(diffdst, irb*vlen)));
        IRB_LOOP(vmulps(zreg(irb, zdiffsrc), zreg(irb, zdiffdst),
                zreg(irb, zws1)));

        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(xreg(irb, xws1_next), ptr[workspace1 + (irb
                    + 2 * HW) * vlen]));
            IRB_LOOP(vmovups(xreg(irb, xdiffdst_next), ptr[diffdst +  (irb
                    + HW) * vlen]));
            IRB_LOOP(vmulps(xreg(irb, xdiffdst_next), xreg(irb, xdiffdst_next),
                    xreg(irb, xws1_next)));
        }

        if (!is_first && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb*BUFFER_BLOCK],
                    xreg(irb, xdiffdst_prev)));
        }
        IRB_LOOP(vmovups(EVEX_compress_addr(t, irb*BUFFER_BLOCK + XMM_SIZE),
                 zreg(irb, zdiffsrc)));
        if (!is_last && !is_single) {
            IRB_LOOP(vmovups(ptr[t + irb*BUFFER_BLOCK + BUFFER_NEXT_OFFSET],
                 xreg(irb, xdiffdst_next)));
        }

        IRB_LOOP(vmovups(zreg(irb, za), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                + XMM_SIZE - 2*sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, zb), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                + XMM_SIZE - 1*sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, zd), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                + XMM_SIZE + 1*sizeof(float))));
        IRB_LOOP(vmovups(zreg(irb, ze), EVEX_compress_addr(t, irb*BUFFER_BLOCK
                + XMM_SIZE + 2*sizeof(float))));
        IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc),
                zreg(irb, za)));
        assert(zsrc == za);
        IRB_LOOP(vmovups(zreg(irb, zsrc), EVEX_compress_addr(src, irb*vlen)));
        IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc),
                zreg(irb, zb)));
        IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc),
                zreg(irb, zd)));
        IRB_LOOP(vaddps(zreg(irb, zdiffsrc), zreg(irb, zdiffsrc),
                zreg(irb, ze)));
        IRB_LOOP(vmulps(zreg(irb, zsrc), zreg(irb, zsrc), znalphabeta));

        IRB_LOOP(vmovups(zreg(irb, zws0),
                 EVEX_compress_addr(workspace0, irb*vlen)));
        IRB_LOOP(vdivps(zreg(irb, zdiffdst), zreg(irb, zdiffdst),
                 zreg(irb, zws0)));
        IRB_LOOP(vfmadd213ps(zreg(irb, zdiffsrc), zreg(irb, zsrc),
                 zreg(irb, zdiffdst)));

        Label unaligned_store, end_store;
        test(diffsrc, vlen - 1);
        jnz(unaligned_store, T_NEAR);
        IRB_LOOP(uni_vmovntps(EVEX_compress_addr(diffsrc, irb*vlen),
                 zreg(irb, zdiffsrc)));
        jmp(end_store, T_NEAR);
        L(unaligned_store); {
            IRB_LOOP(uni_vmovups(EVEX_compress_addr(diffsrc, irb*vlen),
                     zreg(irb, zdiffsrc)));
        }
        L(end_store);
    }

    jit_avx512_common_lrn_kernel_f32(
        const struct nChw16c_across &J,
        float A,
        float B,
        int use_h_parallel,
        void *code_ptr = nullptr,
        size_t code_size = 1 * Xbyak::DEFAULT_MAX_CODE_SIZE)
        : jit_generator(code_ptr, code_size)
        , nalphabeta(-2*A*B)
        , use_h_parallelism(use_h_parallel)
    {
        this->preamble();

        mov(src, ptr[param + 0]);
        mov(diffdst, ptr[param + 8]);
        mov(workspace0, ptr[param + 16]);
        mov(workspace1, ptr[param + 24]);
        mov(diffsrc, ptr[param + 32]);

        W = J.W;
        HW = J.H*J.W;
        int LSB = this->use_h_parallelism ? W : HW;

        sub(t, BWD_RBC*BUFFER_BLOCK);
        mov(imm_addr64, float2int(this->nalphabeta));
        movq(xnalphabeta, imm_addr64);
        vbroadcastss(znalphabeta, xnalphabeta);

        is_first = J.version == -1 || J.version == -2;
        is_last  = J.version == +1 || J.version == +2;
        is_single = J.version == 3;

        if (is_first || is_single) {
            vxorps(xmm1, xmm1, xmm1);
            for(int irb = 0; irb < BWD_RBC; irb++) {
                vmovups(ptr[t + irb*BUFFER_BLOCK], xmm1);
            }
        }
        if (is_last || is_single) {
            vxorps(xmm1, xmm1, xmm1);
            for(int irb = 0; irb < BWD_RBC; irb++) {
                vmovups(ptr[t + irb*BUFFER_BLOCK + BUFFER_NEXT_OFFSET], xmm1);
            }
        }

        int LSREST = LSB % BWD_RBC;
        int LS = LSB - LSREST;

        Label lrn_loop;

        if (LS > 0) {
            mov(hw, LS);

            L(lrn_loop);
            {
                compute_loop(BWD_RBC, 1, 1);

                add(src, BWD_RBC*vlen);
                add(diffsrc, BWD_RBC*vlen);
                add(diffdst, BWD_RBC*vlen);
                add(workspace0, BWD_RBC*vlen);
                add(workspace1, BWD_RBC*vlen);

                for(int irb = 0; irb < BWD_RBC; irb++)
                    dec(hw);
                cmp(hw, 0);
                jne(lrn_loop, T_NEAR);
            }
        }

        compute_loop(LSREST, 1, this->use_h_parallelism ? 0 : 1);

        add(t, BWD_RBC*BUFFER_BLOCK);
        this->postamble();

        ker = reinterpret_cast<decltype(ker)>(const_cast<uint8_t*>(
                    this->getCode()));
    }

};

status_t jit_avx512_common_lrn_bwd_t::pd_t::init() {
    using namespace alg_kind;

    const memory_desc_wrapper data_d(src_md());
    bool ok = true
        && mayiuse(avx512_common)
        && !is_fwd()
        && utils::everyone_is(data_type::f32, data_d.data_type())
        && !has_zero_dim_memory()
        && data_d.ndims() == 4
        && data_d.dims()[1] % vsize == 0
        && attr()->has_default_values();
    if (!ok) return unimplemented;

    dims_t ws_dims = { MB(), C(), H(), 2*W() };
    mkldnn_memory_desc_init_by_tag(&ws_md_, 4, ws_dims, data_type::f32,
            format_tag::nChw16c);

    if (!compare_ws(hint_fwd_pd_)) return unimplemented;

    bool args_ok_across = true
        && desc()->alg_kind == lrn_across_channels
        && desc()->local_size == 5
        && desc()->lrn_beta == 0.75
        && data_d.matches_tag(format_tag::nChw16c);

    return args_ok_across ? success : unimplemented;
}

jit_avx512_common_lrn_bwd_t::jit_avx512_common_lrn_bwd_t(const pd_t *apd)
    : cpu_primitive_t(apd)
    , use_h_parallelism(0),  ker_(nullptr), ker_first_(nullptr)
    , ker_last_(nullptr) {
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();
    const int ls = pd()->desc()->local_size;
    const float alpha = pd()->desc()->lrn_alpha / ls;
    const float beta = pd()->desc()->lrn_beta;

    use_h_parallelism = H > 28 ? 1 : 0;

    if (C / vsize == 1) {
        ker_ = new jit_avx512_common_lrn_kernel_f32(nChw16c_across(H, W, 3),
        alpha, beta, use_h_parallelism);
    } else {
        ker_ = new jit_avx512_common_lrn_kernel_f32(nChw16c_across(H, W, 0),
            alpha, beta, use_h_parallelism);
        ker_first_ = new jit_avx512_common_lrn_kernel_f32(
            nChw16c_across(H, W, -1), alpha, beta, use_h_parallelism);
        ker_last_ = new jit_avx512_common_lrn_kernel_f32(
            nChw16c_across(H, W, +1), alpha, beta, use_h_parallelism);
    }
}

jit_avx512_common_lrn_bwd_t::~jit_avx512_common_lrn_bwd_t()
{ delete ker_; delete ker_first_; delete ker_last_; }

void jit_avx512_common_lrn_bwd_t::execute_backward(const exec_ctx_t &ctx) const
{
    auto src = CTX_IN_MEM(const data_t *, MKLDNN_ARG_SRC);
    auto diff_dst = CTX_IN_MEM(const data_t *, MKLDNN_ARG_DIFF_DST);
    auto ws = CTX_IN_MEM(const data_t *, MKLDNN_ARG_WORKSPACE);
    auto diff_src = CTX_OUT_MEM(data_t *, MKLDNN_ARG_DIFF_SRC);

    const int N = pd()->MB();
    const int C = pd()->C();
    const int H = pd()->H();
    const int W = pd()->W();

    parallel(0, [&](const int ithr, const int nthr) {
        size_t start{0}, end{0};
        const int C16 = C / vsize;
        const size_t work_amount = use_h_parallelism ? N*C16*H : N*C16;

        balance211(work_amount, nthr, ithr, start, end);
        if (use_h_parallelism) {
            int n{0}, c16{0}, h{0};
            nd_iterator_init(start, n, N,  h, H, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize
                    + h*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize
                    + h*2*W*vsize;
                auto ws_offset1 = ws_offset0 + W*vsize;

                jit_args_bwd_t args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);
                nd_iterator_step(n, N, h, H, c16, C16);
            }
        } else {
            int n{0}, c16{0};
            nd_iterator_init(start, n, N, c16, C16);
            for (size_t iwork = start; iwork < end; ++iwork) {
                auto offset = n*C*H*W + c16*H*W*vsize;
                auto ws_offset0 = n*C*H*2*W + c16*H*2*W*vsize;
                auto ws_offset1 = ws_offset0 + H*W*vsize;

                jit_args_bwd_t args;
                args.src = &src[offset];
                args.diff_dst = &diff_dst[offset];
                args.ws0 = &ws[ws_offset0];
                args.ws1 = &ws[ws_offset1];
                args.diff_src = &diff_src[offset];

                if (C16 == 1)
                    (*ker_)(&args);
                else if (c16 == 0)
                    (*ker_first_)(&args);
                else if (c16 == C16 - 1)
                    (*ker_last_)(&args);
                else
                    (*ker_)(&args);

                nd_iterator_step(n, N, c16, C16);
            }
        }
    });
}

}
}
}
