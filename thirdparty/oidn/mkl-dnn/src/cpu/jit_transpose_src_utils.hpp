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

#ifndef CPU_JIT_TRANSPOSE_SRC_HPP
#define CPU_JIT_TRANSPOSE_SRC_HPP

#include "cpu_barrier.hpp"
#include "jit_primitive_conf.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

struct jit_trans_src_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;

        /* 1st conv 4fma: backward by weights */
        int nthr_oc_b; /* number of threads process given src image */
        int tr_src_ih_start, tr_src_ih_end; /* thread's transposition bounds */
        simple_barrier::ctx_t *tr_src_bctx; /* transposition synchronization */
    };

    jit_trans_src_t(const jit_conv_conf_t *conf)
        : conf_(conf), ker_(nullptr) {}
    virtual ~jit_trans_src_t() {}

    void operator()(const ctx_t *ctx)
    { assert(ker_); ker_(ctx); }

    const jit_conv_conf_t *conf_;
    void (*ker_)(const ctx_t *);
};

struct jit_src_transpose_s {
    size_t size;
    const void *src;
    const void *tr_src;
    const void *src_prf;
    const void *tr_src_prf;
};

struct jit_trans_dst_t {
    struct ctx_t {
        const void *src;
        const void *tr_src;
        const void *src_prf;
        const void *tr_src_prf;

        /* 1st conv 4fma: backward by weights */
        int nthr_oc_b; /* number of threads process given src image */
        int tr_src_ih_start, tr_src_ih_end; /* thread's transposition bounds */
        simple_barrier::ctx_t *tr_src_bctx; /* transposition synchronization */
    };

    jit_trans_dst_t(const jit_conv_conf_t *conf)
        : conf_(conf), ker_(nullptr) {}
    virtual ~jit_trans_dst_t() {}

    void operator()(const ctx_t *ctx)
    { assert(ker_); ker_(ctx); }

    const jit_conv_conf_t *conf_;
    void (*ker_)(const ctx_t *);
};

struct jit_transpose4x16_src_t {
    int src_pf0_distance;
    int tr_src_pf0_distance;
    bool src_pf1;
    bool tr_src_pf1;
};

struct jit_transpose4x16_src : public jit_generator {
    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_transpose4x16_src)

    jit_transpose4x16_src(const jit_1x1_conv_conf_t *aparams,
            jit_transpose4x16_src_t *tparams_)
        : params(aparams), tparams(tparams_)
    {
        this->generate();
        jit_ker = (decltype(jit_ker))this->getCode();
    }

    const jit_1x1_conv_conf_t *params;
    const jit_transpose4x16_src_t *tparams;
    void (*jit_ker)(jit_src_transpose_s *);

    void operator()(jit_src_transpose_s *arg) { jit_ker(arg); }

    static const int transpose_size = 4;
private:
    static const int typesize = sizeof(float);

    int src_stride, tr_src_stride;

    Xbyak::Reg64 imm_addr64 = rbx;

    Xbyak::Opmask kF0 = k1;
    Xbyak::Opmask kCC = k2;
    Xbyak::Opmask k33 = k3;
    Xbyak::Opmask kFFFF = k4;

    Xbyak::Zmm vidx01 = zmm31;
    Xbyak::Zmm vidx10 = zmm30;
    Xbyak::Zmm vidx1 = zmm29;
    Xbyak::Zmm vidxP = zmm28;

    Xbyak::Reg64 reg_src = r8;
    Xbyak::Reg64 reg_tr_src = r9;
    Xbyak::Reg64 reg_src_prf = r10;
    Xbyak::Reg64 reg_tr_src_prf = r11;
    Xbyak::Reg64 reg_loop = r12;
    Xbyak::Reg64 reg_tr_src_tmp = r13;
    Xbyak::Reg32 regw_tmp = r14d;

    void transpose_block(int ur, int nrows);
    void transpose(int nrows);
    void generate();
};

jit_trans_src_t *create_trans_src(const jit_conv_conf_t *conf);
jit_trans_dst_t *create_trans_dst(const jit_conv_conf_t *conf);

}
}
}

#endif
