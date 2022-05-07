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

#include <assert.h>

#include "cpu_barrier.hpp"

namespace mkldnn {
namespace impl {
namespace cpu {

namespace simple_barrier {

void generate(jit_generator &code, Xbyak::Reg64 reg_ctx,
            Xbyak::Reg64 reg_nthr) {
#   define BAR_CTR_OFF offsetof(ctx_t, ctr)
#   define BAR_SENSE_OFF offsetof(ctx_t, sense)
    using namespace Xbyak;

    Xbyak::Reg64 reg_tmp = [&]() {
        /* returns register which is neither reg_ctx nor reg_nthr */
        Xbyak::Reg64 regs[] = { util::rax, util::rbx, util::rcx };
        for (size_t i = 0; i < sizeof(regs) / sizeof(regs[0]); ++i)
            if (!utils::one_of(regs[i], reg_ctx, reg_nthr))
                return regs[i];
        return regs[0]; /* should not happen */
    }();

    Label barrier_exit_label, barrier_exit_restore_label, spin_label;

    code.cmp(reg_nthr, 1);
    code.jbe(barrier_exit_label);

    code.push(reg_tmp);

    /* take and save current sense */
    code.mov(reg_tmp, code.ptr[reg_ctx + BAR_SENSE_OFF]);
    code.push(reg_tmp);
    code.mov(reg_tmp, 1);

    if (mayiuse(avx512_mic)) {
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
        code.prefetchwt1(code.ptr[reg_ctx + BAR_CTR_OFF]);
    }

    code.lock(); code.xadd(code.ptr[reg_ctx + BAR_CTR_OFF], reg_tmp);
    code.add(reg_tmp, 1);
    code.cmp(reg_tmp, reg_nthr);
    code.pop(reg_tmp); /* restore previous sense */
    code.jne(spin_label);

    /* the last thread {{{ */
    code.mov(code.qword[reg_ctx + BAR_CTR_OFF], 0); // reset ctx

    // notify waiting threads
    code.not_(reg_tmp);
    code.mov(code.ptr[reg_ctx + BAR_SENSE_OFF], reg_tmp);
    code.jmp(barrier_exit_restore_label);
    /* }}} the last thread */

    code.CodeGenerator::L(spin_label);
    code.pause();
    code.cmp(reg_tmp, code.ptr[reg_ctx + BAR_SENSE_OFF]);
    code.je(spin_label);

    code.CodeGenerator::L(barrier_exit_restore_label);
    code.pop(reg_tmp);

    code.CodeGenerator::L(barrier_exit_label);
#    undef BAR_CTR_OFF
#    undef BAR_SENSE_OFF
}

/** jit barrier generator */
struct jit_t: public jit_generator {
    void (*barrier)(ctx_t *ctx, size_t nthr);

    jit_t() {
        generate(*this, abi_param1, abi_param2);
        ret();
        barrier = reinterpret_cast<decltype(barrier)>(const_cast<uint8_t*>(
                    this->getCode()));
    }

    DECLARE_CPU_JIT_AUX_FUNCTIONS(jit_t)
};

void barrier(ctx_t *ctx, int nthr) {
    static jit_t j; /* XXX: constructed on load ... */
    j.barrier(ctx, nthr);
}

}

}
}
}

// vim: et ts=4 sw=4 cindent cino^=l0,\:0,N-s
