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

#ifndef CPU_ISA_TRAITS_HPP
#define CPU_ISA_TRAITS_HPP

#include <type_traits>

#define XBYAK64
#define XBYAK_NO_OP_NAMES
/* in order to make selinux happy memory that would be marked with X-bit should
 * be obtained with mmap */
#define XBYAK_USE_MMAP_ALLOCATOR
#if defined(_MSC_VER) && !defined(__INTEL_COMPILER)
/* turn off `size_t to other-type implicit casting` warning
 * currently we have a lot of jit-generated instructions that
 * take uint32_t, but we pass size_t (e.g. due to using sizeof).
 * FIXME: replace size_t parameters with the appropriate ones */
#pragma warning (disable: 4267)
#endif
#include "xbyak/xbyak.h"
#include "xbyak/xbyak_util.h"

namespace mkldnn {
namespace impl {
namespace cpu {

typedef enum {
    isa_any,
    sse41,
    sse42,
    avx,
    avx2,
    avx512_common,
    avx512_core,
    avx512_core_vnni,
    avx512_mic,
    avx512_mic_4ops,
} cpu_isa_t;

template <cpu_isa_t> struct cpu_isa_traits {}; /* ::vlen -> 32 (for avx2) */

template <> struct cpu_isa_traits<sse42> {
    typedef Xbyak::Xmm Vmm;
    static constexpr int vlen_shift = 4;
    static constexpr int vlen = 16;
    static constexpr int n_vregs = 16;
};
template <> struct cpu_isa_traits<avx> {
    typedef Xbyak::Ymm Vmm;
    static constexpr int vlen_shift = 5;
    static constexpr int vlen = 32;
    static constexpr int n_vregs = 16;
};
template <> struct cpu_isa_traits<avx2>:
    public cpu_isa_traits<avx> {};

template <> struct cpu_isa_traits<avx512_common> {
    typedef Xbyak::Zmm Vmm;
    static constexpr int vlen_shift = 6;
    static constexpr int vlen = 64;
    static constexpr int n_vregs = 32;
};
template <> struct cpu_isa_traits<avx512_core>:
    public cpu_isa_traits<avx512_common> {};

template <> struct cpu_isa_traits<avx512_mic>:
    public cpu_isa_traits<avx512_common> {};

template <> struct cpu_isa_traits<avx512_mic_4ops>:
    public cpu_isa_traits<avx512_common> {};

namespace {

static Xbyak::util::Cpu cpu;
static inline bool mayiuse(const cpu_isa_t cpu_isa) {
    using namespace Xbyak::util;

    switch (cpu_isa) {
    case sse41:
    case sse42:
        // FIXME: SSE4.2 is actually NOT required
        //return cpu.has(Cpu::tSSE42);
        return cpu.has(Cpu::tSSE41);
    case avx:
        return cpu.has(Cpu::tAVX);
    case avx2:
        return cpu.has(Cpu::tAVX2);
    case avx512_common:
        return cpu.has(Cpu::tAVX512F);
    case avx512_core:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512BW)
            && cpu.has(Cpu::tAVX512VL)
            && cpu.has(Cpu::tAVX512DQ);
    case avx512_core_vnni:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512BW)
            && cpu.has(Cpu::tAVX512VL)
            && cpu.has(Cpu::tAVX512DQ)
            && cpu.has(Cpu::tAVX512_VNNI);
    case avx512_mic:
        return true
            && cpu.has(Cpu::tAVX512F)
            && cpu.has(Cpu::tAVX512CD)
            && cpu.has(Cpu::tAVX512ER)
            && cpu.has(Cpu::tAVX512PF);
    case avx512_mic_4ops:
        return true
            && mayiuse(avx512_mic)
            && cpu.has(Cpu::tAVX512_4FMAPS)
            && cpu.has(Cpu::tAVX512_4VNNIW);
    case isa_any:
        return true;
    }
    return false;
}
}

/* whatever is required to generate string literals... */
#include "z_magic.hpp"
#define JIT_IMPL_NAME_HELPER(prefix, isa, suffix_if_any) \
    (isa == sse42 ? prefix STRINGIFY(sse42) : \
    (isa == avx ? prefix STRINGIFY(avx) : \
    (isa == avx2 ? prefix STRINGIFY(avx2) : \
    (isa == avx512_common ? prefix STRINGIFY(avx512_common) : \
    (isa == avx512_core ? prefix STRINGIFY(avx512_core) : \
    (isa == avx512_mic ? prefix STRINGIFY(avx512_mic) : \
    (isa == avx512_mic_4ops ? prefix STRINGIFY(avx512_mic_4ops) : \
    prefix suffix_if_any)))))))

}
}
}

#endif
