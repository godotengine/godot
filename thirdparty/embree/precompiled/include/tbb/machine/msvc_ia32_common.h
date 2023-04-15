/*
    Copyright (c) 2005-2020 Intel Corporation

    Licensed under the Apache License, Version 2.0 (the "License");
    you may not use this file except in compliance with the License.
    You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
*/

#if !defined(__TBB_machine_H) || defined(__TBB_machine_msvc_ia32_common_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_msvc_ia32_common_H

#include <intrin.h>

//TODO: consider moving this macro to tbb_config.h and using where MSVC asm is used
#if  !_M_X64 || __INTEL_COMPILER
    #define __TBB_X86_MSVC_INLINE_ASM_AVAILABLE 1
#else
    //MSVC in x64 mode does not accept inline assembler
    #define __TBB_X86_MSVC_INLINE_ASM_AVAILABLE 0
    #define __TBB_NO_X86_MSVC_INLINE_ASM_MSG "The compiler being used is not supported (outdated?)"
#endif

#if _M_X64
    #define __TBB_r(reg_name) r##reg_name
    #define __TBB_W(name) name##64
    namespace tbb { namespace internal { namespace msvc_intrinsics {
        typedef __int64 word;
    }}}
#else
    #define __TBB_r(reg_name) e##reg_name
    #define __TBB_W(name) name
    namespace tbb { namespace internal { namespace msvc_intrinsics {
        typedef long word;
    }}}
#endif

#if __TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT
    // S is the operand size in bytes, B is the suffix for intrinsics for that size
    #define __TBB_MACHINE_DEFINE_ATOMICS(S,B,T,U)                                           \
    __pragma(intrinsic( _InterlockedCompareExchange##B ))                                   \
    static inline T __TBB_machine_cmpswp##S ( volatile void * ptr, U value, U comparand ) { \
        return _InterlockedCompareExchange##B ( (T*)ptr, value, comparand );                \
    }                                                                                       \
    __pragma(intrinsic( _InterlockedExchangeAdd##B ))                                       \
    static inline T __TBB_machine_fetchadd##S ( volatile void * ptr, U addend ) {           \
        return _InterlockedExchangeAdd##B ( (T*)ptr, addend );                              \
    }                                                                                       \
    __pragma(intrinsic( _InterlockedExchange##B ))                                          \
    static inline T __TBB_machine_fetchstore##S ( volatile void * ptr, U value ) {          \
        return _InterlockedExchange##B ( (T*)ptr, value );                                  \
    }

    // Atomic intrinsics for 1, 2, and 4 bytes are available for x86 & x64
    __TBB_MACHINE_DEFINE_ATOMICS(1,8,char,__int8)
    __TBB_MACHINE_DEFINE_ATOMICS(2,16,short,__int16)
    __TBB_MACHINE_DEFINE_ATOMICS(4,,long,__int32)

    #if __TBB_WORDSIZE==8
    __TBB_MACHINE_DEFINE_ATOMICS(8,64,__int64,__int64)
    #endif

    #undef __TBB_MACHINE_DEFINE_ATOMICS
#endif /* __TBB_MSVC_PART_WORD_INTERLOCKED_INTRINSICS_PRESENT */

#if _MSC_VER>=1300 || __INTEL_COMPILER>=1100
    #pragma intrinsic(_ReadWriteBarrier)
    #pragma intrinsic(_mm_mfence)
    #define __TBB_compiler_fence()    _ReadWriteBarrier()
    #define __TBB_full_memory_fence() _mm_mfence()
#elif __TBB_X86_MSVC_INLINE_ASM_AVAILABLE
    #define __TBB_compiler_fence()    __asm { __asm nop }
    #define __TBB_full_memory_fence() __asm { __asm mfence }
#else
    #error Unsupported compiler; define __TBB_{control,acquire,release}_consistency_helper to support it
#endif

#define __TBB_control_consistency_helper() __TBB_compiler_fence()
#define __TBB_acquire_consistency_helper() __TBB_compiler_fence()
#define __TBB_release_consistency_helper() __TBB_compiler_fence()

#if (_MSC_VER>=1300) || (__INTEL_COMPILER)
    #pragma intrinsic(_mm_pause)
    namespace tbb { namespace internal { namespace msvc_intrinsics {
        static inline void pause (uintptr_t delay ) {
            for (;delay>0; --delay )
                _mm_pause();
        }
    }}}
    #define __TBB_Pause(V) tbb::internal::msvc_intrinsics::pause(V)
    #define __TBB_SINGLE_PAUSE _mm_pause()
#else
    #if !__TBB_X86_MSVC_INLINE_ASM_AVAILABLE
        #error __TBB_NO_X86_MSVC_INLINE_ASM_MSG
    #endif
    namespace tbb { namespace internal { namespace msvc_inline_asm
        static inline void pause (uintptr_t delay ) {
            _asm
            {
                mov __TBB_r(ax), delay
              __TBB_L1:
                pause
                add __TBB_r(ax), -1
                jne __TBB_L1
            }
            return;
        }
    }}}
    #define __TBB_Pause(V) tbb::internal::msvc_inline_asm::pause(V)
    #define __TBB_SINGLE_PAUSE __asm pause
#endif

#if (_MSC_VER>=1400 && !__INTEL_COMPILER) || (__INTEL_COMPILER>=1200)
// MSVC did not have this intrinsic prior to VC8.
// ICL 11.1 fails to compile a TBB example if __TBB_Log2 uses the intrinsic.
    #pragma intrinsic(__TBB_W(_BitScanReverse))
    namespace tbb { namespace internal { namespace msvc_intrinsics {
        static inline uintptr_t lg_bsr( uintptr_t i ){
            unsigned long j;
            __TBB_W(_BitScanReverse)( &j, i );
            return j;
        }
    }}}
    #define __TBB_Log2(V) tbb::internal::msvc_intrinsics::lg_bsr(V)
#else
    #if !__TBB_X86_MSVC_INLINE_ASM_AVAILABLE
        #error __TBB_NO_X86_MSVC_INLINE_ASM_MSG
    #endif
    namespace tbb { namespace internal { namespace msvc_inline_asm {
        static inline uintptr_t lg_bsr( uintptr_t i ){
            uintptr_t j;
            __asm
            {
                bsr __TBB_r(ax), i
                mov j, __TBB_r(ax)
            }
            return j;
        }
    }}}
    #define __TBB_Log2(V) tbb::internal::msvc_inline_asm::lg_bsr(V)
#endif

#if _MSC_VER>=1400
    #pragma intrinsic(__TBB_W(_InterlockedOr))
    #pragma intrinsic(__TBB_W(_InterlockedAnd))
    namespace tbb { namespace internal { namespace msvc_intrinsics {
        static inline void lock_or( volatile void *operand, intptr_t addend ){
            __TBB_W(_InterlockedOr)((volatile word*)operand, addend);
        }
        static inline void lock_and( volatile void *operand, intptr_t addend ){
            __TBB_W(_InterlockedAnd)((volatile word*)operand, addend);
        }
    }}}
    #define __TBB_AtomicOR(P,V)  tbb::internal::msvc_intrinsics::lock_or(P,V)
    #define __TBB_AtomicAND(P,V) tbb::internal::msvc_intrinsics::lock_and(P,V)
#else
    #if !__TBB_X86_MSVC_INLINE_ASM_AVAILABLE
        #error __TBB_NO_X86_MSVC_INLINE_ASM_MSG
    #endif
    namespace tbb { namespace internal { namespace msvc_inline_asm {
        static inline void lock_or( volatile void *operand, __int32 addend ) {
            __asm
            {
                mov eax, addend
                mov edx, [operand]
                lock or [edx], eax
            }
         }
         static inline void lock_and( volatile void *operand, __int32 addend ) {
            __asm
            {
                mov eax, addend
                mov edx, [operand]
                lock and [edx], eax
            }
         }
    }}}
    #define __TBB_AtomicOR(P,V)  tbb::internal::msvc_inline_asm::lock_or(P,V)
    #define __TBB_AtomicAND(P,V) tbb::internal::msvc_inline_asm::lock_and(P,V)
#endif

#pragma intrinsic(__rdtsc)
namespace tbb { namespace internal { typedef uint64_t machine_tsc_t; } }
static inline tbb::internal::machine_tsc_t __TBB_machine_time_stamp() {
    return __rdtsc();
}
#define __TBB_time_stamp() __TBB_machine_time_stamp()

// API to retrieve/update FPU control setting
#define __TBB_CPU_CTL_ENV_PRESENT 1

namespace tbb { namespace internal { class cpu_ctl_env; } }
#if __TBB_X86_MSVC_INLINE_ASM_AVAILABLE
    inline void __TBB_get_cpu_ctl_env ( tbb::internal::cpu_ctl_env* ctl ) {
        __asm {
            __asm mov     __TBB_r(ax), ctl
            __asm stmxcsr [__TBB_r(ax)]
            __asm fstcw   [__TBB_r(ax)+4]
        }
    }
    inline void __TBB_set_cpu_ctl_env ( const tbb::internal::cpu_ctl_env* ctl ) {
        __asm {
            __asm mov     __TBB_r(ax), ctl
            __asm ldmxcsr [__TBB_r(ax)]
            __asm fldcw   [__TBB_r(ax)+4]
        }
    }
#else
    extern "C" {
        void __TBB_EXPORTED_FUNC __TBB_get_cpu_ctl_env ( tbb::internal::cpu_ctl_env* );
        void __TBB_EXPORTED_FUNC __TBB_set_cpu_ctl_env ( const tbb::internal::cpu_ctl_env* );
    }
#endif

namespace tbb {
namespace internal {
class cpu_ctl_env {
private:
    int         mxcsr;
    short       x87cw;
    static const int MXCSR_CONTROL_MASK = ~0x3f; /* all except last six status bits */
public:
    bool operator!=( const cpu_ctl_env& ctl ) const { return mxcsr != ctl.mxcsr || x87cw != ctl.x87cw; }
    void get_env() {
        __TBB_get_cpu_ctl_env( this );
        mxcsr &= MXCSR_CONTROL_MASK;
    }
    void set_env() const { __TBB_set_cpu_ctl_env( this ); }
};
} // namespace internal
} // namespace tbb

#if !__TBB_WIN8UI_SUPPORT
extern "C" __declspec(dllimport) int __stdcall SwitchToThread( void );
#define __TBB_Yield()  SwitchToThread()
#else
#include<thread>
#define __TBB_Yield()  std::this_thread::yield()
#endif

#undef __TBB_r
#undef __TBB_W
#undef __TBB_word

extern "C" {
    __int8 __TBB_EXPORTED_FUNC __TBB_machine_try_lock_elided (volatile void* ptr);
    void   __TBB_EXPORTED_FUNC __TBB_machine_unlock_elided (volatile void* ptr);

    // 'pause' instruction aborts HLE/RTM transactions
    inline static void __TBB_machine_try_lock_elided_cancel() { __TBB_SINGLE_PAUSE; }

#if __TBB_TSX_INTRINSICS_PRESENT
    #define __TBB_machine_is_in_transaction _xtest
    #define __TBB_machine_begin_transaction _xbegin
    #define __TBB_machine_end_transaction   _xend
    // The value (0xFF) below comes from the
    // Intel(R) 64 and IA-32 Architectures Optimization Reference Manual 12.4.5 lock not free
    #define __TBB_machine_transaction_conflict_abort() _xabort(0xFF)
#else
    __int8           __TBB_EXPORTED_FUNC __TBB_machine_is_in_transaction();
    unsigned __int32 __TBB_EXPORTED_FUNC __TBB_machine_begin_transaction();
    void             __TBB_EXPORTED_FUNC __TBB_machine_end_transaction();
    void             __TBB_EXPORTED_FUNC __TBB_machine_transaction_conflict_abort();
#endif /* __TBB_TSX_INTRINSICS_PRESENT */
}
