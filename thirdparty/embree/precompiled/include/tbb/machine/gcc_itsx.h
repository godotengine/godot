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

#if !defined(__TBB_machine_H) || defined(__TBB_machine_gcc_itsx_H)
#error Do not #include this internal file directly; use public TBB headers instead.
#endif

#define __TBB_machine_gcc_itsx_H

#define __TBB_OP_XACQUIRE 0xF2
#define __TBB_OP_XRELEASE 0xF3
#define __TBB_OP_LOCK     0xF0

#define __TBB_STRINGIZE_INTERNAL(arg) #arg
#define __TBB_STRINGIZE(arg) __TBB_STRINGIZE_INTERNAL(arg)

#ifdef __TBB_x86_64
#define __TBB_r_out "=r"
#else
#define __TBB_r_out "=q"
#endif

inline static uint8_t __TBB_machine_try_lock_elided( volatile uint8_t* lk )
{
    uint8_t value = 1;
    __asm__ volatile (".byte " __TBB_STRINGIZE(__TBB_OP_XACQUIRE)"; lock; xchgb %0, %1;"
                      : __TBB_r_out(value), "=m"(*lk)  : "0"(value), "m"(*lk) : "memory" );
    return uint8_t(value^1);
}

inline static void __TBB_machine_try_lock_elided_cancel()
{
    // 'pause' instruction aborts HLE/RTM transactions
    __asm__ volatile ("pause\n" : : : "memory" );
}

inline static void __TBB_machine_unlock_elided( volatile uint8_t* lk )
{
    __asm__ volatile (".byte " __TBB_STRINGIZE(__TBB_OP_XRELEASE)"; movb $0, %0"
                      : "=m"(*lk) : "m"(*lk) : "memory" );
}

#if __TBB_TSX_INTRINSICS_PRESENT
#include <immintrin.h>

#define __TBB_machine_is_in_transaction _xtest
#define __TBB_machine_begin_transaction _xbegin
#define __TBB_machine_end_transaction   _xend
#define __TBB_machine_transaction_conflict_abort() _xabort(0xff)

#else

/*!
 * Check if the instruction is executed in a transaction or not
 */
inline static bool __TBB_machine_is_in_transaction()
{
    int8_t res = 0;
#if __TBB_x86_32
    __asm__ volatile (".byte 0x0F; .byte 0x01; .byte 0xD6;\n"
                      "setz %0" : "=q"(res) : : "memory" );
#else
    __asm__ volatile (".byte 0x0F; .byte 0x01; .byte 0xD6;\n"
                      "setz %0" : "=r"(res) : : "memory" );
#endif
    return res==0;
}

/*!
 * Enter speculative execution mode.
 * @return -1 on success
 *         abort cause ( or 0 ) on abort
 */
inline static uint32_t __TBB_machine_begin_transaction()
{
    uint32_t res = ~uint32_t(0);   // success value
    __asm__ volatile ("1: .byte  0xC7; .byte 0xF8;\n"           //  XBEGIN <abort-offset>
                      "   .long  2f-1b-6\n"                     //  2f-1b == difference in addresses of start
                                                                //  of XBEGIN and the MOVL
                                                                //  2f - 1b - 6 == that difference minus the size of the
                                                                //  XBEGIN instruction.  This is the abort offset to
                                                                //  2: below.
                      "    jmp   3f\n"                          //  success (leave -1 in res)
                      "2:  movl  %%eax,%0\n"                    //  store failure code in res
                      "3:"
                      :"=r"(res):"0"(res):"memory","%eax");
    return res;
}

/*!
 * Attempt to commit/end transaction
 */
inline static void __TBB_machine_end_transaction()
{
    __asm__ volatile (".byte 0x0F; .byte 0x01; .byte 0xD5" :::"memory");   // XEND
}

/*
 * aborts with code 0xFF (lock already held)
 */
inline static void __TBB_machine_transaction_conflict_abort()
{
    __asm__ volatile (".byte 0xC6; .byte 0xF8; .byte 0xFF" :::"memory");
}

#endif /* __TBB_TSX_INTRINSICS_PRESENT */
