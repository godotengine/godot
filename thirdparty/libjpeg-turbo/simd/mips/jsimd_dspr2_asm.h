/*
 * MIPS DSPr2 optimizations for libjpeg-turbo
 *
 * Copyright (C) 2013, MIPS Technologies, Inc., California.
 * Copyright (C) 2018, Matthieu Darbois.
 * All Rights Reserved.
 * Authors:  Teodora Novkovic (teodora.novkovic@imgtec.com)
 *           Darko Laus       (darko.laus@imgtec.com)
 * This software is provided 'as-is', without any express or implied
 * warranty.  In no event will the authors be held liable for any damages
 * arising from the use of this software.
 *
 * Permission is granted to anyone to use this software for any purpose,
 * including commercial applications, and to alter it and redistribute it
 * freely, subject to the following restrictions:
 *
 * 1. The origin of this software must not be misrepresented; you must not
 *    claim that you wrote the original software. If you use this software
 *    in a product, an acknowledgment in the product documentation would be
 *    appreciated but is not required.
 * 2. Altered source versions must be plainly marked as such, and must not be
 *    misrepresented as being the original software.
 * 3. This notice may not be removed or altered from any source distribution.
 */

#define zero  $0
#define AT    $1
#define v0    $2
#define v1    $3
#define a0    $4
#define a1    $5
#define a2    $6
#define a3    $7
#define t0    $8
#define t1    $9
#define t2    $10
#define t3    $11
#define t4    $12
#define t5    $13
#define t6    $14
#define t7    $15
#define s0    $16
#define s1    $17
#define s2    $18
#define s3    $19
#define s4    $20
#define s5    $21
#define s6    $22
#define s7    $23
#define t8    $24
#define t9    $25
#define k0    $26
#define k1    $27
#define gp    $28
#define sp    $29
#define fp    $30
#define s8    $30
#define ra    $31

#define f0    $f0
#define f1    $f1
#define f2    $f2
#define f3    $f3
#define f4    $f4
#define f5    $f5
#define f6    $f6
#define f7    $f7
#define f8    $f8
#define f9    $f9
#define f10   $f10
#define f11   $f11
#define f12   $f12
#define f13   $f13
#define f14   $f14
#define f15   $f15
#define f16   $f16
#define f17   $f17
#define f18   $f18
#define f19   $f19
#define f20   $f20
#define f21   $f21
#define f22   $f22
#define f23   $f23
#define f24   $f24
#define f25   $f25
#define f26   $f26
#define f27   $f27
#define f28   $f28
#define f29   $f29
#define f30   $f30
#define f31   $f31

#ifdef __ELF__
#define HIDDEN_SYMBOL(symbol)  .hidden symbol;
#else
#define HIDDEN_SYMBOL(symbol)
#endif

/*
 * LEAF_MIPS32R2 - declare leaf routine for MIPS32r2
 */
#define LEAF_MIPS32R2(symbol) \
    .globl      symbol; \
    HIDDEN_SYMBOL(symbol) \
    .align      2; \
    .type       symbol, @function; \
    .ent        symbol, 0; \
symbol: \
    .frame      sp, 0, ra; \
    .set        push; \
    .set        arch = mips32r2; \
    .set        noreorder; \
    .set        noat;

/*
 * LEAF_DSPR2 - declare leaf routine for MIPS DSPr2
 */
#define LEAF_DSPR2(symbol) \
LEAF_MIPS32R2(symbol) \
    .set        dspr2;

/*
 * END - mark end of function
 */
#define END(function) \
    .set        pop; \
    .end        function; \
    .size       function, .-function

/*
 * Checks if stack offset is big enough for storing/restoring regs_num
 * number of register to/from stack. Stack offset must be greater than
 * or equal to the number of bytes needed for storing registers (regs_num*4).
 * Since MIPS ABI allows usage of first 16 bytes of stack frame (this is
 * preserved for input arguments of the functions, already stored in a0-a3),
 * stack size can be further optimized by utilizing this space.
 */
.macro CHECK_STACK_OFFSET regs_num, stack_offset
.if \stack_offset < \regs_num * 4 - 16
.error "Stack offset too small."
.endif
.endm

/*
 * Saves set of registers on stack. Maximum number of registers that
 * can be saved on stack is limitted to 14 (a0-a3, v0-v1 and s0-s7).
 * Stack offset is number of bytes that are added to stack pointer (sp)
 * before registers are pushed in order to provide enough space on stack
 * (offset must be multiple of 4, and must be big enough, as described by
 * CHECK_STACK_OFFSET macro). This macro is intended to be used in
 * combination with RESTORE_REGS_FROM_STACK macro. Example:
 *  SAVE_REGS_ON_STACK      4, v0, v1, s0, s1
 *  RESTORE_REGS_FROM_STACK 4, v0, v1, s0, s1
 */
.macro SAVE_REGS_ON_STACK  stack_offset = 0, r1, \
                           r2  = 0, r3  = 0, r4  = 0, \
                           r5  = 0, r6  = 0, r7  = 0, \
                           r8  = 0, r9  = 0, r10 = 0, \
                           r11 = 0, r12 = 0, r13 = 0, \
                           r14 = 0
.if (\stack_offset < 0) || (\stack_offset - (\stack_offset / 4) * 4)
    .error "Stack offset must be pozitive and multiple of 4."
.endif
.if \stack_offset != 0
    addiu       sp, sp, -\stack_offset
.endif
    sw          \r1, 0(sp)
.if \r2 != 0
    sw          \r2, 4(sp)
.endif
.if \r3 != 0
    sw          \r3, 8(sp)
.endif
.if \r4 != 0
    sw          \r4, 12(sp)
.endif
.if \r5 != 0
    CHECK_STACK_OFFSET 5, \stack_offset
    sw          \r5, 16(sp)
.endif
.if \r6 != 0
    CHECK_STACK_OFFSET 6, \stack_offset
    sw          \r6, 20(sp)
.endif
.if \r7 != 0
    CHECK_STACK_OFFSET 7, \stack_offset
    sw          \r7, 24(sp)
.endif
.if \r8 != 0
    CHECK_STACK_OFFSET 8, \stack_offset
    sw          \r8, 28(sp)
.endif
.if \r9 != 0
    CHECK_STACK_OFFSET 9, \stack_offset
    sw          \r9, 32(sp)
.endif
.if \r10 != 0
    CHECK_STACK_OFFSET 10, \stack_offset
    sw          \r10, 36(sp)
.endif
.if \r11 != 0
    CHECK_STACK_OFFSET 11, \stack_offset
    sw          \r11, 40(sp)
.endif
.if \r12 != 0
    CHECK_STACK_OFFSET 12, \stack_offset
    sw          \r12, 44(sp)
.endif
.if \r13 != 0
    CHECK_STACK_OFFSET 13, \stack_offset
    sw          \r13, 48(sp)
.endif
.if \r14 != 0
    CHECK_STACK_OFFSET 14, \stack_offset
    sw          \r14, 52(sp)
.endif
.endm

/*
 * Restores set of registers from stack. Maximum number of registers that
 * can be restored from stack is limitted to 14 (a0-a3, v0-v1 and s0-s7).
 * Stack offset is number of bytes that are added to stack pointer (sp)
 * after registers are restored (offset must be multiple of 4, and must
 * be big enough, as described by CHECK_STACK_OFFSET macro). This macro is
 * intended to be used in combination with RESTORE_REGS_FROM_STACK macro.
 * Example:
 *  SAVE_REGS_ON_STACK      4, v0, v1, s0, s1
 *  RESTORE_REGS_FROM_STACK 4, v0, v1, s0, s1
 */
.macro RESTORE_REGS_FROM_STACK  stack_offset = 0, r1, \
                                r2  = 0, r3  = 0, r4  = 0, \
                                r5  = 0, r6  = 0, r7  = 0, \
                                r8  = 0, r9  = 0, r10 = 0, \
                                r11 = 0, r12 = 0, r13 = 0, \
                                r14 = 0
.if (\stack_offset < 0) || (\stack_offset - (\stack_offset / 4) * 4)
    .error "Stack offset must be pozitive and multiple of 4."
.endif
    lw          \r1, 0(sp)
.if \r2 != 0
    lw          \r2, 4(sp)
.endif
.if \r3 != 0
    lw          \r3, 8(sp)
.endif
.if \r4 != 0
    lw          \r4, 12(sp)
.endif
.if \r5 != 0
    CHECK_STACK_OFFSET 5, \stack_offset
    lw          \r5, 16(sp)
.endif
.if \r6 != 0
    CHECK_STACK_OFFSET 6, \stack_offset
    lw          \r6, 20(sp)
.endif
.if \r7 != 0
    CHECK_STACK_OFFSET 7, \stack_offset
    lw          \r7, 24(sp)
.endif
.if \r8 != 0
    CHECK_STACK_OFFSET 8, \stack_offset
    lw          \r8, 28(sp)
.endif
.if \r9 != 0
    CHECK_STACK_OFFSET 9, \stack_offset
    lw          \r9, 32(sp)
.endif
.if \r10 != 0
    CHECK_STACK_OFFSET 10, \stack_offset
    lw          \r10, 36(sp)
.endif
.if \r11 != 0
    CHECK_STACK_OFFSET 11, \stack_offset
    lw          \r11, 40(sp)
.endif
.if \r12 != 0
    CHECK_STACK_OFFSET 12, \stack_offset
    lw          \r12, 44(sp)
.endif
.if \r13 != 0
    CHECK_STACK_OFFSET 13, \stack_offset
    lw          \r13, 48(sp)
.endif
.if \r14 != 0
    CHECK_STACK_OFFSET 14, \stack_offset
    lw          \r14, 52(sp)
.endif
.if \stack_offset != 0
    addiu       sp, sp, \stack_offset
.endif
.endm
