// Copyright (c) 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// A minimalistic implementation of getcontext() to be used by
// Google Breakpad when getcontext() is not available in libc.

#include "common/linux/ucontext_constants.h"

/* int getcontext (ucontext_t* ucp) */

#if defined(__arm__)

  .text
  .global breakpad_getcontext
  .hidden breakpad_getcontext
  .type breakpad_getcontext, #function
  .align 0
  .fnstart
breakpad_getcontext:

  /* First, save r4-r11 */
  add   r1, r0, #(MCONTEXT_GREGS_OFFSET + 4*4)
  stm   r1, {r4-r11}

  /* r12 is a scratch register, don't save it */

  /* Save sp and lr explicitly. */
  /* - sp can't be stored with stmia in Thumb-2 */
  /* - STM instructions that store sp and pc are deprecated in ARM */
  str   sp, [r0, #(MCONTEXT_GREGS_OFFSET + 13*4)]
  str   lr, [r0, #(MCONTEXT_GREGS_OFFSET + 14*4)]

  /* Save the caller's address in 'pc' */
  str   lr, [r0, #(MCONTEXT_GREGS_OFFSET + 15*4)]

  /* Save ucontext_t* pointer across next call */
  mov   r4, r0

  /* Call sigprocmask(SIG_BLOCK, NULL, &(ucontext->uc_sigmask)) */
  mov   r0, #0  /* SIG_BLOCK */
  mov   r1, #0  /* NULL */
  add   r2, r4, #UCONTEXT_SIGMASK_OFFSET
  bl    sigprocmask(PLT)

  /* Intentionally do not save the FPU state here. This is because on
   * Linux/ARM, one should instead use ptrace(PTRACE_GETFPREGS) or
   * ptrace(PTRACE_GETVFPREGS) to get it.
   *
   * Note that a real implementation of getcontext() would need to save
   * this here to allow setcontext()/swapcontext() to work correctly.
   */

  /* Restore the values of r4 and lr */
  mov   r0, r4
  ldr   lr, [r0, #(MCONTEXT_GREGS_OFFSET + 14*4)]
  ldr   r4, [r0, #(MCONTEXT_GREGS_OFFSET +  4*4)]

  /* Return 0 */
  mov   r0, #0
  bx    lr

  .fnend
  .size breakpad_getcontext, . - breakpad_getcontext

#elif defined(__aarch64__)

#if defined(__ARM_FEATURE_PAC_DEFAULT) && __ARM_FEATURE_PAC_DEFAULT
    // ENABLE_PAUTH must be defined to 1 since this value will be used in
    // bitwise-shift later!
    #define ENABLE_PAUTH 1

    #if ((__ARM_FEATURE_PAC_DEFAULT&((1<<0)|(1<<1)))==0)
        #error Pointer authentication defines no valid key!
    #endif
#else
    #define ENABLE_PAUTH 0
#endif

#if defined(__ARM_FEATURE_BTI_DEFAULT) && (__ARM_FEATURE_BTI_DEFAULT==1)
    // ENABLE_BTI must be defined to 1 since this value will be used in
    // bitwise-shift later!
    #define ENABLE_BTI 1
#else
    #define ENABLE_BTI 0
#endif


// Although Pointer Authentication and Branch Target Instructions are technically
// seperate features they work together, i.e. the paciasp and pacibsp instructions
// serve as BTI landing pads.
// Therefore PA-instructions are enabled when PA _or_ BTI is enabled!
#if ENABLE_PAUTH || ENABLE_BTI
    // See section "Pointer Authentication" of
    // https://developer.arm.com/documentation/101028/0012/5--Feature-test-macros
    // for details how to interpret __ARM_FEATURE_PAC_DEFAULT
    #if (__ARM_FEATURE_PAC_DEFAULT & (1<<0))
        #define PAUTH_SIGN_SP paciasp
        #define PAUTH_AUTH_SP autiasp
    #else
        #define PAUTH_SIGN_SP pacibsp
        #define PAUTH_AUTH_SP autibsp
    #endif
#else
    #define PAUTH_SIGN_SP
    #define PAUTH_AUTH_SP
#endif

#define  _NSIG                       64
#define  __NR_rt_sigprocmask         135

  .text
  .global breakpad_getcontext
  .hidden breakpad_getcontext
  .type breakpad_getcontext, #function
  .align 4
  .cfi_startproc
breakpad_getcontext:

  PAUTH_SIGN_SP

  /* The saved context will return to the getcontext() call point
     with a return value of 0 */
  str     xzr,      [x0, MCONTEXT_GREGS_OFFSET +  0 * REGISTER_SIZE]

  stp     x18, x19, [x0, MCONTEXT_GREGS_OFFSET + 18 * REGISTER_SIZE]
  stp     x20, x21, [x0, MCONTEXT_GREGS_OFFSET + 20 * REGISTER_SIZE]
  stp     x22, x23, [x0, MCONTEXT_GREGS_OFFSET + 22 * REGISTER_SIZE]
  stp     x24, x25, [x0, MCONTEXT_GREGS_OFFSET + 24 * REGISTER_SIZE]
  stp     x26, x27, [x0, MCONTEXT_GREGS_OFFSET + 26 * REGISTER_SIZE]
  stp     x28, x29, [x0, MCONTEXT_GREGS_OFFSET + 28 * REGISTER_SIZE]
  str     x30,      [x0, MCONTEXT_GREGS_OFFSET + 30 * REGISTER_SIZE]

  /* Place LR into the saved PC, this will ensure that when
     switching to this saved context with setcontext() control
     will pass back to the caller of getcontext(), we have
     already arranged to return the appropriate return value in x0
     above.  */
  str     x30, [x0, MCONTEXT_PC_OFFSET]

  /* Save the current SP */
  mov     x2, sp
  str     x2, [x0, MCONTEXT_SP_OFFSET]

  /* Initialize the pstate.  */
  str     xzr, [x0, MCONTEXT_PSTATE_OFFSET]

  /* Figure out where to place the first context extension
     block.  */
  add     x2, x0, #MCONTEXT_EXTENSION_OFFSET

  /* Write the context extension fpsimd header.  */
  mov     w3, #(FPSIMD_MAGIC & 0xffff)
  movk    w3, #(FPSIMD_MAGIC >> 16), lsl #16
  str     w3, [x2, #FPSIMD_CONTEXT_MAGIC_OFFSET]
  mov     w3, #FPSIMD_CONTEXT_SIZE
  str     w3, [x2, #FPSIMD_CONTEXT_SIZE_OFFSET]

  /* Fill in the FP SIMD context.  */
  add     x3, x2, #(FPSIMD_CONTEXT_VREGS_OFFSET + 8 * SIMD_REGISTER_SIZE)
  stp     d8,  d9, [x3], #(2 * SIMD_REGISTER_SIZE)
  stp     d10, d11, [x3], #(2 * SIMD_REGISTER_SIZE)
  stp     d12, d13, [x3], #(2 * SIMD_REGISTER_SIZE)
  stp     d14, d15, [x3], #(2 * SIMD_REGISTER_SIZE)

  add     x3, x2, FPSIMD_CONTEXT_FPSR_OFFSET

  mrs     x4, fpsr
  str     w4, [x3]

  mrs     x4, fpcr
  str     w4, [x3, FPSIMD_CONTEXT_FPCR_OFFSET - FPSIMD_CONTEXT_FPSR_OFFSET]

  /* Write the termination context extension header.  */
  add     x2, x2, #FPSIMD_CONTEXT_SIZE

  str     xzr, [x2, #FPSIMD_CONTEXT_MAGIC_OFFSET]
  str     xzr, [x2, #FPSIMD_CONTEXT_SIZE_OFFSET]

  /* Grab the signal mask */
  /* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
  add     x2, x0, #UCONTEXT_SIGMASK_OFFSET
  mov     x0, #0  /* SIG_BLOCK */
  mov     x1, #0  /* NULL */
  mov     x3, #(_NSIG / 8)
  mov     x8, #__NR_rt_sigprocmask
  svc     0

  /* Return x0 for success */
  mov     x0, 0

  PAUTH_AUTH_SP

  ret

  .cfi_endproc
  .size breakpad_getcontext, . - breakpad_getcontext

#elif defined(__i386__)

  .text
  .global breakpad_getcontext
  .hidden breakpad_getcontext
  .align 4
  .type breakpad_getcontext, @function

breakpad_getcontext:

  movl 4(%esp), %eax   /* eax = uc */

  /* Save register values */
  movl %ecx, MCONTEXT_ECX_OFFSET(%eax)
  movl %edx, MCONTEXT_EDX_OFFSET(%eax)
  movl %ebx, MCONTEXT_EBX_OFFSET(%eax)
  movl %edi, MCONTEXT_EDI_OFFSET(%eax)
  movl %esi, MCONTEXT_ESI_OFFSET(%eax)
  movl %ebp, MCONTEXT_EBP_OFFSET(%eax)

  movl (%esp), %edx   /* return address */
  lea  4(%esp), %ecx  /* exclude return address from stack */
  mov  %edx, MCONTEXT_EIP_OFFSET(%eax)
  mov  %ecx, MCONTEXT_ESP_OFFSET(%eax)

  xorl %ecx, %ecx
  movw %fs, %cx
  mov  %ecx, MCONTEXT_FS_OFFSET(%eax)

  movl $0, MCONTEXT_EAX_OFFSET(%eax)

  /* Save floating point state to fpregstate, then update
   * the fpregs pointer to point to it */
  leal UCONTEXT_FPREGS_MEM_OFFSET(%eax), %ecx
  fnstenv (%ecx)
  fldenv  (%ecx)
  mov %ecx, UCONTEXT_FPREGS_OFFSET(%eax)

  /* Save signal mask: sigprocmask(SIGBLOCK, NULL, &uc->uc_sigmask) */
  leal UCONTEXT_SIGMASK_OFFSET(%eax), %edx
  xorl %ecx, %ecx
  push %edx   /* &uc->uc_sigmask */
  push %ecx   /* NULL */
  push %ecx   /* SIGBLOCK == 0 on i386 */
  call sigprocmask@PLT
  addl $12, %esp

  movl $0, %eax
  ret

  .size breakpad_getcontext, . - breakpad_getcontext

#elif defined(__mips__)

// This implementation is inspired by implementation of getcontext in glibc.
#include <asm-mips/asm.h>
#include <asm-mips/regdef.h>
#if _MIPS_SIM == _ABIO32
#include <asm-mips/fpregdef.h>
#endif

// from asm-mips/asm.h
#if _MIPS_SIM == _ABIO32
#define ALSZ 7
#define ALMASK ~7
#define SZREG 4
#else // _MIPS_SIM != _ABIO32
#define ALSZ 15
#define ALMASK ~15
#define SZREG 8
#endif

#include <asm/unistd.h> // for __NR_rt_sigprocmask

#define _NSIG8 128 / 8
#define SIG_BLOCK 1


  .text
LOCALS_NUM = 1 // save gp on stack
FRAME_SIZE = ((LOCALS_NUM * SZREG) + ALSZ) & ALMASK

GP_FRAME_OFFSET = FRAME_SIZE - (1 * SZREG)
MCONTEXT_REG_SIZE = 8

#if _MIPS_SIM == _ABIO32

NESTED (breakpad_getcontext, FRAME_SIZE, ra)
  .mask	0x00000000, 0
  .fmask 0x00000000, 0

  .set noreorder
  .cpload t9
  .set reorder

  move a2, sp
#define _SP a2

  addiu sp, -FRAME_SIZE
  .cprestore GP_FRAME_OFFSET

  sw s0, (16 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s1, (17 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s2, (18 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s3, (19 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s4, (20 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s5, (21 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s6, (22 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw s7, (23 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw _SP, (29 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw fp, (30 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw ra, (31 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sw ra, MCONTEXT_PC_OFFSET(a0)

#ifdef __mips_hard_float
  s.d fs0, (20 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d fs1, (22 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d fs2, (24 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d fs3, (26 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d fs4, (28 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d fs5, (30 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)

  cfc1 v1, fcr31
  sw v1, MCONTEXT_FPC_CSR(a0)
#endif  // __mips_hard_float

  /* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
  li a3, _NSIG8
  addu a2, a0, UCONTEXT_SIGMASK_OFFSET
  move a1, zero
  li a0, SIG_BLOCK
  li v0, __NR_rt_sigprocmask
  syscall

  addiu sp, FRAME_SIZE
  jr ra

END (breakpad_getcontext)
#else

#ifndef NESTED
/*
 * NESTED - declare nested routine entry point
 */
#define NESTED(symbol, framesize, rpc)  \
    .globl  symbol;                     \
    .align  2;                          \
    .type symbol,@function;             \
    .ent  symbol,0;                     \
symbol:   .frame  sp, framesize, rpc;
#endif

/*
 * END - mark end of function
 */
#ifndef END
# define END(function)                  \
    .end  function;                     \
    .size function,.-function
#endif

/* int getcontext (ucontext_t* ucp) */

NESTED (breakpad_getcontext, FRAME_SIZE, ra)
  .mask   0x10000000, 0
  .fmask  0x00000000, 0

  move  a2, sp
#define _SP a2
  move  a3, gp
#define _GP a3

  daddiu sp, -FRAME_SIZE
  .cpsetup $25, GP_FRAME_OFFSET, breakpad_getcontext

  /* Store a magic flag.  */
  li  v1, 1
  sd v1, (0 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)  /* zero */

  sd s0, (16 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s1, (17 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s2, (18 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s3, (19 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s4, (20 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s5, (21 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s6, (22 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s7, (23 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd _GP, (28 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd _SP, (29 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd s8, (30 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd ra, (31 * MCONTEXT_REG_SIZE + MCONTEXT_GREGS_OFFSET)(a0)
  sd ra, MCONTEXT_PC_OFFSET(a0)

#ifdef __mips_hard_float
  s.d $f24, (24 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f25, (25 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f26, (26 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f27, (27 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f28, (28 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f29, (29 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f30, (30 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)
  s.d $f31, (31 * MCONTEXT_REG_SIZE + MCONTEXT_FPREGS_OFFSET)(a0)

  cfc1  v1, $31
  sw  v1, MCONTEXT_FPC_CSR(a0)
#endif /* __mips_hard_float */

/* rt_sigprocmask (SIG_BLOCK, NULL, &ucp->uc_sigmask, _NSIG8) */
  li  a3, _NSIG8
  daddu a2, a0, UCONTEXT_SIGMASK_OFFSET
  move  a1, zero
  li  a0, SIG_BLOCK

  li  v0, __NR_rt_sigprocmask
  syscall

  .cpreturn
  daddiu sp, FRAME_SIZE
  move  v0, zero
  jr  ra

END (breakpad_getcontext)
#endif // _MIPS_SIM == _ABIO32

#elif defined(__x86_64__)
/* The x64 implementation of breakpad_getcontext was derived in part
   from the implementation of libunwind which requires the following
   notice. */
/* libunwind - a platform-independent unwind library
   Copyright (C) 2008 Google, Inc
	Contributed by Paul Pluzhnikov <ppluzhnikov@google.com>
   Copyright (C) 2010 Konstantin Belousov <kib@freebsd.org>

This file is part of libunwind.

Permission is hereby granted, free of charge, to any person obtaining
a copy of this software and associated documentation files (the
"Software"), to deal in the Software without restriction, including
without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to
the following conditions:

The above copyright notice and this permission notice shall be
included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.  */

  .text
  .global breakpad_getcontext
  .hidden breakpad_getcontext
  .align 4
  .type breakpad_getcontext, @function

breakpad_getcontext:
  .cfi_startproc

  /* Callee saved: RBX, RBP, R12-R15  */
  movq %r12, MCONTEXT_GREGS_R12(%rdi)
  movq %r13, MCONTEXT_GREGS_R13(%rdi)
  movq %r14, MCONTEXT_GREGS_R14(%rdi)
  movq %r15, MCONTEXT_GREGS_R15(%rdi)
  movq %rbp, MCONTEXT_GREGS_RBP(%rdi)
  movq %rbx, MCONTEXT_GREGS_RBX(%rdi)

  /* Save argument registers (not strictly needed, but setcontext
     restores them, so don't restore garbage).  */
  movq %r8,  MCONTEXT_GREGS_R8(%rdi)
  movq %r9,  MCONTEXT_GREGS_R9(%rdi)
  movq %rdi, MCONTEXT_GREGS_RDI(%rdi)
  movq %rsi, MCONTEXT_GREGS_RSI(%rdi)
  movq %rdx, MCONTEXT_GREGS_RDX(%rdi)
  movq %rax, MCONTEXT_GREGS_RAX(%rdi)
  movq %rcx, MCONTEXT_GREGS_RCX(%rdi)

  /* Save fp state (not needed, except for setcontext not
     restoring garbage).  */
  leaq MCONTEXT_FPREGS_MEM(%rdi),%r8
  movq %r8, MCONTEXT_FPREGS_PTR(%rdi)
  fnstenv (%r8)
  stmxcsr FPREGS_OFFSET_MXCSR(%r8)

  leaq 8(%rsp), %rax /* exclude this call.  */
  movq %rax, MCONTEXT_GREGS_RSP(%rdi)

  movq 0(%rsp), %rax
  movq %rax, MCONTEXT_GREGS_RIP(%rdi)

  /* Save signal mask: sigprocmask(SIGBLOCK, NULL, &uc->uc_sigmask) */
  leaq UCONTEXT_SIGMASK_OFFSET(%rdi), %rdx  // arg3
  xorq %rsi, %rsi  // arg2 NULL
  xorq %rdi, %rdi  // arg1 SIGBLOCK == 0
  call sigprocmask@PLT

  /* Always return 0 for success, even if sigprocmask failed. */
  xorl %eax, %eax
  ret
  .cfi_endproc
  .size breakpad_getcontext, . - breakpad_getcontext

#else
#error "This file has not been ported for your CPU!"
#endif

#if defined(__aarch64__)
// ENABLE_PAUTH and ENABLE_BTI would be enabled at the definition
//  of AArch64 specific breakpad_getcontext function
#if ENABLE_PAUTH || ENABLE_BTI
// for further information on the .note.gnu.property section see
// https://github.com/ARM-software/abi-aa/blob/main/aaelf64/aaelf64.rst#program-property
.pushsection .note.gnu.property, "a";
    .balign 8
    .long 4
    .long 0x10
    .long 0x5
    .asciz "GNU"
    .long 0xc0000000 /* GNU_PROPERTY_AARCH64_FEATURE_1_AND */
    .long 4
    .long ((ENABLE_PAUTH)<<1) | ((ENABLE_BTI)<<0) /* PAuth and BTI */
    .long 0
.popsection
#endif
#endif
