/**
 * \file bn_mul.h
 *
 * \brief Multi-precision integer library
 */
/*
 *  Copyright (C) 2006-2015, ARM Limited, All Rights Reserved
 *  SPDX-License-Identifier: Apache-2.0
 *
 *  Licensed under the Apache License, Version 2.0 (the "License"); you may
 *  not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  This file is part of mbed TLS (https://tls.mbed.org)
 */
/*
 *      Multiply source vector [s] with b, add result
 *       to destination vector [d] and set carry c.
 *
 *      Currently supports:
 *
 *         . IA-32 (386+)         . AMD64 / EM64T
 *         . IA-32 (SSE2)         . Motorola 68000
 *         . PowerPC, 32-bit      . MicroBlaze
 *         . PowerPC, 64-bit      . TriCore
 *         . SPARC v8             . ARM v3+
 *         . Alpha                . MIPS32
 *         . C, longlong          . C, generic
 */
#ifndef MBEDTLS_BN_MUL_H
#define MBEDTLS_BN_MUL_H

#include "bignum.h"

#if defined(MBEDTLS_HAVE_ASM)

#ifndef asm
#define asm __asm
#endif

/* armcc5 --gnu defines __GNUC__ but doesn't support GNU's extended asm */
#if defined(__GNUC__) && \
    ( !defined(__ARMCC_VERSION) || __ARMCC_VERSION >= 6000000 )

/*
 * Disable use of the i386 assembly code below if option -O0, to disable all
 * compiler optimisations, is passed, detected with __OPTIMIZE__
 * This is done as the number of registers used in the assembly code doesn't
 * work with the -O0 option.
 */
#if defined(__i386__) && defined(__OPTIMIZE__)

#define MULADDC_INIT                        \
    asm(                                    \
        "movl   %%ebx, %0           \n\t"   \
        "movl   %5, %%esi           \n\t"   \
        "movl   %6, %%edi           \n\t"   \
        "movl   %7, %%ecx           \n\t"   \
        "movl   %8, %%ebx           \n\t"

#define MULADDC_CORE                        \
        "lodsl                      \n\t"   \
        "mull   %%ebx               \n\t"   \
        "addl   %%ecx,   %%eax      \n\t"   \
        "adcl   $0,      %%edx      \n\t"   \
        "addl   (%%edi), %%eax      \n\t"   \
        "adcl   $0,      %%edx      \n\t"   \
        "movl   %%edx,   %%ecx      \n\t"   \
        "stosl                      \n\t"

#if defined(MBEDTLS_HAVE_SSE2)

#define MULADDC_HUIT                            \
        "movd     %%ecx,     %%mm1      \n\t"   \
        "movd     %%ebx,     %%mm0      \n\t"   \
        "movd     (%%edi),   %%mm3      \n\t"   \
        "paddq    %%mm3,     %%mm1      \n\t"   \
        "movd     (%%esi),   %%mm2      \n\t"   \
        "pmuludq  %%mm0,     %%mm2      \n\t"   \
        "movd     4(%%esi),  %%mm4      \n\t"   \
        "pmuludq  %%mm0,     %%mm4      \n\t"   \
        "movd     8(%%esi),  %%mm6      \n\t"   \
        "pmuludq  %%mm0,     %%mm6      \n\t"   \
        "movd     12(%%esi), %%mm7      \n\t"   \
        "pmuludq  %%mm0,     %%mm7      \n\t"   \
        "paddq    %%mm2,     %%mm1      \n\t"   \
        "movd     4(%%edi),  %%mm3      \n\t"   \
        "paddq    %%mm4,     %%mm3      \n\t"   \
        "movd     8(%%edi),  %%mm5      \n\t"   \
        "paddq    %%mm6,     %%mm5      \n\t"   \
        "movd     12(%%edi), %%mm4      \n\t"   \
        "paddq    %%mm4,     %%mm7      \n\t"   \
        "movd     %%mm1,     (%%edi)    \n\t"   \
        "movd     16(%%esi), %%mm2      \n\t"   \
        "pmuludq  %%mm0,     %%mm2      \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "movd     20(%%esi), %%mm4      \n\t"   \
        "pmuludq  %%mm0,     %%mm4      \n\t"   \
        "paddq    %%mm3,     %%mm1      \n\t"   \
        "movd     24(%%esi), %%mm6      \n\t"   \
        "pmuludq  %%mm0,     %%mm6      \n\t"   \
        "movd     %%mm1,     4(%%edi)   \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "movd     28(%%esi), %%mm3      \n\t"   \
        "pmuludq  %%mm0,     %%mm3      \n\t"   \
        "paddq    %%mm5,     %%mm1      \n\t"   \
        "movd     16(%%edi), %%mm5      \n\t"   \
        "paddq    %%mm5,     %%mm2      \n\t"   \
        "movd     %%mm1,     8(%%edi)   \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "paddq    %%mm7,     %%mm1      \n\t"   \
        "movd     20(%%edi), %%mm5      \n\t"   \
        "paddq    %%mm5,     %%mm4      \n\t"   \
        "movd     %%mm1,     12(%%edi)  \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "paddq    %%mm2,     %%mm1      \n\t"   \
        "movd     24(%%edi), %%mm5      \n\t"   \
        "paddq    %%mm5,     %%mm6      \n\t"   \
        "movd     %%mm1,     16(%%edi)  \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "paddq    %%mm4,     %%mm1      \n\t"   \
        "movd     28(%%edi), %%mm5      \n\t"   \
        "paddq    %%mm5,     %%mm3      \n\t"   \
        "movd     %%mm1,     20(%%edi)  \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "paddq    %%mm6,     %%mm1      \n\t"   \
        "movd     %%mm1,     24(%%edi)  \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "paddq    %%mm3,     %%mm1      \n\t"   \
        "movd     %%mm1,     28(%%edi)  \n\t"   \
        "addl     $32,       %%edi      \n\t"   \
        "addl     $32,       %%esi      \n\t"   \
        "psrlq    $32,       %%mm1      \n\t"   \
        "movd     %%mm1,     %%ecx      \n\t"

#define MULADDC_STOP                    \
        "emms                   \n\t"   \
        "movl   %4, %%ebx       \n\t"   \
        "movl   %%ecx, %1       \n\t"   \
        "movl   %%edi, %2       \n\t"   \
        "movl   %%esi, %3       \n\t"   \
        : "=m" (t), "=m" (c), "=m" (d), "=m" (s)        \
        : "m" (t), "m" (s), "m" (d), "m" (c), "m" (b)   \
        : "eax", "ebx", "ecx", "edx", "esi", "edi"      \
    );

#else

#define MULADDC_STOP                    \
        "movl   %4, %%ebx       \n\t"   \
        "movl   %%ecx, %1       \n\t"   \
        "movl   %%edi, %2       \n\t"   \
        "movl   %%esi, %3       \n\t"   \
        : "=m" (t), "=m" (c), "=m" (d), "=m" (s)        \
        : "m" (t), "m" (s), "m" (d), "m" (c), "m" (b)   \
        : "eax", "ebx", "ecx", "edx", "esi", "edi"      \
    );
#endif /* SSE2 */
#endif /* i386 */

#if defined(__amd64__) || defined (__x86_64__)

#define MULADDC_INIT                        \
    asm(                                    \
        "xorq   %%r8, %%r8\n"

#define MULADDC_CORE                        \
        "movq   (%%rsi), %%rax\n"           \
        "mulq   %%rbx\n"                    \
        "addq   $8, %%rsi\n"                \
        "addq   %%rcx, %%rax\n"             \
        "movq   %%r8, %%rcx\n"              \
        "adcq   $0, %%rdx\n"                \
        "nop    \n"                         \
        "addq   %%rax, (%%rdi)\n"           \
        "adcq   %%rdx, %%rcx\n"             \
        "addq   $8, %%rdi\n"

#define MULADDC_STOP                        \
        : "+c" (c), "+D" (d), "+S" (s)      \
        : "b" (b)                           \
        : "rax", "rdx", "r8"                \
    );

#endif /* AMD64 */

#if defined(__mc68020__) || defined(__mcpu32__)

#define MULADDC_INIT                    \
    asm(                                \
        "movl   %3, %%a2        \n\t"   \
        "movl   %4, %%a3        \n\t"   \
        "movl   %5, %%d3        \n\t"   \
        "movl   %6, %%d2        \n\t"   \
        "moveq  #0, %%d0        \n\t"

#define MULADDC_CORE                    \
        "movel  %%a2@+, %%d1    \n\t"   \
        "mulul  %%d2, %%d4:%%d1 \n\t"   \
        "addl   %%d3, %%d1      \n\t"   \
        "addxl  %%d0, %%d4      \n\t"   \
        "moveq  #0,   %%d3      \n\t"   \
        "addl   %%d1, %%a3@+    \n\t"   \
        "addxl  %%d4, %%d3      \n\t"

#define MULADDC_STOP                    \
        "movl   %%d3, %0        \n\t"   \
        "movl   %%a3, %1        \n\t"   \
        "movl   %%a2, %2        \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "d0", "d1", "d2", "d3", "d4", "a2", "a3"  \
    );

#define MULADDC_HUIT                        \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d4:%%d1  \n\t"   \
        "addxl  %%d3,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d4       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d3:%%d1  \n\t"   \
        "addxl  %%d4,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d3       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d4:%%d1  \n\t"   \
        "addxl  %%d3,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d4       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d3:%%d1  \n\t"   \
        "addxl  %%d4,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d3       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d4:%%d1  \n\t"   \
        "addxl  %%d3,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d4       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d3:%%d1  \n\t"   \
        "addxl  %%d4,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d3       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d4:%%d1  \n\t"   \
        "addxl  %%d3,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d4       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "movel  %%a2@+,  %%d1       \n\t"   \
        "mulul  %%d2,    %%d3:%%d1  \n\t"   \
        "addxl  %%d4,    %%d1       \n\t"   \
        "addxl  %%d0,    %%d3       \n\t"   \
        "addl   %%d1,    %%a3@+     \n\t"   \
        "addxl  %%d0,    %%d3       \n\t"

#endif /* MC68000 */

#if defined(__powerpc64__) || defined(__ppc64__)

#if defined(__MACH__) && defined(__APPLE__)

#define MULADDC_INIT                        \
    asm(                                    \
        "ld     r3, %3              \n\t"   \
        "ld     r4, %4              \n\t"   \
        "ld     r5, %5              \n\t"   \
        "ld     r6, %6              \n\t"   \
        "addi   r3, r3, -8          \n\t"   \
        "addi   r4, r4, -8          \n\t"   \
        "addic  r5, r5,  0          \n\t"

#define MULADDC_CORE                        \
        "ldu    r7, 8(r3)           \n\t"   \
        "mulld  r8, r7, r6          \n\t"   \
        "mulhdu r9, r7, r6          \n\t"   \
        "adde   r8, r8, r5          \n\t"   \
        "ld     r7, 8(r4)           \n\t"   \
        "addze  r5, r9              \n\t"   \
        "addc   r8, r8, r7          \n\t"   \
        "stdu   r8, 8(r4)           \n\t"

#define MULADDC_STOP                        \
        "addze  r5, r5              \n\t"   \
        "addi   r4, r4, 8           \n\t"   \
        "addi   r3, r3, 8           \n\t"   \
        "std    r5, %0              \n\t"   \
        "std    r4, %1              \n\t"   \
        "std    r3, %2              \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "r3", "r4", "r5", "r6", "r7", "r8", "r9"  \
    );


#else /* __MACH__ && __APPLE__ */

#define MULADDC_INIT                        \
    asm(                                    \
        "ld     %%r3, %3            \n\t"   \
        "ld     %%r4, %4            \n\t"   \
        "ld     %%r5, %5            \n\t"   \
        "ld     %%r6, %6            \n\t"   \
        "addi   %%r3, %%r3, -8      \n\t"   \
        "addi   %%r4, %%r4, -8      \n\t"   \
        "addic  %%r5, %%r5,  0      \n\t"

#define MULADDC_CORE                        \
        "ldu    %%r7, 8(%%r3)       \n\t"   \
        "mulld  %%r8, %%r7, %%r6    \n\t"   \
        "mulhdu %%r9, %%r7, %%r6    \n\t"   \
        "adde   %%r8, %%r8, %%r5    \n\t"   \
        "ld     %%r7, 8(%%r4)       \n\t"   \
        "addze  %%r5, %%r9          \n\t"   \
        "addc   %%r8, %%r8, %%r7    \n\t"   \
        "stdu   %%r8, 8(%%r4)       \n\t"

#define MULADDC_STOP                        \
        "addze  %%r5, %%r5          \n\t"   \
        "addi   %%r4, %%r4, 8       \n\t"   \
        "addi   %%r3, %%r3, 8       \n\t"   \
        "std    %%r5, %0            \n\t"   \
        "std    %%r4, %1            \n\t"   \
        "std    %%r3, %2            \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "r3", "r4", "r5", "r6", "r7", "r8", "r9"  \
    );

#endif /* __MACH__ && __APPLE__ */

#elif defined(__powerpc__) || defined(__ppc__) /* end PPC64/begin PPC32  */

#if defined(__MACH__) && defined(__APPLE__)

#define MULADDC_INIT                    \
    asm(                                \
        "lwz    r3, %3          \n\t"   \
        "lwz    r4, %4          \n\t"   \
        "lwz    r5, %5          \n\t"   \
        "lwz    r6, %6          \n\t"   \
        "addi   r3, r3, -4      \n\t"   \
        "addi   r4, r4, -4      \n\t"   \
        "addic  r5, r5,  0      \n\t"

#define MULADDC_CORE                    \
        "lwzu   r7, 4(r3)       \n\t"   \
        "mullw  r8, r7, r6      \n\t"   \
        "mulhwu r9, r7, r6      \n\t"   \
        "adde   r8, r8, r5      \n\t"   \
        "lwz    r7, 4(r4)       \n\t"   \
        "addze  r5, r9          \n\t"   \
        "addc   r8, r8, r7      \n\t"   \
        "stwu   r8, 4(r4)       \n\t"

#define MULADDC_STOP                    \
        "addze  r5, r5          \n\t"   \
        "addi   r4, r4, 4       \n\t"   \
        "addi   r3, r3, 4       \n\t"   \
        "stw    r5, %0          \n\t"   \
        "stw    r4, %1          \n\t"   \
        "stw    r3, %2          \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "r3", "r4", "r5", "r6", "r7", "r8", "r9"  \
    );

#else /* __MACH__ && __APPLE__ */

#define MULADDC_INIT                        \
    asm(                                    \
        "lwz    %%r3, %3            \n\t"   \
        "lwz    %%r4, %4            \n\t"   \
        "lwz    %%r5, %5            \n\t"   \
        "lwz    %%r6, %6            \n\t"   \
        "addi   %%r3, %%r3, -4      \n\t"   \
        "addi   %%r4, %%r4, -4      \n\t"   \
        "addic  %%r5, %%r5,  0      \n\t"

#define MULADDC_CORE                        \
        "lwzu   %%r7, 4(%%r3)       \n\t"   \
        "mullw  %%r8, %%r7, %%r6    \n\t"   \
        "mulhwu %%r9, %%r7, %%r6    \n\t"   \
        "adde   %%r8, %%r8, %%r5    \n\t"   \
        "lwz    %%r7, 4(%%r4)       \n\t"   \
        "addze  %%r5, %%r9          \n\t"   \
        "addc   %%r8, %%r8, %%r7    \n\t"   \
        "stwu   %%r8, 4(%%r4)       \n\t"

#define MULADDC_STOP                        \
        "addze  %%r5, %%r5          \n\t"   \
        "addi   %%r4, %%r4, 4       \n\t"   \
        "addi   %%r3, %%r3, 4       \n\t"   \
        "stw    %%r5, %0            \n\t"   \
        "stw    %%r4, %1            \n\t"   \
        "stw    %%r3, %2            \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "r3", "r4", "r5", "r6", "r7", "r8", "r9"  \
    );

#endif /* __MACH__ && __APPLE__ */

#endif /* PPC32 */

/*
 * The Sparc(64) assembly is reported to be broken.
 * Disable it for now, until we're able to fix it.
 */
#if 0 && defined(__sparc__)
#if defined(__sparc64__)

#define MULADDC_INIT                                    \
    asm(                                                \
                "ldx     %3, %%o0               \n\t"   \
                "ldx     %4, %%o1               \n\t"   \
                "ld      %5, %%o2               \n\t"   \
                "ld      %6, %%o3               \n\t"

#define MULADDC_CORE                                    \
                "ld      [%%o0], %%o4           \n\t"   \
                "inc     4, %%o0                \n\t"   \
                "ld      [%%o1], %%o5           \n\t"   \
                "umul    %%o3, %%o4, %%o4       \n\t"   \
                "addcc   %%o4, %%o2, %%o4       \n\t"   \
                "rd      %%y, %%g1              \n\t"   \
                "addx    %%g1, 0, %%g1          \n\t"   \
                "addcc   %%o4, %%o5, %%o4       \n\t"   \
                "st      %%o4, [%%o1]           \n\t"   \
                "addx    %%g1, 0, %%o2          \n\t"   \
                "inc     4, %%o1                \n\t"

        #define MULADDC_STOP                            \
                "st      %%o2, %0               \n\t"   \
                "stx     %%o1, %1               \n\t"   \
                "stx     %%o0, %2               \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)          \
        : "m" (s), "m" (d), "m" (c), "m" (b)    \
        : "g1", "o0", "o1", "o2", "o3", "o4",   \
          "o5"                                  \
        );

#else /* __sparc64__ */

#define MULADDC_INIT                                    \
    asm(                                                \
                "ld      %3, %%o0               \n\t"   \
                "ld      %4, %%o1               \n\t"   \
                "ld      %5, %%o2               \n\t"   \
                "ld      %6, %%o3               \n\t"

#define MULADDC_CORE                                    \
                "ld      [%%o0], %%o4           \n\t"   \
                "inc     4, %%o0                \n\t"   \
                "ld      [%%o1], %%o5           \n\t"   \
                "umul    %%o3, %%o4, %%o4       \n\t"   \
                "addcc   %%o4, %%o2, %%o4       \n\t"   \
                "rd      %%y, %%g1              \n\t"   \
                "addx    %%g1, 0, %%g1          \n\t"   \
                "addcc   %%o4, %%o5, %%o4       \n\t"   \
                "st      %%o4, [%%o1]           \n\t"   \
                "addx    %%g1, 0, %%o2          \n\t"   \
                "inc     4, %%o1                \n\t"

#define MULADDC_STOP                                    \
                "st      %%o2, %0               \n\t"   \
                "st      %%o1, %1               \n\t"   \
                "st      %%o0, %2               \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)          \
        : "m" (s), "m" (d), "m" (c), "m" (b)    \
        : "g1", "o0", "o1", "o2", "o3", "o4",   \
          "o5"                                  \
        );

#endif /* __sparc64__ */
#endif /* __sparc__ */

#if defined(__microblaze__) || defined(microblaze)

#define MULADDC_INIT                    \
    asm(                                \
        "lwi   r3,   %3         \n\t"   \
        "lwi   r4,   %4         \n\t"   \
        "lwi   r5,   %5         \n\t"   \
        "lwi   r6,   %6         \n\t"   \
        "andi  r7,   r6, 0xffff \n\t"   \
        "bsrli r6,   r6, 16     \n\t"

#define MULADDC_CORE                    \
        "lhui  r8,   r3,   0    \n\t"   \
        "addi  r3,   r3,   2    \n\t"   \
        "lhui  r9,   r3,   0    \n\t"   \
        "addi  r3,   r3,   2    \n\t"   \
        "mul   r10,  r9,  r6    \n\t"   \
        "mul   r11,  r8,  r7    \n\t"   \
        "mul   r12,  r9,  r7    \n\t"   \
        "mul   r13,  r8,  r6    \n\t"   \
        "bsrli  r8, r10,  16    \n\t"   \
        "bsrli  r9, r11,  16    \n\t"   \
        "add   r13, r13,  r8    \n\t"   \
        "add   r13, r13,  r9    \n\t"   \
        "bslli r10, r10,  16    \n\t"   \
        "bslli r11, r11,  16    \n\t"   \
        "add   r12, r12, r10    \n\t"   \
        "addc  r13, r13,  r0    \n\t"   \
        "add   r12, r12, r11    \n\t"   \
        "addc  r13, r13,  r0    \n\t"   \
        "lwi   r10,  r4,   0    \n\t"   \
        "add   r12, r12, r10    \n\t"   \
        "addc  r13, r13,  r0    \n\t"   \
        "add   r12, r12,  r5    \n\t"   \
        "addc   r5, r13,  r0    \n\t"   \
        "swi   r12,  r4,   0    \n\t"   \
        "addi   r4,  r4,   4    \n\t"

#define MULADDC_STOP                    \
        "swi   r5,   %0         \n\t"   \
        "swi   r4,   %1         \n\t"   \
        "swi   r3,   %2         \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "r3", "r4", "r5", "r6", "r7", "r8",       \
          "r9", "r10", "r11", "r12", "r13"          \
    );

#endif /* MicroBlaze */

#if defined(__tricore__)

#define MULADDC_INIT                            \
    asm(                                        \
        "ld.a   %%a2, %3                \n\t"   \
        "ld.a   %%a3, %4                \n\t"   \
        "ld.w   %%d4, %5                \n\t"   \
        "ld.w   %%d1, %6                \n\t"   \
        "xor    %%d5, %%d5              \n\t"

#define MULADDC_CORE                            \
        "ld.w   %%d0,   [%%a2+]         \n\t"   \
        "madd.u %%e2, %%e4, %%d0, %%d1  \n\t"   \
        "ld.w   %%d0,   [%%a3]          \n\t"   \
        "addx   %%d2,    %%d2,  %%d0    \n\t"   \
        "addc   %%d3,    %%d3,    0     \n\t"   \
        "mov    %%d4,    %%d3           \n\t"   \
        "st.w  [%%a3+],  %%d2           \n\t"

#define MULADDC_STOP                            \
        "st.w   %0, %%d4                \n\t"   \
        "st.a   %1, %%a3                \n\t"   \
        "st.a   %2, %%a2                \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)          \
        : "m" (s), "m" (d), "m" (c), "m" (b)    \
        : "d0", "d1", "e2", "d4", "a2", "a3"    \
    );

#endif /* TriCore */

/*
 * Note, gcc -O0 by default uses r7 for the frame pointer, so it complains about
 * our use of r7 below, unless -fomit-frame-pointer is passed.
 *
 * On the other hand, -fomit-frame-pointer is implied by any -Ox options with
 * x !=0, which we can detect using __OPTIMIZE__ (which is also defined by
 * clang and armcc5 under the same conditions).
 *
 * So, only use the optimized assembly below for optimized build, which avoids
 * the build error and is pretty reasonable anyway.
 */
#if defined(__GNUC__) && !defined(__OPTIMIZE__)
#define MULADDC_CANNOT_USE_R7
#endif

#if defined(__arm__) && !defined(MULADDC_CANNOT_USE_R7)

#if defined(__thumb__) && !defined(__thumb2__)

#define MULADDC_INIT                                    \
    asm(                                                \
            "ldr    r0, %3                      \n\t"   \
            "ldr    r1, %4                      \n\t"   \
            "ldr    r2, %5                      \n\t"   \
            "ldr    r3, %6                      \n\t"   \
            "lsr    r7, r3, #16                 \n\t"   \
            "mov    r9, r7                      \n\t"   \
            "lsl    r7, r3, #16                 \n\t"   \
            "lsr    r7, r7, #16                 \n\t"   \
            "mov    r8, r7                      \n\t"

#define MULADDC_CORE                                    \
            "ldmia  r0!, {r6}                   \n\t"   \
            "lsr    r7, r6, #16                 \n\t"   \
            "lsl    r6, r6, #16                 \n\t"   \
            "lsr    r6, r6, #16                 \n\t"   \
            "mov    r4, r8                      \n\t"   \
            "mul    r4, r6                      \n\t"   \
            "mov    r3, r9                      \n\t"   \
            "mul    r6, r3                      \n\t"   \
            "mov    r5, r9                      \n\t"   \
            "mul    r5, r7                      \n\t"   \
            "mov    r3, r8                      \n\t"   \
            "mul    r7, r3                      \n\t"   \
            "lsr    r3, r6, #16                 \n\t"   \
            "add    r5, r5, r3                  \n\t"   \
            "lsr    r3, r7, #16                 \n\t"   \
            "add    r5, r5, r3                  \n\t"   \
            "add    r4, r4, r2                  \n\t"   \
            "mov    r2, #0                      \n\t"   \
            "adc    r5, r2                      \n\t"   \
            "lsl    r3, r6, #16                 \n\t"   \
            "add    r4, r4, r3                  \n\t"   \
            "adc    r5, r2                      \n\t"   \
            "lsl    r3, r7, #16                 \n\t"   \
            "add    r4, r4, r3                  \n\t"   \
            "adc    r5, r2                      \n\t"   \
            "ldr    r3, [r1]                    \n\t"   \
            "add    r4, r4, r3                  \n\t"   \
            "adc    r2, r5                      \n\t"   \
            "stmia  r1!, {r4}                   \n\t"

#define MULADDC_STOP                                    \
            "str    r2, %0                      \n\t"   \
            "str    r1, %1                      \n\t"   \
            "str    r0, %2                      \n\t"   \
         : "=m" (c),  "=m" (d), "=m" (s)        \
         : "m" (s), "m" (d), "m" (c), "m" (b)   \
         : "r0", "r1", "r2", "r3", "r4", "r5",  \
           "r6", "r7", "r8", "r9", "cc"         \
         );

#elif defined (__ARM_FEATURE_DSP) && (__ARM_FEATURE_DSP == 1)

#define MULADDC_INIT                            \
    asm(

#define MULADDC_CORE                            \
            "ldr    r0, [%0], #4        \n\t"   \
            "ldr    r1, [%1]            \n\t"   \
            "umaal  r1, %2, %3, r0      \n\t"   \
            "str    r1, [%1], #4        \n\t"

#define MULADDC_STOP                            \
         : "=r" (s),  "=r" (d), "=r" (c)        \
         : "r" (b), "0" (s), "1" (d), "2" (c)   \
         : "r0", "r1", "memory"                 \
         );

#else

#define MULADDC_INIT                                    \
    asm(                                                \
            "ldr    r0, %3                      \n\t"   \
            "ldr    r1, %4                      \n\t"   \
            "ldr    r2, %5                      \n\t"   \
            "ldr    r3, %6                      \n\t"

#define MULADDC_CORE                                    \
            "ldr    r4, [r0], #4                \n\t"   \
            "mov    r5, #0                      \n\t"   \
            "ldr    r6, [r1]                    \n\t"   \
            "umlal  r2, r5, r3, r4              \n\t"   \
            "adds   r7, r6, r2                  \n\t"   \
            "adc    r2, r5, #0                  \n\t"   \
            "str    r7, [r1], #4                \n\t"

#define MULADDC_STOP                                    \
            "str    r2, %0                      \n\t"   \
            "str    r1, %1                      \n\t"   \
            "str    r0, %2                      \n\t"   \
         : "=m" (c),  "=m" (d), "=m" (s)        \
         : "m" (s), "m" (d), "m" (c), "m" (b)   \
         : "r0", "r1", "r2", "r3", "r4", "r5",  \
           "r6", "r7", "cc"                     \
         );

#endif /* Thumb */

#endif /* ARMv3 */

#if defined(__alpha__)

#define MULADDC_INIT                    \
    asm(                                \
        "ldq    $1, %3          \n\t"   \
        "ldq    $2, %4          \n\t"   \
        "ldq    $3, %5          \n\t"   \
        "ldq    $4, %6          \n\t"

#define MULADDC_CORE                    \
        "ldq    $6,  0($1)      \n\t"   \
        "addq   $1,  8, $1      \n\t"   \
        "mulq   $6, $4, $7      \n\t"   \
        "umulh  $6, $4, $6      \n\t"   \
        "addq   $7, $3, $7      \n\t"   \
        "cmpult $7, $3, $3      \n\t"   \
        "ldq    $5,  0($2)      \n\t"   \
        "addq   $7, $5, $7      \n\t"   \
        "cmpult $7, $5, $5      \n\t"   \
        "stq    $7,  0($2)      \n\t"   \
        "addq   $2,  8, $2      \n\t"   \
        "addq   $6, $3, $3      \n\t"   \
        "addq   $5, $3, $3      \n\t"

#define MULADDC_STOP                                    \
        "stq    $3, %0          \n\t"   \
        "stq    $2, %1          \n\t"   \
        "stq    $1, %2          \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)              \
        : "m" (s), "m" (d), "m" (c), "m" (b)        \
        : "$1", "$2", "$3", "$4", "$5", "$6", "$7"  \
    );
#endif /* Alpha */

#if defined(__mips__) && !defined(__mips64)

#define MULADDC_INIT                    \
    asm(                                \
        "lw     $10, %3         \n\t"   \
        "lw     $11, %4         \n\t"   \
        "lw     $12, %5         \n\t"   \
        "lw     $13, %6         \n\t"

#define MULADDC_CORE                    \
        "lw     $14, 0($10)     \n\t"   \
        "multu  $13, $14        \n\t"   \
        "addi   $10, $10, 4     \n\t"   \
        "mflo   $14             \n\t"   \
        "mfhi   $9              \n\t"   \
        "addu   $14, $12, $14   \n\t"   \
        "lw     $15, 0($11)     \n\t"   \
        "sltu   $12, $14, $12   \n\t"   \
        "addu   $15, $14, $15   \n\t"   \
        "sltu   $14, $15, $14   \n\t"   \
        "addu   $12, $12, $9    \n\t"   \
        "sw     $15, 0($11)     \n\t"   \
        "addu   $12, $12, $14   \n\t"   \
        "addi   $11, $11, 4     \n\t"

#define MULADDC_STOP                    \
        "sw     $12, %0         \n\t"   \
        "sw     $11, %1         \n\t"   \
        "sw     $10, %2         \n\t"   \
        : "=m" (c), "=m" (d), "=m" (s)                      \
        : "m" (s), "m" (d), "m" (c), "m" (b)                \
        : "$9", "$10", "$11", "$12", "$13", "$14", "$15"    \
    );

#endif /* MIPS */
#endif /* GNUC */

#if (defined(_MSC_VER) && defined(_M_IX86)) || defined(__WATCOMC__)

#define MULADDC_INIT                            \
    __asm   mov     esi, s                      \
    __asm   mov     edi, d                      \
    __asm   mov     ecx, c                      \
    __asm   mov     ebx, b

#define MULADDC_CORE                            \
    __asm   lodsd                               \
    __asm   mul     ebx                         \
    __asm   add     eax, ecx                    \
    __asm   adc     edx, 0                      \
    __asm   add     eax, [edi]                  \
    __asm   adc     edx, 0                      \
    __asm   mov     ecx, edx                    \
    __asm   stosd

#if defined(MBEDTLS_HAVE_SSE2)

#define EMIT __asm _emit

#define MULADDC_HUIT                            \
    EMIT 0x0F  EMIT 0x6E  EMIT 0xC9             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0xC3             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x1F             \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCB             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x16             \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xD0             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x66  EMIT 0x04  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xE0             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x76  EMIT 0x08  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xF0             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x7E  EMIT 0x0C  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xF8             \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCA             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x5F  EMIT 0x04  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xDC             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x6F  EMIT 0x08  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xEE             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x67  EMIT 0x0C  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xFC             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x0F             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x56  EMIT 0x10  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xD0             \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x66  EMIT 0x14  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xE0             \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCB             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x76  EMIT 0x18  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xF0             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x04  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x5E  EMIT 0x1C  \
    EMIT 0x0F  EMIT 0xF4  EMIT 0xD8             \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCD             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x6F  EMIT 0x10  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xD5             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x08  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCF             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x6F  EMIT 0x14  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xE5             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x0C  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCA             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x6F  EMIT 0x18  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xF5             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x10  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCC             \
    EMIT 0x0F  EMIT 0x6E  EMIT 0x6F  EMIT 0x1C  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xDD             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x14  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCE             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x18  \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0xD4  EMIT 0xCB             \
    EMIT 0x0F  EMIT 0x7E  EMIT 0x4F  EMIT 0x1C  \
    EMIT 0x83  EMIT 0xC7  EMIT 0x20             \
    EMIT 0x83  EMIT 0xC6  EMIT 0x20             \
    EMIT 0x0F  EMIT 0x73  EMIT 0xD1  EMIT 0x20  \
    EMIT 0x0F  EMIT 0x7E  EMIT 0xC9

#define MULADDC_STOP                            \
    EMIT 0x0F  EMIT 0x77                        \
    __asm   mov     c, ecx                      \
    __asm   mov     d, edi                      \
    __asm   mov     s, esi                      \

#else

#define MULADDC_STOP                            \
    __asm   mov     c, ecx                      \
    __asm   mov     d, edi                      \
    __asm   mov     s, esi                      \

#endif /* SSE2 */
#endif /* MSVC */

#endif /* MBEDTLS_HAVE_ASM */

#if !defined(MULADDC_CORE)
#if defined(MBEDTLS_HAVE_UDBL)

#define MULADDC_INIT                    \
{                                       \
    mbedtls_t_udbl r;                           \
    mbedtls_mpi_uint r0, r1;

#define MULADDC_CORE                    \
    r   = *(s++) * (mbedtls_t_udbl) b;          \
    r0  = (mbedtls_mpi_uint) r;                   \
    r1  = (mbedtls_mpi_uint)( r >> biL );         \
    r0 += c;  r1 += (r0 <  c);          \
    r0 += *d; r1 += (r0 < *d);          \
    c = r1; *(d++) = r0;

#define MULADDC_STOP                    \
}

#else
#define MULADDC_INIT                    \
{                                       \
    mbedtls_mpi_uint s0, s1, b0, b1;              \
    mbedtls_mpi_uint r0, r1, rx, ry;              \
    b0 = ( b << biH ) >> biH;           \
    b1 = ( b >> biH );

#define MULADDC_CORE                    \
    s0 = ( *s << biH ) >> biH;          \
    s1 = ( *s >> biH ); s++;            \
    rx = s0 * b1; r0 = s0 * b0;         \
    ry = s1 * b0; r1 = s1 * b1;         \
    r1 += ( rx >> biH );                \
    r1 += ( ry >> biH );                \
    rx <<= biH; ry <<= biH;             \
    r0 += rx; r1 += (r0 < rx);          \
    r0 += ry; r1 += (r0 < ry);          \
    r0 +=  c; r1 += (r0 <  c);          \
    r0 += *d; r1 += (r0 < *d);          \
    c = r1; *(d++) = r0;

#define MULADDC_STOP                    \
}

#endif /* C (generic)  */
#endif /* C (longlong) */

#endif /* bn_mul.h */
