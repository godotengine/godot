/*
 *  AES-NI support functions
 *
 *  Copyright The Mbed TLS Contributors
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
 */

/*
 * [AES-WP] http://software.intel.com/en-us/articles/intel-advanced-encryption-standard-aes-instructions-set
 * [CLMUL-WP] http://software.intel.com/en-us/articles/intel-carry-less-multiplication-instruction-and-its-usage-for-computing-the-gcm-mode/
 */

#include "common.h"

#if defined(MBEDTLS_AESNI_C)

#if defined(__has_feature)
#if __has_feature(memory_sanitizer)
#warning "MBEDTLS_AESNI_C is known to cause spurious error reports with some memory sanitizers as they do not understand the assembly code."
#endif
#endif

#include "mbedtls/aesni.h"

#include <string.h>

#ifndef asm
#define asm __asm
#endif

#if defined(MBEDTLS_HAVE_X86_64)

/*
 * AES-NI support detection routine
 */
int mbedtls_aesni_has_support( unsigned int what )
{
    static int done = 0;
    static unsigned int c = 0;

    if( ! done )
    {
        asm( "movl  $1, %%eax   \n\t"
             "cpuid             \n\t"
             : "=c" (c)
             :
             : "eax", "ebx", "edx" );
        done = 1;
    }

    return( ( c & what ) != 0 );
}

/*
 * Binutils needs to be at least 2.19 to support AES-NI instructions.
 * Unfortunately, a lot of users have a lower version now (2014-04).
 * Emit bytecode directly in order to support "old" version of gas.
 *
 * Opcodes from the Intel architecture reference manual, vol. 3.
 * We always use registers, so we don't need prefixes for memory operands.
 * Operand macros are in gas order (src, dst) as opposed to Intel order
 * (dst, src) in order to blend better into the surrounding assembly code.
 */
#define AESDEC      ".byte 0x66,0x0F,0x38,0xDE,"
#define AESDECLAST  ".byte 0x66,0x0F,0x38,0xDF,"
#define AESENC      ".byte 0x66,0x0F,0x38,0xDC,"
#define AESENCLAST  ".byte 0x66,0x0F,0x38,0xDD,"
#define AESIMC      ".byte 0x66,0x0F,0x38,0xDB,"
#define AESKEYGENA  ".byte 0x66,0x0F,0x3A,0xDF,"
#define PCLMULQDQ   ".byte 0x66,0x0F,0x3A,0x44,"

#define xmm0_xmm0   "0xC0"
#define xmm0_xmm1   "0xC8"
#define xmm0_xmm2   "0xD0"
#define xmm0_xmm3   "0xD8"
#define xmm0_xmm4   "0xE0"
#define xmm1_xmm0   "0xC1"
#define xmm1_xmm2   "0xD1"

/*
 * AES-NI AES-ECB block en(de)cryption
 */
int mbedtls_aesni_crypt_ecb( mbedtls_aes_context *ctx,
                     int mode,
                     const unsigned char input[16],
                     unsigned char output[16] )
{
    asm( "movdqu    (%3), %%xmm0    \n\t" // load input
         "movdqu    (%1), %%xmm1    \n\t" // load round key 0
         "pxor      %%xmm1, %%xmm0  \n\t" // round 0
         "add       $16, %1         \n\t" // point to next round key
         "subl      $1, %0          \n\t" // normal rounds = nr - 1
         "test      %2, %2          \n\t" // mode?
         "jz        2f              \n\t" // 0 = decrypt

         "1:                        \n\t" // encryption loop
         "movdqu    (%1), %%xmm1    \n\t" // load round key
         AESENC     xmm1_xmm0      "\n\t" // do round
         "add       $16, %1         \n\t" // point to next round key
         "subl      $1, %0          \n\t" // loop
         "jnz       1b              \n\t"
         "movdqu    (%1), %%xmm1    \n\t" // load round key
         AESENCLAST xmm1_xmm0      "\n\t" // last round
         "jmp       3f              \n\t"

         "2:                        \n\t" // decryption loop
         "movdqu    (%1), %%xmm1    \n\t"
         AESDEC     xmm1_xmm0      "\n\t" // do round
         "add       $16, %1         \n\t"
         "subl      $1, %0          \n\t"
         "jnz       2b              \n\t"
         "movdqu    (%1), %%xmm1    \n\t" // load round key
         AESDECLAST xmm1_xmm0      "\n\t" // last round

         "3:                        \n\t"
         "movdqu    %%xmm0, (%4)    \n\t" // export output
         :
         : "r" (ctx->nr), "r" (ctx->rk), "r" (mode), "r" (input), "r" (output)
         : "memory", "cc", "xmm0", "xmm1" );


    return( 0 );
}

/*
 * GCM multiplication: c = a times b in GF(2^128)
 * Based on [CLMUL-WP] algorithms 1 (with equation 27) and 5.
 */
void mbedtls_aesni_gcm_mult( unsigned char c[16],
                     const unsigned char a[16],
                     const unsigned char b[16] )
{
    unsigned char aa[16], bb[16], cc[16];
    size_t i;

    /* The inputs are in big-endian order, so byte-reverse them */
    for( i = 0; i < 16; i++ )
    {
        aa[i] = a[15 - i];
        bb[i] = b[15 - i];
    }

    asm( "movdqu (%0), %%xmm0               \n\t" // a1:a0
         "movdqu (%1), %%xmm1               \n\t" // b1:b0

         /*
          * Caryless multiplication xmm2:xmm1 = xmm0 * xmm1
          * using [CLMUL-WP] algorithm 1 (p. 13).
          */
         "movdqa %%xmm1, %%xmm2             \n\t" // copy of b1:b0
         "movdqa %%xmm1, %%xmm3             \n\t" // same
         "movdqa %%xmm1, %%xmm4             \n\t" // same
         PCLMULQDQ xmm0_xmm1 ",0x00         \n\t" // a0*b0 = c1:c0
         PCLMULQDQ xmm0_xmm2 ",0x11         \n\t" // a1*b1 = d1:d0
         PCLMULQDQ xmm0_xmm3 ",0x10         \n\t" // a0*b1 = e1:e0
         PCLMULQDQ xmm0_xmm4 ",0x01         \n\t" // a1*b0 = f1:f0
         "pxor %%xmm3, %%xmm4               \n\t" // e1+f1:e0+f0
         "movdqa %%xmm4, %%xmm3             \n\t" // same
         "psrldq $8, %%xmm4                 \n\t" // 0:e1+f1
         "pslldq $8, %%xmm3                 \n\t" // e0+f0:0
         "pxor %%xmm4, %%xmm2               \n\t" // d1:d0+e1+f1
         "pxor %%xmm3, %%xmm1               \n\t" // c1+e0+f1:c0

         /*
          * Now shift the result one bit to the left,
          * taking advantage of [CLMUL-WP] eq 27 (p. 20)
          */
         "movdqa %%xmm1, %%xmm3             \n\t" // r1:r0
         "movdqa %%xmm2, %%xmm4             \n\t" // r3:r2
         "psllq $1, %%xmm1                  \n\t" // r1<<1:r0<<1
         "psllq $1, %%xmm2                  \n\t" // r3<<1:r2<<1
         "psrlq $63, %%xmm3                 \n\t" // r1>>63:r0>>63
         "psrlq $63, %%xmm4                 \n\t" // r3>>63:r2>>63
         "movdqa %%xmm3, %%xmm5             \n\t" // r1>>63:r0>>63
         "pslldq $8, %%xmm3                 \n\t" // r0>>63:0
         "pslldq $8, %%xmm4                 \n\t" // r2>>63:0
         "psrldq $8, %%xmm5                 \n\t" // 0:r1>>63
         "por %%xmm3, %%xmm1                \n\t" // r1<<1|r0>>63:r0<<1
         "por %%xmm4, %%xmm2                \n\t" // r3<<1|r2>>62:r2<<1
         "por %%xmm5, %%xmm2                \n\t" // r3<<1|r2>>62:r2<<1|r1>>63

         /*
          * Now reduce modulo the GCM polynomial x^128 + x^7 + x^2 + x + 1
          * using [CLMUL-WP] algorithm 5 (p. 20).
          * Currently xmm2:xmm1 holds x3:x2:x1:x0 (already shifted).
          */
         /* Step 2 (1) */
         "movdqa %%xmm1, %%xmm3             \n\t" // x1:x0
         "movdqa %%xmm1, %%xmm4             \n\t" // same
         "movdqa %%xmm1, %%xmm5             \n\t" // same
         "psllq $63, %%xmm3                 \n\t" // x1<<63:x0<<63 = stuff:a
         "psllq $62, %%xmm4                 \n\t" // x1<<62:x0<<62 = stuff:b
         "psllq $57, %%xmm5                 \n\t" // x1<<57:x0<<57 = stuff:c

         /* Step 2 (2) */
         "pxor %%xmm4, %%xmm3               \n\t" // stuff:a+b
         "pxor %%xmm5, %%xmm3               \n\t" // stuff:a+b+c
         "pslldq $8, %%xmm3                 \n\t" // a+b+c:0
         "pxor %%xmm3, %%xmm1               \n\t" // x1+a+b+c:x0 = d:x0

         /* Steps 3 and 4 */
         "movdqa %%xmm1,%%xmm0              \n\t" // d:x0
         "movdqa %%xmm1,%%xmm4              \n\t" // same
         "movdqa %%xmm1,%%xmm5              \n\t" // same
         "psrlq $1, %%xmm0                  \n\t" // e1:x0>>1 = e1:e0'
         "psrlq $2, %%xmm4                  \n\t" // f1:x0>>2 = f1:f0'
         "psrlq $7, %%xmm5                  \n\t" // g1:x0>>7 = g1:g0'
         "pxor %%xmm4, %%xmm0               \n\t" // e1+f1:e0'+f0'
         "pxor %%xmm5, %%xmm0               \n\t" // e1+f1+g1:e0'+f0'+g0'
         // e0'+f0'+g0' is almost e0+f0+g0, ex\tcept for some missing
         // bits carried from d. Now get those\t bits back in.
         "movdqa %%xmm1,%%xmm3              \n\t" // d:x0
         "movdqa %%xmm1,%%xmm4              \n\t" // same
         "movdqa %%xmm1,%%xmm5              \n\t" // same
         "psllq $63, %%xmm3                 \n\t" // d<<63:stuff
         "psllq $62, %%xmm4                 \n\t" // d<<62:stuff
         "psllq $57, %%xmm5                 \n\t" // d<<57:stuff
         "pxor %%xmm4, %%xmm3               \n\t" // d<<63+d<<62:stuff
         "pxor %%xmm5, %%xmm3               \n\t" // missing bits of d:stuff
         "psrldq $8, %%xmm3                 \n\t" // 0:missing bits of d
         "pxor %%xmm3, %%xmm0               \n\t" // e1+f1+g1:e0+f0+g0
         "pxor %%xmm1, %%xmm0               \n\t" // h1:h0
         "pxor %%xmm2, %%xmm0               \n\t" // x3+h1:x2+h0

         "movdqu %%xmm0, (%2)               \n\t" // done
         :
         : "r" (aa), "r" (bb), "r" (cc)
         : "memory", "cc", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5" );

    /* Now byte-reverse the outputs */
    for( i = 0; i < 16; i++ )
        c[i] = cc[15 - i];

    return;
}

/*
 * Compute decryption round keys from encryption round keys
 */
void mbedtls_aesni_inverse_key( unsigned char *invkey,
                        const unsigned char *fwdkey, int nr )
{
    unsigned char *ik = invkey;
    const unsigned char *fk = fwdkey + 16 * nr;

    memcpy( ik, fk, 16 );

    for( fk -= 16, ik += 16; fk > fwdkey; fk -= 16, ik += 16 )
        asm( "movdqu (%0), %%xmm0       \n\t"
             AESIMC  xmm0_xmm0         "\n\t"
             "movdqu %%xmm0, (%1)       \n\t"
             :
             : "r" (fk), "r" (ik)
             : "memory", "xmm0" );

    memcpy( ik, fk, 16 );
}

/*
 * Key expansion, 128-bit case
 */
static void aesni_setkey_enc_128( unsigned char *rk,
                                  const unsigned char *key )
{
    asm( "movdqu (%1), %%xmm0               \n\t" // copy the original key
         "movdqu %%xmm0, (%0)               \n\t" // as round key 0
         "jmp 2f                            \n\t" // skip auxiliary routine

         /*
          * Finish generating the next round key.
          *
          * On entry xmm0 is r3:r2:r1:r0 and xmm1 is X:stuff:stuff:stuff
          * with X = rot( sub( r3 ) ) ^ RCON.
          *
          * On exit, xmm0 is r7:r6:r5:r4
          * with r4 = X + r0, r5 = r4 + r1, r6 = r5 + r2, r7 = r6 + r3
          * and those are written to the round key buffer.
          */
         "1:                                \n\t"
         "pshufd $0xff, %%xmm1, %%xmm1      \n\t" // X:X:X:X
         "pxor %%xmm0, %%xmm1               \n\t" // X+r3:X+r2:X+r1:r4
         "pslldq $4, %%xmm0                 \n\t" // r2:r1:r0:0
         "pxor %%xmm0, %%xmm1               \n\t" // X+r3+r2:X+r2+r1:r5:r4
         "pslldq $4, %%xmm0                 \n\t" // etc
         "pxor %%xmm0, %%xmm1               \n\t"
         "pslldq $4, %%xmm0                 \n\t"
         "pxor %%xmm1, %%xmm0               \n\t" // update xmm0 for next time!
         "add $16, %0                       \n\t" // point to next round key
         "movdqu %%xmm0, (%0)               \n\t" // write it
         "ret                               \n\t"

         /* Main "loop" */
         "2:                                \n\t"
         AESKEYGENA xmm0_xmm1 ",0x01        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x02        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x04        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x08        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x10        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x20        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x40        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x80        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x1B        \n\tcall 1b \n\t"
         AESKEYGENA xmm0_xmm1 ",0x36        \n\tcall 1b \n\t"
         :
         : "r" (rk), "r" (key)
         : "memory", "cc", "0" );
}

/*
 * Key expansion, 192-bit case
 */
static void aesni_setkey_enc_192( unsigned char *rk,
                                  const unsigned char *key )
{
    asm( "movdqu (%1), %%xmm0   \n\t" // copy original round key
         "movdqu %%xmm0, (%0)   \n\t"
         "add $16, %0           \n\t"
         "movq 16(%1), %%xmm1   \n\t"
         "movq %%xmm1, (%0)     \n\t"
         "add $8, %0            \n\t"
         "jmp 2f                \n\t" // skip auxiliary routine

         /*
          * Finish generating the next 6 quarter-keys.
          *
          * On entry xmm0 is r3:r2:r1:r0, xmm1 is stuff:stuff:r5:r4
          * and xmm2 is stuff:stuff:X:stuff with X = rot( sub( r3 ) ) ^ RCON.
          *
          * On exit, xmm0 is r9:r8:r7:r6 and xmm1 is stuff:stuff:r11:r10
          * and those are written to the round key buffer.
          */
         "1:                            \n\t"
         "pshufd $0x55, %%xmm2, %%xmm2  \n\t" // X:X:X:X
         "pxor %%xmm0, %%xmm2           \n\t" // X+r3:X+r2:X+r1:r4
         "pslldq $4, %%xmm0             \n\t" // etc
         "pxor %%xmm0, %%xmm2           \n\t"
         "pslldq $4, %%xmm0             \n\t"
         "pxor %%xmm0, %%xmm2           \n\t"
         "pslldq $4, %%xmm0             \n\t"
         "pxor %%xmm2, %%xmm0           \n\t" // update xmm0 = r9:r8:r7:r6
         "movdqu %%xmm0, (%0)           \n\t"
         "add $16, %0                   \n\t"
         "pshufd $0xff, %%xmm0, %%xmm2  \n\t" // r9:r9:r9:r9
         "pxor %%xmm1, %%xmm2           \n\t" // stuff:stuff:r9+r5:r10
         "pslldq $4, %%xmm1             \n\t" // r2:r1:r0:0
         "pxor %%xmm2, %%xmm1           \n\t" // xmm1 = stuff:stuff:r11:r10
         "movq %%xmm1, (%0)             \n\t"
         "add $8, %0                    \n\t"
         "ret                           \n\t"

         "2:                            \n\t"
         AESKEYGENA xmm1_xmm2 ",0x01    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x02    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x04    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x08    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x10    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x20    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x40    \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x80    \n\tcall 1b \n\t"

         :
         : "r" (rk), "r" (key)
         : "memory", "cc", "0" );
}

/*
 * Key expansion, 256-bit case
 */
static void aesni_setkey_enc_256( unsigned char *rk,
                                  const unsigned char *key )
{
    asm( "movdqu (%1), %%xmm0           \n\t"
         "movdqu %%xmm0, (%0)           \n\t"
         "add $16, %0                   \n\t"
         "movdqu 16(%1), %%xmm1         \n\t"
         "movdqu %%xmm1, (%0)           \n\t"
         "jmp 2f                        \n\t" // skip auxiliary routine

         /*
          * Finish generating the next two round keys.
          *
          * On entry xmm0 is r3:r2:r1:r0, xmm1 is r7:r6:r5:r4 and
          * xmm2 is X:stuff:stuff:stuff with X = rot( sub( r7 )) ^ RCON
          *
          * On exit, xmm0 is r11:r10:r9:r8 and xmm1 is r15:r14:r13:r12
          * and those have been written to the output buffer.
          */
         "1:                                \n\t"
         "pshufd $0xff, %%xmm2, %%xmm2      \n\t"
         "pxor %%xmm0, %%xmm2               \n\t"
         "pslldq $4, %%xmm0                 \n\t"
         "pxor %%xmm0, %%xmm2               \n\t"
         "pslldq $4, %%xmm0                 \n\t"
         "pxor %%xmm0, %%xmm2               \n\t"
         "pslldq $4, %%xmm0                 \n\t"
         "pxor %%xmm2, %%xmm0               \n\t"
         "add $16, %0                       \n\t"
         "movdqu %%xmm0, (%0)               \n\t"

         /* Set xmm2 to stuff:Y:stuff:stuff with Y = subword( r11 )
          * and proceed to generate next round key from there */
         AESKEYGENA xmm0_xmm2 ",0x00        \n\t"
         "pshufd $0xaa, %%xmm2, %%xmm2      \n\t"
         "pxor %%xmm1, %%xmm2               \n\t"
         "pslldq $4, %%xmm1                 \n\t"
         "pxor %%xmm1, %%xmm2               \n\t"
         "pslldq $4, %%xmm1                 \n\t"
         "pxor %%xmm1, %%xmm2               \n\t"
         "pslldq $4, %%xmm1                 \n\t"
         "pxor %%xmm2, %%xmm1               \n\t"
         "add $16, %0                       \n\t"
         "movdqu %%xmm1, (%0)               \n\t"
         "ret                               \n\t"

         /*
          * Main "loop" - Generating one more key than necessary,
          * see definition of mbedtls_aes_context.buf
          */
         "2:                                \n\t"
         AESKEYGENA xmm1_xmm2 ",0x01        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x02        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x04        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x08        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x10        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x20        \n\tcall 1b \n\t"
         AESKEYGENA xmm1_xmm2 ",0x40        \n\tcall 1b \n\t"
         :
         : "r" (rk), "r" (key)
         : "memory", "cc", "0" );
}

/*
 * Key expansion, wrapper
 */
int mbedtls_aesni_setkey_enc( unsigned char *rk,
                      const unsigned char *key,
                      size_t bits )
{
    switch( bits )
    {
        case 128: aesni_setkey_enc_128( rk, key ); break;
        case 192: aesni_setkey_enc_192( rk, key ); break;
        case 256: aesni_setkey_enc_256( rk, key ); break;
        default : return( MBEDTLS_ERR_AES_INVALID_KEY_LENGTH );
    }

    return( 0 );
}

#endif /* MBEDTLS_HAVE_X86_64 */

#endif /* MBEDTLS_AESNI_C */
