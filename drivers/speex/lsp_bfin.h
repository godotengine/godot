/* Copyright (C) 2006 David Rowe */
/**
   @file lsp_bfin.h
   @author David Rowe
   @brief LSP routines optimised for the Blackfin
*/
/*
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions
   are met:
   
   - Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.
   
   - Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.
   
   - Neither the name of the Xiph.org Foundation nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.
   
   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE FOUNDATION OR
   CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#define OVERRIDE_CHEB_POLY_EVA
#ifdef OVERRIDE_CHEB_POLY_EVA
static inline spx_word32_t cheb_poly_eva(
  spx_word16_t *coef, /* P or Q coefs in Q13 format               */
  spx_word16_t     x, /* cos of freq (-1.0 to 1.0) in Q14 format  */
  int              m, /* LPC order/2                              */
  char         *stack
)
{
    spx_word32_t sum;

   __asm__ __volatile__
     (
      "P0 = %2;\n\t"           /* P0: coef[m], coef[m-1],..., coef[0] */
      "R4 = 8192;\n\t"         /* R4: rounding constant               */
      "R2 = %1;\n\t"           /* R2: x  */

      "R5 = -16383;\n\t"
      "R2 = MAX(R2,R5);\n\t"
      "R5 = 16383;\n\t"
      "R2 = MIN(R2,R5);\n\t"

      "R3 = W[P0--] (X);\n\t"  /* R3: sum */
      "R5 = W[P0--] (X);\n\t"
      "R5 = R5.L * R2.L (IS);\n\t"
      "R5 = R5 + R4;\n\t"
      "R5 >>>= 14;\n\t"
      "R3 = R3 + R5;\n\t" 
      
      "R0 = R2;\n\t"           /* R0: b0 */
      "R1 = 16384;\n\t"        /* R1: b1 */
      "LOOP cpe%= LC0 = %3;\n\t"
      "LOOP_BEGIN cpe%=;\n\t"
        "P1 = R0;\n\t" 
        "R0 = R2.L * R0.L (IS) || R5 = W[P0--] (X);\n\t"
        "R0 >>>= 13;\n\t"
        "R0 = R0 - R1;\n\t"
        "R1 = P1;\n\t"
        "R5 = R5.L * R0.L (IS);\n\t"
        "R5 = R5 + R4;\n\t"
        "R5 >>>= 14;\n\t"
        "R3 = R3 + R5;\n\t"
      "LOOP_END cpe%=;\n\t"
      "%0 = R3;\n\t"
      : "=&d" (sum)
      : "a" (x), "a" (&coef[m]), "a" (m-1)
      : "R0", "R1", "R3", "R2", "R4", "R5", "P0", "P1"
      );
    return sum;
}
#endif



