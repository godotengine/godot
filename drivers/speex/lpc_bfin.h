/* Copyright (C) 2005 Analog Devices */
/**
   @file lpc_bfin.h
   @author Jean-Marc Valin 
   @brief Functions for LPC (Linear Prediction Coefficients) analysis (Blackfin version)
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

#define OVERRIDE_SPEEX_AUTOCORR
void _spx_autocorr(
const spx_word16_t *x,   /*  in: [0...n-1] samples x   */
spx_word16_t       *ac,  /* out: [0...lag-1] ac values */
int          lag, 
int          n
                  )
{
   spx_word32_t d;
   const spx_word16_t *xs;
   int i, j;
   spx_word32_t ac0=1;
   spx_word32_t ac32[11], *ac32top;
   int shift, ac_shift;
   ac32top = ac32+lag-1;
   int lag_1, N_lag;
   int nshift;
   lag_1 = lag-1;
   N_lag = n-lag_1;
   for (j=0;j<n;j++)
      ac0 = ADD32(ac0,SHR32(MULT16_16(x[j],x[j]),8));
   ac0 = ADD32(ac0,n);
   shift = 8;
   while (shift && ac0<0x40000000)
   {
      shift--;
      ac0 <<= 1;
   }
   ac_shift = 18;
   while (ac_shift && ac0<0x40000000)
   {
      ac_shift--;
      ac0 <<= 1;
   }
   
   xs = x+lag-1;
   nshift = -shift;
   __asm__ __volatile__ 
   (
         "P2 = %0;\n\t"
         "I0 = P2;\n\t" /* x in I0 */
         "B0 = P2;\n\t" /* x in B0 */
         "R0 = %3;\n\t" /* len in R0 */
         "P3 = %3;\n\t" /* len in R0 */
         "P4 = %4;\n\t" /* nb_pitch in R0 */
         "R1 = R0 << 1;\n\t" /* number of bytes in x */
         "L0 = R1;\n\t"
         "P0 = %1;\n\t"
         "P1 = %2;\n\t"
         "B1 = P1;\n\t"
         "R4 = %5;\n\t"
         "L1 = 0;\n\t" /*Disable looping on I1*/

         "r0 = [I0++];\n\t"
         "R2 = 0;R3=0;"
         "LOOP pitch%= LC0 = P4 >> 1;\n\t"
         "LOOP_BEGIN pitch%=;\n\t"
            "I1 = P0;\n\t"
            "A1 = A0 = 0;\n\t"
            "R1 = [I1++];\n\t"
            "LOOP inner_prod%= LC1 = P3 >> 1;\n\t"
            "LOOP_BEGIN inner_prod%=;\n\t"
               "A1 += R0.L*R1.H, A0 += R0.L*R1.L (IS) || R1.L = W[I1++];\n\t"
               "A1 += R0.H*R1.L, A0 += R0.H*R1.H (IS) || R1.H = W[I1++] || R0 = [I0++];\n\t"
            "LOOP_END inner_prod%=;\n\t"
            "A0 = ASHIFT A0 by R4.L;\n\t"
            "A1 = ASHIFT A1 by R4.L;\n\t"
   
            "R2 = A0, R3 = A1;\n\t"
            "[P1--] = R2;\n\t"
            "[P1--] = R3;\n\t"
            "P0 += 4;\n\t"
         "LOOP_END pitch%=;\n\t"
   : : "m" (xs), "m" (x), "m" (ac32top), "m" (N_lag), "m" (lag_1), "m" (nshift)
   : "A0", "A1", "P0", "P1", "P2", "P3", "P4", "R0", "R1", "R2", "R3", "R4", "I0", "I1", "L0", "L1", "B0", "B1", "memory"
   );
   d=0;
   for (j=0;j<n;j++)
   {
      d = ADD32(d,SHR32(MULT16_16(x[j],x[j]), shift));
   }
   ac32[0] = d;
   
   for (i=0;i<lag;i++)
   {
      d=0;
      for (j=i;j<lag_1;j++)
      {
         d = ADD32(d,SHR32(MULT16_16(x[j],x[j-i]), shift));
      }
      if (i)
         ac32[i] += d;
      ac[i] = SHR32(ac32[i], ac_shift);
   }
}

