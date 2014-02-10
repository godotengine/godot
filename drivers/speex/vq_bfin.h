/* Copyright (C) 2005 Analog Devices */
/**
   @file vq_bfin.h
   @author Jean-Marc Valin 
   @brief Blackfin-optimized vq routine
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

#define OVERRIDE_VQ_NBEST
void vq_nbest(spx_word16_t *in, const spx_word16_t *codebook, int len, int entries, spx_word32_t *E, int N, int *nbest, spx_word32_t *best_dist, char *stack)
{
   if (N==1)
   {
      best_dist[0] = 2147483647;
      {
         spx_word32_t dist;
         __asm__ __volatile__
               (
            "LC0 = %8;\n\t"
            "R2 = 0;\n\t"
            "I0 = %6;\n\t"
            "B0 = %6;\n\t"
            "L0 = %9;\n\t"
            "LOOP entries_loop%= LC0;\n\t"
            "LOOP_BEGIN entries_loop%=;\n\t"
               "%0 = [%4++];\n\t"
               "%0 >>= 1;\n\t"
               "A0 = %0;\n\t"
               "R0.L = W[%1++%7] || R1.L = W[I0++];\n\t"
               "LOOP vq_loop%= LC1 = %5;\n\t"
               "LOOP_BEGIN vq_loop%=;\n\t"
                  "%0 = (A0 -= R0.L*R1.L) (IS) || R0.L = W[%1++%7] || R1.L = W[I0++];\n\t"
               "LOOP_END vq_loop%=;\n\t"
               "%0 = (A0 -= R0.L*R1.L) (IS);\n\t"
               "cc = %0 < %2;\n\t"
               "if cc %2 = %0;\n\t"
               "if cc %3 = R2;\n\t"
               "R2 += 1;\n\t"
            "LOOP_END entries_loop%=;\n\t"
            : "=&D" (dist), "=&a" (codebook), "=&d" (best_dist[0]), "=&d" (nbest[0]), "=&a" (E)
            : "a" (len-1), "a" (in), "a" (2), "d" (entries), "d" (len<<1), "1" (codebook), "4" (E), "2" (best_dist[0]), "3" (nbest[0])
            : "R0", "R1", "R2", "I0", "L0", "B0", "A0", "cc", "memory"
               );
      }
   } else {
   int i,k,used;
   used = 0;
   for (i=0;i<entries;i++)
   {
      spx_word32_t dist;
      __asm__
            (
            "%0 >>= 1;\n\t"
            "A0 = %0;\n\t"
            "I0 = %3;\n\t"
            "L0 = 0;\n\t"
            "R0.L = W[%1++%4] || R1.L = W[I0++];\n\t"
            "LOOP vq_loop%= LC0 = %2;\n\t"
            "LOOP_BEGIN vq_loop%=;\n\t"
               "%0 = (A0 -= R0.L*R1.L) (IS) || R0.L = W[%1++%4] || R1.L = W[I0++];\n\t"
            "LOOP_END vq_loop%=;\n\t"
            "%0 = (A0 -= R0.L*R1.L) (IS);\n\t"
         : "=D" (dist), "=a" (codebook)
         : "a" (len-1), "a" (in), "a" (2), "1" (codebook), "0" (E[i])
         : "R0", "R1", "I0", "L0", "A0"
            );
      if (i<N || dist<best_dist[N-1])
      {
         for (k=N-1; (k >= 1) && (k > used || dist < best_dist[k-1]); k--)
         {
            best_dist[k]=best_dist[k-1];
            nbest[k] = nbest[k-1];
         }
         best_dist[k]=dist;
         nbest[k]=i;
         used++;
      }
   }
   }
}
