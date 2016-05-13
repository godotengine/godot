/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file vq_arm4.h
   @brief ARM4-optimized vq routine
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
   int i,j;
   for (i=0;i<entries;i+=4)
   {
#if 1
      spx_word32_t dist1, dist2, dist3, dist4;
      int dead1, dead2, dead3, dead4, dead5, dead6, dead7, dead8;
      __asm__ __volatile__ (
            "mov %0, #0 \n\t"
            "mov %1, #0 \n\t"
            "mov %2, #0 \n\t"
            "mov %3, #0 \n\t"
            "mov %10, %4 \n\t"
            "add %4, %4, %4\n\t"
            ".vqloop%=:\n\t"
            "ldrsh %7, [%5], #2 \n\t"
            "ldrsh %8, [%6] \n\t"
            "mov %9, %6 \n\t"
            "mla %0, %7, %8, %0 \n\t"
            "ldrsh %8, [%9, %4]! \n\t"
            "mla %1, %7, %8, %1 \n\t"
            "ldrsh %8, [%9, %4]!\n\t"
            "mla %2, %7, %8, %2 \n\t"
            "ldrsh %8, [%9, %4]! \n\t"
            "mla %3, %7, %8, %3 \n\t"
            "subs %10, %10, #1 \n\t"
            "add %6, %6, #2 \n\t"
            "bne .vqloop%="
         : "=r" (dist1), "=r" (dist2), "=r" (dist3), "=r" (dist4),
      "=r" (dead1), "=r" (dead2), "=r" (codebook), "=r" (dead4),
      "=r" (dead5), "=r" (dead6), "=r" (dead7)
         : "4" (len), "5" (in), "6" (codebook)
         : "cc");
#else
dist1=dist2=dist3=dist4=0;
   /*   spx_word32_t dist1=0;
      spx_word32_t dist2=0;
      spx_word32_t dist3=0;
      spx_word32_t dist4=0;*/
      for (j=0;j<2;j++)
      {
         const spx_word16_t *code = codebook;
         dist1 = MAC16_16(dist1,in[j],*code);
         code += len;
         dist2 = MAC16_16(dist2,in[j],*code);
         code += len;
         dist3 = MAC16_16(dist3,in[j],*code);
         code += len;
         dist4 = MAC16_16(dist4,in[j],*code);
         codebook++;
      }
#endif
      dist1=SUB32(SHR(*E++,1),dist1);
      if (dist1<*best_dist || i==0)
      {
         *best_dist=dist1;
         *nbest=i;
      }
      dist2=SUB32(SHR(*E++,1),dist2);
      if (dist2<*best_dist)
      {
         *best_dist=dist2;
         *nbest=i+1;
      }
      dist3=SUB32(SHR(*E++,1),dist3);
      if (dist3<*best_dist)
      {
         *best_dist=dist3;
         *nbest=i+2;
      }
      dist4=SUB32(SHR(*E++,1),dist4);
      if (dist4<*best_dist)
      {
         *best_dist=dist4;
         *nbest=i+3;
      }
      codebook += 3*len;
   }
}
