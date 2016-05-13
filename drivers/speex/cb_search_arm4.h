/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file cb_search_arm4.h
   @brief Fixed codebook functions (ARM4 version)
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

/* This optimization is temporaly disabled until it is fixed to account for the fact 
   that "r" is now a 16-bit array */
#if 0
#define OVERRIDE_COMPUTE_WEIGHTED_CODEBOOK
static void compute_weighted_codebook(const signed char *shape_cb, const spx_word16_t *r, spx_word16_t *resp, spx_word16_t *resp2, spx_word32_t *E, int shape_cb_size, int subvect_size, char *stack)
{
   int i, j, k;
  //const signed char *shape;
   for (i=0;i<shape_cb_size;i+=4)
   {

      //shape = shape_cb;
      E[0]=0;
      E[1]=0;
      E[2]=0;
      E[3]=0;

      /* Compute codeword response using convolution with impulse response */
      for(j=0;j<subvect_size;j++)
      {
#if 1
          spx_word16_t *res;
          res = resp+j;
          spx_word32_t resj0,resj1,resj2,resj3;
          spx_word32_t dead1, dead2, dead3, dead4, dead5, dead6, dead7, dead8;
          __asm__ __volatile__ (
                "mov %0, #0 \n\t"
                "mov %1, #0 \n\t"
                "mov %2, #0 \n\t"
                "mov %3, #0 \n\t"
                ".weighted%=: \n\t"
                "ldrsb %8, [%6] \n\t"
                "ldr %10, [%5], #-4 \n\t"
                "mov %9, %6 \n\t"
                "ldrsb %11, [%9, %7]! \n\t"
                "mla %0, %10, %8, %0 \n\t"
                "ldrsb %8, [%9, %7]! \n\t"
                "mla %1, %10, %11, %1 \n\t"
                "ldrsb %11, [%9, %7]! \n\t"
                "mla %2, %10, %8, %2 \n\t"
                "subs %4, %4, #1 \n\t"
                "mla %3, %10, %11, %3 \n\t"
                "add %6, %6, #1 \n\t"
                "bne .weighted%= \n\t"
            : "=r" (resj0), "=r" (resj1), "=r" (resj2), "=r" (resj3),
          "=r" (dead1), "=r" (dead2), "=r" (dead3), "=r" (dead4),
          "=r" (dead5), "=r" (dead6), "=r" (dead7), "=r" (dead8)
            : "4" (j+1), "5" (r+j), "6" (shape_cb), "7" (subvect_size)
            : "cc", "memory");
#else
          spx_word16_t *res;
          res = resp+j;
          spx_word32_t resj0=0;
          spx_word32_t resj1=0;
          spx_word32_t resj2=0;
          spx_word32_t resj3=0;
          for (k=0;k<=j;k++)
          {
             const signed char *shape=shape_cb+k;
             resj0 = MAC16_16(resj0,*shape,r[j-k]);
             shape += subvect_size;
             resj1 = MAC16_16(resj1,*shape,r[j-k]);
             shape += subvect_size;
             resj2 = MAC16_16(resj2,*shape,r[j-k]);
             shape += subvect_size;
             resj3 = MAC16_16(resj3,*shape,r[j-k]);
             shape += subvect_size;
          }
#endif

#ifdef FIXED_POINT
          resj0 = SHR(resj0, 11);
          resj1 = SHR(resj1, 11);
          resj2 = SHR(resj2, 11);
          resj3 = SHR(resj3, 11);
#else
          resj0 *= 0.03125;
          resj1 *= 0.03125;
          resj2 *= 0.03125;
          resj3 *= 0.03125;
#endif

          /* Compute codeword energy */
          E[0]=ADD32(E[0],MULT16_16(resj0,resj0));
          E[1]=ADD32(E[1],MULT16_16(resj1,resj1));
          E[2]=ADD32(E[2],MULT16_16(resj2,resj2));
          E[3]=ADD32(E[3],MULT16_16(resj3,resj3));
          *res = resj0;
          res += subvect_size;
          *res = resj1;
          res += subvect_size;
          *res = resj2;
          res += subvect_size;
          *res = resj3;
          res += subvect_size;
      }
      resp += subvect_size<<2;
      shape_cb += subvect_size<<2;
      E+=4;
   }

}
#endif
