/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file ltp_arm4.h
   @brief Long-Term Prediction functions (ARM4 version)
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

#define OVERRIDE_INNER_PROD
spx_word32_t inner_prod(const spx_word16_t *x, const spx_word16_t *y, int len)
{
   spx_word32_t sum1=0,sum2=0;
   spx_word16_t *deadx, *deady;
   int deadlen, dead1, dead2, dead3, dead4, dead5, dead6;
   __asm__ __volatile__ (
         "\tldrsh %5, [%0], #2 \n"
         "\tldrsh %6, [%1], #2 \n"
         ".inner_prod_loop%=:\n"
         "\tsub %7, %7, %7\n"
         "\tsub %10, %10, %10\n"

         "\tldrsh %8, [%0], #2 \n"
         "\tldrsh %9, [%1], #2 \n"
         "\tmla %7, %5, %6, %7\n"
         "\tldrsh %5, [%0], #2 \n"
         "\tldrsh %6, [%1], #2 \n"
         "\tmla %10, %8, %9, %10\n"
         "\tldrsh %8, [%0], #2 \n"
         "\tldrsh %9, [%1], #2 \n"
         "\tmla %7, %5, %6, %7\n"
         "\tldrsh %5, [%0], #2 \n"
         "\tldrsh %6, [%1], #2 \n"
         "\tmla %10, %8, %9, %10\n"

         "\tldrsh %8, [%0], #2 \n"
         "\tldrsh %9, [%1], #2 \n"
         "\tmla %7, %5, %6, %7\n"
         "\tldrsh %5, [%0], #2 \n"
         "\tldrsh %6, [%1], #2 \n"
         "\tmla %10, %8, %9, %10\n"
         "\tldrsh %8, [%0], #2 \n"
         "\tldrsh %9, [%1], #2 \n"
         "\tmla %7, %5, %6, %7\n"
         "\tldrsh %5, [%0], #2 \n"
         "\tldrsh %6, [%1], #2 \n"
         "\tmla %10, %8, %9, %10\n"

         "\tsubs %4, %4, #1\n"
         "\tadd %2, %2, %7, asr #5\n"
         "\tadd %3, %3, %10, asr #5\n"
         "\tbne .inner_prod_loop%=\n"
   : "=r" (deadx), "=r" (deady), "+r" (sum1),  "+r" (sum2),
     "=r" (deadlen), "=r" (dead1), "=r" (dead2), "=r" (dead3),
     "=r" (dead4), "=r" (dead5), "=r" (dead6)
   : "0" (x), "1" (y), "4" (len>>3)
   : "cc"
                        );
   return (sum1+sum2)>>1;
}

#define OVERRIDE_PITCH_XCORR
void pitch_xcorr(const spx_word16_t *_x, const spx_word16_t *_y, spx_word32_t *corr, int len, int nb_pitch, char *stack)
{
   int i,j;
   for (i=0;i<nb_pitch;i+=4)
   {
      /* Compute correlation*/
      //corr[nb_pitch-1-i]=inner_prod(x, _y+i, len);
      spx_word32_t sum1=0;
      spx_word32_t sum2=0;
      spx_word32_t sum3=0;
      spx_word32_t sum4=0;
      const spx_word16_t *y = _y+i;
      const spx_word16_t *x = _x;
      spx_word32_t y0, y1, y2, y3;
      y0=*y++;
      y1=*y++;
      y2=*y++;
      y3=*y++;
      for (j=0;j<len;j+=4)
      {
         spx_word32_t part1, part2, part3, part4, x0;
         spx_word32_t dead1;
         __asm__ __volatile__ (
#ifdef SHORTCUTS
               "\tldrsh %10, [%8], #4 \n"
               "\tmul %4, %10, %0 \n"
               "\tldrsh %15, [%8], #4 \n"
               "\tmul %5, %10, %1 \n"
               "\tldrsh %0, [%9], #2 \n"
               "\tmul %6, %10, %2 \n"
               "\tldrsh %1, [%9], #2 \n"
               "\tmul %7, %10, %3 \n"
               
               
               "\tmla %4, %15, %2, %4 \n"
               "\tldrsh %2, [%9], #2 \n"
               "\tmla %5, %15, %3, %5 \n"
               "\tldrsh %3, [%9], #2 \n"
               "\tmla %6, %15, %0, %6 \n"
               "\tmla %7, %15, %1, %7 \n"

#else
               "\tldrsh %10, [%8], #2 \n"
               "\tmul %4, %10, %0 \n"
               "\tmul %5, %10, %1 \n"
               "\tmul %6, %10, %2 \n"
               "\tmul %7, %10, %3 \n"

               "\tldrsh %10, [%8], #2 \n"
               "\tldrsh %0, [%9], #2 \n"
               "\tmla %4, %10, %1, %4 \n"
               "\tmla %5, %10, %2, %5 \n"
               "\tmla %6, %10, %3, %6 \n"
               "\tmla %7, %10, %0, %7 \n"

               "\tldrsh %10, [%8], #2 \n"
               "\tldrsh %1, [%9], #2 \n"
               "\tmla %4, %10, %2, %4 \n"
               "\tmla %5, %10, %3, %5 \n"
               "\tmla %6, %10, %0, %6 \n"
               "\tmla %7, %10, %1, %7 \n"

               "\tldrsh %10, [%8], #2 \n"
               "\tldrsh %2, [%9], #2 \n"
               "\tmla %4, %10, %3, %4 \n"
               "\tmla %5, %10, %0, %5 \n"
               "\tmla %6, %10, %1, %6 \n"
               "\tmla %7, %10, %2, %7 \n"

               "\tldrsh %3, [%9], #2 \n"
#endif

               "\tldr %10, %11 \n"
               "\tldr %15, %12 \n"
               "\tadd %4, %10, %4, asr #6 \n"
               "\tstr %4, %11 \n"
               "\tldr %10, %13 \n"
               "\tadd %5, %15, %5, asr #6 \n"
               "\tstr %5, %12 \n"
               "\tldr %15, %14 \n"
               "\tadd %6, %10, %6, asr #6 \n"
               "\tadd %7, %15, %7, asr #6 \n"
               "\tstr %6, %13 \n"
               "\tstr %7, %14 \n"

            : "+r" (y0), "+r" (y1), "+r" (y2), "+r" (y3),
         "=r" (part1),  "=r" (part2),  "=r" (part3),  "=r" (part4),
              "+r" (x), "+r" (y), "=r" (x0), "+m" (sum1),
              "+m" (sum2), "+m" (sum3), "+m" (sum4), "=r" (dead1)
            :
            : "cc", "memory"
                              );
      }
      corr[nb_pitch-1-i]=sum1;
      corr[nb_pitch-2-i]=sum2;
      corr[nb_pitch-3-i]=sum3;
      corr[nb_pitch-4-i]=sum4;
   }

}
