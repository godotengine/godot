/* Copyright (C) 2004 Jean-Marc Valin */
/**
   @file filters_arm4.h
   @brief Various analysis/synthesis filters (ARM4 version)
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

#define OVERRIDE_NORMALIZE16
int normalize16(const spx_sig_t *x, spx_word16_t *y, spx_sig_t max_scale, int len)
{
   spx_sig_t max_val=1;
   int sig_shift;
   int dead1, dead2, dead3, dead4, dead5, dead6;

   __asm__ __volatile__ (
         "\tmov %1, #1 \n"
         "\tmov %3, #0 \n"

         ".normalize16loop1%=: \n"

         "\tldr %4, [%0], #4 \n"
         "\tcmps %4, %1 \n"
         "\tmovgt %1, %4 \n"
         "\tcmps %4, %3 \n"
         "\tmovlt %3, %4 \n"

         "\tsubs %2, %2, #1 \n"
         "\tbne .normalize16loop1%=\n"

         "\trsb %3, %3, #0 \n"
         "\tcmp %1, %3 \n"
         "\tmovlt %1, %3 \n"
   : "=r" (dead1), "=r" (max_val), "=r" (dead3), "=r" (dead4),
   "=r" (dead5), "=r" (dead6)
   : "0" (x), "2" (len)
   : "cc");

   sig_shift=0;
   while (max_val>max_scale)
   {
      sig_shift++;
      max_val >>= 1;
   }
   
   __asm__ __volatile__ (
         ".normalize16loop%=: \n"

         "\tldr %4, [%0], #4 \n"
         "\tldr %5, [%0], #4 \n"
         "\tmov %4, %4, asr %3 \n"
         "\tstrh %4, [%1], #2 \n"
         "\tldr %4, [%0], #4 \n"
         "\tmov %5, %5, asr %3 \n"
         "\tstrh %5, [%1], #2 \n"
         "\tldr %5, [%0], #4 \n"
         "\tmov %4, %4, asr %3 \n"
         "\tstrh %4, [%1], #2 \n"
         "\tsubs %2, %2, #1 \n"
         "\tmov %5, %5, asr %3 \n"
         "\tstrh %5, [%1], #2 \n"

         "\tbgt .normalize16loop%=\n"
   : "=r" (dead1), "=r" (dead2), "=r" (dead3), "=r" (dead4),
   "=r" (dead5), "=r" (dead6)
   : "0" (x), "1" (y), "2" (len>>2), "3" (sig_shift)
   : "cc", "memory");
   return sig_shift;
}

