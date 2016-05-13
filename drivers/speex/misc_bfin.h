/* Copyright (C) 2005 Analog Devices */
/**
   @file misc_bfin.h
   @author Jean-Marc Valin 
   @brief Various compatibility routines for Speex (Blackfin version)
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

#define OVERRIDE_SPEEX_MOVE
void *speex_move (void *dest, void *src, int n)
{
   __asm__ __volatile__
         (
         "L0 = 0;\n\t"
         "I0 = %0;\n\t"
         "R0 = [I0++];\n\t"
         "LOOP move%= LC0 = %2;\n\t"
         "LOOP_BEGIN move%=;\n\t"
            "[%1++] = R0 || R0 = [I0++];\n\t"
         "LOOP_END move%=;\n\t"
         "[%1++] = R0;\n\t"
   : "=a" (src), "=a" (dest)
   : "a" ((n>>2)-1), "0" (src), "1" (dest)
   : "R0", "I0", "L0", "memory"
         );
   return dest;
}
