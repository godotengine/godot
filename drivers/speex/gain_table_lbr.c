/* Copyright (C) 2002 Jean-Marc Valin 
   File: gain_table_lbr.c
   Codebook for 3-tap pitch prediction gain (32 entries)
  
   Redistribution and use in source and binary forms, with or without
   modification, are permitted provided that the following conditions are
   met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.  

   2. Redistributions in binary form must reproduce the above copyright
   notice, this list of conditions and the following disclaimer in the
   documentation and/or other materials provided with the distribution.

   3. The name of the author may not be used to endorse or promote products
   derived from this software without specific prior written permission.

   THIS SOFTWARE IS PROVIDED BY THE AUTHOR ``AS IS'' AND ANY EXPRESS OR
   IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
   OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
   DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY DIRECT,
   INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
   (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
   HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
   STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
   ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
   POSSIBILITY OF SUCH DAMAGE.
*/

const signed char gain_cdbk_lbr[128] = {
-32, -32, -32, 0,
-31, -58, -16, 22,
-41, -24, -43, 14,
-56, -22, -55, 29,
-13, 33, -41, 47,
-4, -39, -9, 29,
-41, 15, -12, 38,
-8, -15, -12, 31,
1, 2, -44, 40,
-22, -66, -42, 27,
-38, 28, -23, 38,
-21, 14, -37, 31,
0, 21, -50, 52,
-53, -71, -27, 33,
-37, -1, -19, 25,
-19, -5, -28, 22,
6, 65, -44, 74,
-33, -48, -33, 9,
-40, 57, -14, 58,
-17, 4, -45, 32,
-31, 38, -33, 36,
-23, 28, -40, 39,
-43, 29, -12, 46,
-34, 13, -23, 28,
-16, 15, -27, 34,
-14, -82, -15, 43,
-31, 25, -32, 29,
-21, 5, -5, 38,
-47, -63, -51, 33,
-46, 12, 3, 47,
-28, -17, -29, 11,
-10, 14, -40, 38};
