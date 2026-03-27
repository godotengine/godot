/***********************************************************************
Copyright (c) 2006-2011, Skype Limited. All rights reserved.
Copyright (c) 2013       Parrot
Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:
- Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.
- Neither the name of Internet Society, IETF or IETF Trust, nor the
names of specific contributors, may be used to endorse or promote
products derived from this software without specific prior written
permission.
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
***********************************************************************/

#ifndef SILK_SIGPROC_FIX_ARMv5E_H
#define SILK_SIGPROC_FIX_ARMv5E_H

#undef silk_SMULTT
static OPUS_INLINE opus_int32 silk_SMULTT_armv5e(opus_int32 a, opus_int32 b)
{
  opus_int32 res;
  __asm__(
      "#silk_SMULTT\n\t"
      "smultt %0, %1, %2\n\t"
      : "=r"(res)
      : "%r"(a), "r"(b)
  );
  return res;
}
#define silk_SMULTT(a, b) (silk_SMULTT_armv5e(a, b))

#undef silk_SMLATT
static OPUS_INLINE opus_int32 silk_SMLATT_armv5e(opus_int32 a, opus_int32 b,
 opus_int32 c)
{
  opus_int32 res;
  __asm__(
      "#silk_SMLATT\n\t"
      "smlatt %0, %1, %2, %3\n\t"
      : "=r"(res)
      : "%r"(b), "r"(c), "r"(a)
  );
  return res;
}
#define silk_SMLATT(a, b, c) (silk_SMLATT_armv5e(a, b, c))

#endif
