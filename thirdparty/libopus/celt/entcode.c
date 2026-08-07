/* Copyright (c) 2001-2011 Timothy B. Terriberry
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

   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
   ``AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
   LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
   A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER
   OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
   EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
   PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
   PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
   LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
   NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
   SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

#ifdef HAVE_CONFIG_H
#include "config.h"
#endif

#include "entcode.h"
#include "arch.h"

#if !defined(EC_CLZ)
/*This is a fallback for systems where we don't know how to access
   a BSR or CLZ instruction (see ecintrin.h).
  If you are optimizing Opus on a new platform and it has a native CLZ or
   BZR (e.g. cell, MIPS, x86, etc) then making it available to Opus will be
   an easy performance win.*/
int ec_ilog(opus_uint32 _v){
  /*On a Pentium M, this branchless version tested as the fastest on
     1,000,000,000 random 32-bit integers, edging out a similar version with
     branches, and a 256-entry LUT version.*/
  int ret;
  int m;
  ret=!!_v;
  m=!!(_v&0xFFFF0000)<<4;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xFF00)<<3;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xF0)<<2;
  _v>>=m;
  ret|=m;
  m=!!(_v&0xC)<<1;
  _v>>=m;
  ret|=m;
  ret+=!!(_v&0x2);
  return ret;
}
#endif

#if 1
/* This is a faster version of ec_tell_frac() that takes advantage
   of the low (1/8 bit) resolution to use just a linear function
   followed by a lookup to determine the exact transition thresholds. */
opus_uint32 ec_tell_frac(ec_ctx *_this){
  static const unsigned correction[8] =
    {35733, 38967, 42495, 46340,
     50535, 55109, 60097, 65535};
  opus_uint32 nbits;
  opus_uint32 r;
  int         l;
  unsigned    b;
  nbits=_this->nbits_total<<BITRES;
  l=EC_ILOG(_this->rng);
  r=_this->rng>>(l-16);
  b = (r>>12)-8;
  b += r>correction[b];
  l = (l<<3)+b;
  return nbits-l;
}
#else
opus_uint32 ec_tell_frac(ec_ctx *_this){
  opus_uint32 nbits;
  opus_uint32 r;
  int         l;
  int         i;
  /*To handle the non-integral number of bits still left in the encoder/decoder
     state, we compute the worst-case number of bits of val that must be
     encoded to ensure that the value is inside the range for any possible
     subsequent bits.
    The computation here is independent of val itself (the decoder does not
     even track that value), even though the real number of bits used after
     ec_enc_done() may be 1 smaller if rng is a power of two and the
     corresponding trailing bits of val are all zeros.
    If we did try to track that special case, then coding a value with a
     probability of 1/(1<<n) might sometimes appear to use more than n bits.
    This may help explain the surprising result that a newly initialized
     encoder or decoder claims to have used 1 bit.*/
  nbits=_this->nbits_total<<BITRES;
  l=EC_ILOG(_this->rng);
  r=_this->rng>>(l-16);
  for(i=BITRES;i-->0;){
    int b;
    r=r*r>>15;
    b=(int)(r>>16);
    l=l<<1|b;
    r>>=b;
  }
  return nbits-l;
}
#endif

#ifdef USE_SMALL_DIV_TABLE
/* Result of 2^32/(2*i+1), except for i=0. */
const opus_uint32 SMALL_DIV_TABLE[129] = {
   0xFFFFFFFF, 0x55555555, 0x33333333, 0x24924924,
   0x1C71C71C, 0x1745D174, 0x13B13B13, 0x11111111,
   0x0F0F0F0F, 0x0D79435E, 0x0C30C30C, 0x0B21642C,
   0x0A3D70A3, 0x097B425E, 0x08D3DCB0, 0x08421084,
   0x07C1F07C, 0x07507507, 0x06EB3E45, 0x06906906,
   0x063E7063, 0x05F417D0, 0x05B05B05, 0x0572620A,
   0x05397829, 0x05050505, 0x04D4873E, 0x04A7904A,
   0x047DC11F, 0x0456C797, 0x04325C53, 0x04104104,
   0x03F03F03, 0x03D22635, 0x03B5CC0E, 0x039B0AD1,
   0x0381C0E0, 0x0369D036, 0x03531DEC, 0x033D91D2,
   0x0329161F, 0x03159721, 0x03030303, 0x02F14990,
   0x02E05C0B, 0x02D02D02, 0x02C0B02C, 0x02B1DA46,
   0x02A3A0FD, 0x0295FAD4, 0x0288DF0C, 0x027C4597,
   0x02702702, 0x02647C69, 0x02593F69, 0x024E6A17,
   0x0243F6F0, 0x0239E0D5, 0x02302302, 0x0226B902,
   0x021D9EAD, 0x0214D021, 0x020C49BA, 0x02040810,
   0x01FC07F0, 0x01F44659, 0x01ECC07B, 0x01E573AC,
   0x01DE5D6E, 0x01D77B65, 0x01D0CB58, 0x01CA4B30,
   0x01C3F8F0, 0x01BDD2B8, 0x01B7D6C3, 0x01B20364,
   0x01AC5701, 0x01A6D01A, 0x01A16D3F, 0x019C2D14,
   0x01970E4F, 0x01920FB4, 0x018D3018, 0x01886E5F,
   0x0183C977, 0x017F405F, 0x017AD220, 0x01767DCE,
   0x01724287, 0x016E1F76, 0x016A13CD, 0x01661EC6,
   0x01623FA7, 0x015E75BB, 0x015AC056, 0x01571ED3,
   0x01539094, 0x01501501, 0x014CAB88, 0x0149539E,
   0x01460CBC, 0x0142D662, 0x013FB013, 0x013C995A,
   0x013991C2, 0x013698DF, 0x0133AE45, 0x0130D190,
   0x012E025C, 0x012B404A, 0x01288B01, 0x0125E227,
   0x01234567, 0x0120B470, 0x011E2EF3, 0x011BB4A4,
   0x01194538, 0x0116E068, 0x011485F0, 0x0112358E,
   0x010FEF01, 0x010DB20A, 0x010B7E6E, 0x010953F3,
   0x01073260, 0x0105197F, 0x0103091B, 0x01010101
};
#endif
