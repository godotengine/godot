// Copyright (c) 2012, Google Inc.
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//     * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//     * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//     * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_STAB_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_STAB_H

#include <sys/cdefs.h>

#ifdef __BIONIC_HAVE_STAB_H
#include <stab.h>
#else

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

#define _STAB_CODE_LIST       \
  _STAB_CODE_DEF(UNDF,0x00)   \
  _STAB_CODE_DEF(GSYM,0x20)   \
  _STAB_CODE_DEF(FNAME,0x22)  \
  _STAB_CODE_DEF(FUN,0x24)    \
  _STAB_CODE_DEF(STSYM,0x26)  \
  _STAB_CODE_DEF(LCSYM,0x28)  \
  _STAB_CODE_DEF(MAIN,0x2a)   \
  _STAB_CODE_DEF(PC,0x30)     \
  _STAB_CODE_DEF(NSYMS,0x32)  \
  _STAB_CODE_DEF(NOMAP,0x34)  \
  _STAB_CODE_DEF(OBJ,0x38)    \
  _STAB_CODE_DEF(OPT,0x3c)    \
  _STAB_CODE_DEF(RSYM,0x40)   \
  _STAB_CODE_DEF(M2C,0x42)    \
  _STAB_CODE_DEF(SLINE,0x44)  \
  _STAB_CODE_DEF(DSLINE,0x46) \
  _STAB_CODE_DEF(BSLINE,0x48) \
  _STAB_CODE_DEF(BROWS,0x48)  \
  _STAB_CODE_DEF(DEFD,0x4a)   \
  _STAB_CODE_DEF(EHDECL,0x50) \
  _STAB_CODE_DEF(MOD2,0x50)   \
  _STAB_CODE_DEF(CATCH,0x54)  \
  _STAB_CODE_DEF(SSYM,0x60)   \
  _STAB_CODE_DEF(SO,0x64)     \
  _STAB_CODE_DEF(LSYM,0x80)   \
  _STAB_CODE_DEF(BINCL,0x82)  \
  _STAB_CODE_DEF(SOL,0x84)    \
  _STAB_CODE_DEF(PSYM,0xa0)   \
  _STAB_CODE_DEF(EINCL,0xa2)  \
  _STAB_CODE_DEF(ENTRY,0xa4)  \
  _STAB_CODE_DEF(LBRAC,0xc0)  \
  _STAB_CODE_DEF(EXCL,0xc2)   \
  _STAB_CODE_DEF(SCOPE,0xc4)  \
  _STAB_CODE_DEF(RBRAC,0xe0)  \
  _STAB_CODE_DEF(BCOMM,0xe2)  \
  _STAB_CODE_DEF(ECOMM,0xe4)  \
  _STAB_CODE_DEF(ECOML,0xe8)  \
  _STAB_CODE_DEF(NBTEXT,0xf0) \
  _STAB_CODE_DEF(NBDATA,0xf2) \
  _STAB_CODE_DEF(NBBSS,0xf4)  \
  _STAB_CODE_DEF(NBSTS,0xf6)  \
  _STAB_CODE_DEF(NBLCS,0xf8)  \
  _STAB_CODE_DEF(LENG,0xfe)

enum __stab_debug_code {
#define _STAB_CODE_DEF(x,y)  N_##x = y,
_STAB_CODE_LIST
#undef _STAB_CODE_DEF
};

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // __BIONIC_HAVE_STAB_H

#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_STAB_H
