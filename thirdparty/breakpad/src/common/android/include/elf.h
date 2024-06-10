// Copyright 2012 Google LLC
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
//     * Neither the name of Google LLC nor the names of its
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

#ifndef GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ELF_H
#define GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ELF_H

#include <stdint.h>
#include <libgen.h>

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// The Android <elf.h> provides BSD-based definitions for the ElfXX_Nhdr
// types
// always source-compatible with the GLibc/kernel ones. To overcome this
// issue without modifying a lot of code in Breakpad, use an ugly macro
// renaming trick with #include_next

// Avoid conflict with BSD-based definition of ElfXX_Nhdr.
// Unfortunately, their field member names do not use a 'n_' prefix.
#define Elf32_Nhdr   __bsd_Elf32_Nhdr
#define Elf64_Nhdr   __bsd_Elf64_Nhdr

// In case they are defined by the NDK version
#define Elf32_auxv_t  __bionic_Elf32_auxv_t
#define Elf64_auxv_t  __bionic_Elf64_auxv_t

#define Elf32_Dyn     __bionic_Elf32_Dyn
#define Elf64_Dyn     __bionic_Elf64_Dyn

#include_next <elf.h>

#undef Elf32_Nhdr
#undef Elf64_Nhdr

typedef struct {
  Elf32_Word n_namesz;
  Elf32_Word n_descsz;
  Elf32_Word n_type;
} Elf32_Nhdr;

typedef struct {
  Elf64_Word n_namesz;
  Elf64_Word n_descsz;
  Elf64_Word n_type;
} Elf64_Nhdr;

#undef Elf32_auxv_t
#undef Elf64_auxv_t

typedef struct {
    uint32_t a_type;
    union {
      uint32_t a_val;
    } a_un;
} Elf32_auxv_t;

typedef struct {
    uint64_t a_type;
    union {
      uint64_t a_val;
    } a_un;
} Elf64_auxv_t;

#undef Elf32_Dyn
#undef Elf64_Dyn

typedef struct {
  Elf32_Sword   d_tag;
  union {
    Elf32_Word  d_val;
    Elf32_Addr  d_ptr;
  } d_un;
} Elf32_Dyn;

typedef struct {
  Elf64_Sxword   d_tag;
  union {
    Elf64_Xword  d_val;
    Elf64_Addr   d_ptr;
  } d_un;
} Elf64_Dyn;


// The Android headers don't always define this constant.
#ifndef EM_X86_64
#define EM_X86_64  62
#endif

#ifndef EM_PPC64
#define EM_PPC64   21
#endif

#ifndef EM_S390
#define EM_S390    22
#endif

#if !defined(AT_SYSINFO_EHDR)
#define AT_SYSINFO_EHDR 33
#endif

#if !defined(NT_PRSTATUS)
#define NT_PRSTATUS 1
#endif

#if !defined(NT_PRPSINFO)
#define NT_PRPSINFO 3
#endif

#if !defined(NT_AUXV)
#define NT_AUXV   6
#endif

#if !defined(NT_PRXFPREG)
#define NT_PRXFPREG 0x46e62b7f
#endif

#if !defined(NT_FPREGSET)
#define NT_FPREGSET 2
#endif

#if !defined(SHT_MIPS_DWARF)
#define SHT_MIPS_DWARF 0x7000001e
#endif

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // GOOGLE_BREAKPAD_COMMON_ANDROID_INCLUDE_ELF_H
